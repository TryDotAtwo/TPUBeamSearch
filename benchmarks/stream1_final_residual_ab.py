"""Restore exact eight-TPU Q by targeting the final residual Dense schedule."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from benchmarks.execution_boundary_ops import mismatch_witnesses
from benchmarks.final_residual_ops import (
    FINAL_BARRIERS, FINAL_CUTS, FINAL_TAPS,
    candidate_final_full, candidate_final_partition,
)
from benchmarks.layernorm_quality import load_puzzle, make_legal_scrambles, tensor_metrics
from benchmarks.stream1_inference_execution_ab import (
    ORACLE_ID, TARGET_DEVICE_COUNT, _array_sha256, _compile,
    _measure_with_failure_record, _prepared_models, select_execution_winner,
)
from benchmarks.stream1_layernorm_arithmetic import (
    checkpoint, runtime_inventory, sha256_file,
)
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_layernorm_reference import (
    layernorm_stream1_weights_from_artgor_params,
)


def final_residual_configs():
    common = dict(
        bm=2048, dense="jax", norm="jax", boundary="none",
        input_boundary="none", final_barrier="none",
    )
    configs = [
        dict(common, id=ORACLE_ID, role="jax_control", backend="monolithic",
             implementation="original", embedding="original"),
        dict(common, id="typed_monolithic", role="jax_control", backend="monolithic",
             implementation="typed", embedding="reference"),
        dict(common, id="pallas_monolithic", role="candidate", backend="monolithic",
             implementation="pallas", embedding="pallas_banked_prepacked",
             paired_control="typed_monolithic"),
    ]
    for barrier in FINAL_BARRIERS:
        if barrier == "none":
            continue
        configs.append(dict(
            common, id=f"pallas_barrier_{barrier}", role="candidate",
            backend="barrier", implementation="pallas",
            embedding="pallas_banked_prepacked", final_barrier=barrier,
            paired_control="typed_monolithic",
        ))
    for tap in FINAL_TAPS:
        configs.extend([
            dict(common, id=f"typed_tap_{tap}", role="jax_control", backend="tap",
                 implementation="typed", embedding="reference", tap=tap),
            dict(common, id=f"pallas_tap_{tap}", role="candidate", backend="tap",
                 implementation="pallas", embedding="pallas_banked_prepacked",
                 tap=tap, paired_control=f"typed_tap_{tap}"),
        ])
    for cut in FINAL_CUTS:
        configs.extend([
            dict(common, id=f"typed_split_{cut}", role="jax_control", backend="split",
                 implementation="typed", embedding="reference", cut=cut),
            dict(common, id=f"pallas_split_{cut}", role="candidate", backend="split",
                 implementation="pallas", embedding="pallas_banked_prepacked",
                 cut=cut, paired_control=f"typed_split_{cut}"),
        ])
    return configs


def output_specs_for_config(config, *, stage="full"):
    sharded = P("core", None)
    backend = config["backend"]
    if backend in ("monolithic", "barrier"):
        return sharded
    if backend == "tap":
        return sharded, sharded
    if backend == "split":
        if stage == "suffix":
            return sharded
        if stage != "prefix":
            raise ValueError("split output specs require prefix or suffix stage")
        if config["cut"] in ("before_final_dense2", "after_final_dense2"):
            return sharded, sharded
        return sharded
    raise ValueError(f"unknown backend: {backend}")


def _mapped(call, *, mesh, weights_example, input_specs, output_specs):
    weight_specs = jax.tree.map(lambda _: P(), weights_example)
    return jax.jit(jax.shard_map(
        call,
        mesh=mesh,
        in_specs=(input_specs, weight_specs),
        out_specs=output_specs,
        check_vma=False,
    ))


def _model_for(config, prepared):
    implementation = config["implementation"]
    if implementation not in ("original", "typed", "pallas"):
        raise ValueError(f"unknown implementation: {implementation}")
    return prepared[implementation][1]


def _full_call_for(config, prepared, architecture, *, interpret=False):
    implementation = config["implementation"]
    if implementation == "original":
        return prepared["original"][0]
    if implementation == "typed" and config["backend"] == "monolithic":
        return prepared["typed"][0]
    return candidate_final_full(
        config, architecture, tap=config.get("tap"), interpret=interpret,
    )


def _build_case(config, *, prepared, architecture, states, mesh,
                replicated_cache, directory, prefix, interpret=False):
    model = _model_for(config, prepared)
    replicated = NamedSharding(mesh, P())
    key = id(model)
    if key not in replicated_cache:
        replicated_cache[key] = jax.tree.map(
            lambda leaf: jax.device_put(leaf, replicated), model,
        )
        jax.block_until_ready(replicated_cache[key])
    model_arg = replicated_cache[key]
    backend = config["backend"]

    if backend != "split":
        call = _full_call_for(
            config, prepared, architecture, interpret=interpret,
        )
        mapped = _mapped(
            call, mesh=mesh, weights_example=model,
            input_specs=P("core", None),
            output_specs=output_specs_for_config(config),
        )
        args = states, model_arg
        compiled, output, compilation = _compile(
            mapped, args, directory, prefix,
        )
        if backend == "tap":
            host_output = lambda value: np.asarray(value[0]).reshape(
                -1, architecture.MOVE_COUNT,
            )
            orchestration = (
                "one shard_map dispatch returning Q and one final-block BF16 tap"
            )
        else:
            host_output = lambda value: np.asarray(value).reshape(
                -1, architecture.MOVE_COUNT,
            )
            orchestration = "one shard_map dispatch"
        return dict(
            runner=lambda: compiled(*args), initial_output=output,
            host_output=host_output, compilation=compilation,
            orchestration=orchestration,
        )

    prefix_call, suffix_call = candidate_final_partition(
        config, architecture, cut=config["cut"], interpret=interpret,
    )
    intermediate_specs = output_specs_for_config(config, stage="prefix")
    prefix_mapped = _mapped(
        prefix_call, mesh=mesh, weights_example=model,
        input_specs=P("core", None), output_specs=intermediate_specs,
    )
    prefix_args = states, model_arg
    prefix_compiled, intermediate, prefix_info = _compile(
        prefix_mapped, prefix_args, directory, f"{prefix}-prefix",
    )
    suffix_mapped = _mapped(
        suffix_call, mesh=mesh, weights_example=model,
        input_specs=intermediate_specs,
        output_specs=output_specs_for_config(config, stage="suffix"),
    )
    suffix_args = intermediate, model_arg
    suffix_compiled, output, suffix_info = _compile(
        suffix_mapped, suffix_args, directory, f"{prefix}-suffix",
    )

    def runner():
        fresh = prefix_compiled(*prefix_args)
        return suffix_compiled(fresh, model_arg)

    return dict(
        runner=runner, initial_output=output,
        host_output=lambda value: np.asarray(value).reshape(
            -1, architecture.MOVE_COUNT,
        ),
        compilation=dict(prefix=prefix_info, suffix=suffix_info),
        orchestration=(
            f"two compiled shard_map dispatches with device-resident {config['cut']} boundary"
        ),
    )


def _execute_section(report, result_path, *, section, prepared, architecture,
                     corpora, configs, devices, local_batch, directory,
                     warmups, repeats, interpret=False):
    device_count = len(devices)
    global_batch = device_count * local_batch
    mesh = Mesh(np.asarray(devices), ("core",))
    state_sharding = NamedSharding(mesh, P("core", None))
    replicated_cache = {}
    hlo_dir = directory / "hlo"

    for corpus, all_states in corpora.items():
        host_states = np.asarray(all_states[:global_batch])
        states = jax.device_put(host_states, state_sharding)
        report["input_scopes"].append(dict(
            section=section, corpus=corpus, device_count=device_count,
            local_batch=local_batch, global_batch=global_batch,
            global_input_sha256=hashlib.sha256(host_states.tobytes()).hexdigest(),
            shard_input_sha256=[
                hashlib.sha256(
                    host_states[index * local_batch:(index + 1) * local_batch].tobytes()
                ).hexdigest()
                for index in range(device_count)
            ],
        ))
        runners = {}
        rows = {}
        oracle = None
        for config in configs:
            identifier = config["id"]
            row = dict(
                section=section, corpus=corpus, id=identifier,
                role=config["role"], config=config, status="running",
                device_count=device_count, local_batch=local_batch,
                global_batch=global_batch,
            )
            report["measurements"].append(row)
            checkpoint(result_path, report)
            print(
                f"START {section} corpus={corpus} id={identifier} "
                f"devices={device_count} local_batch={local_batch}",
                flush=True,
            )
            try:
                case = _build_case(
                    config, prepared=prepared, architecture=architecture,
                    states=states, mesh=mesh, replicated_cache=replicated_cache,
                    directory=hlo_dir,
                    prefix=f"{section}-d{device_count}-b{local_batch}-{corpus}-{identifier}",
                    interpret=interpret,
                )
                host = case["host_output"](case["initial_output"])
                if identifier == ORACLE_ID:
                    oracle = host
                elif oracle is None:
                    raise RuntimeError("canonical original shard_map must compile first")
                reference = host if identifier == ORACLE_ID else oracle
                metrics = tensor_metrics(reference, host)
                exact = bool(metrics["finite"] and metrics["exact_fraction"] == 1.0)
                row.update(
                    status="ok", compilation=case["compilation"],
                    orchestration=case["orchestration"],
                    comparison_vs_original=metrics,
                    mismatch_witnesses=mismatch_witnesses(reference, host),
                    exact_oracle_on_sample=exact,
                    argmin_agreement=float(np.mean(
                        np.argmin(reference, axis=1) == np.argmin(host, axis=1)
                    )),
                    output_sha256=_array_sha256(host),
                )
                runners[identifier] = case["runner"]
                rows[identifier] = row
            except Exception as exc:
                row.update(
                    status="error", error_type=type(exc).__name__,
                    error=str(exc), traceback=traceback.format_exc(),
                    exact_oracle_on_sample=False,
                )
                print(row["traceback"], flush=True)
                if identifier == ORACLE_ID:
                    raise
            checkpoint(result_path, report)

        timing = _measure_with_failure_record(
            runners, warmups=warmups, repeats=repeats,
        )
        timing.update(
            section=section, corpus=corpus, device_count=device_count,
            local_batch=local_batch, global_batch=global_batch,
        )
        report["timing_groups"].append(timing)
        for identifier, row in rows.items():
            row["timing_comparable"] = timing["comparison_valid"]
            row["timing_label"] = timing["label"]
            if identifier in timing["cases"]:
                row["timing"] = timing["cases"][identifier]
                row["states_per_second"] = (
                    global_batch * 1000.0 / row["timing"]["median_ms"]
                )
        checkpoint(result_path, report)


def _confirmation_configs(configs, decision):
    selected = {
        ORACLE_ID, "typed_monolithic", decision["winner_id"],
        *decision.get("jax_baseline_id", {}).values(),
    }
    by_id = {config["id"]: config for config in configs}
    winner = by_id[decision["winner_id"]]
    if winner.get("paired_control"):
        selected.add(winner["paired_control"])
    return [config for config in configs if config["id"] in selected]


def run_suite(params, original_apply, architecture, weights, corpora, directory,
              *, configs=None, target_local_batch=16_384,
              confirmation_local_batch=32_768, warmups=5, repeats=12,
              interpret=False, context=None):
    configs = final_residual_configs() if configs is None else list(configs)
    if not configs or configs[0]["id"] != ORACLE_ID:
        raise ValueError("canonical original_shard_map must be first")
    if len({config["id"] for config in configs}) != len(configs):
        raise ValueError("configuration IDs must be unique")
    devices = jax.devices()[:TARGET_DEVICE_COUNT]
    if len(devices) != TARGET_DEVICE_COUNT:
        raise RuntimeError(f"need eight devices, found {len(devices)}")
    required = TARGET_DEVICE_COUNT * confirmation_local_batch
    for name, values in corpora.items():
        array = np.asarray(values)
        if (array.dtype != np.uint8 or array.ndim != 2
                or array.shape[1] != architecture.STATE_STORAGE_LEN
                or len(array) < required):
            raise ValueError(f"invalid corpus contract: {name}")

    directory = Path(directory)
    result_path = directory / "stream1_final_residual_ab.json"
    if result_path.exists():
        raise FileExistsError("use a new output directory")
    report = dict(
        status="running", context=context or {}, architecture=asdict(architecture),
        protocol=dict(
            scope="full Q inference only; no beam-search work",
            oracle="original jax_model.apply through eight-device shard_map",
            target_devices=TARGET_DEVICE_COUNT,
            target_local_batch=target_local_batch,
            confirmation_local_batch=confirmation_local_batch,
            warmups=warmups, repeats=repeats,
            attribution=(
                "target only final residual block 9 and its second Dense, the sole "
                "changed MXU schedule in execution A/B v1"
            ),
            tap=(
                "one dispatch returns Q plus one BF16 boundary tensor; both outputs "
                "are synchronized and timed"
            ),
            split=(
                "two compiled shard_map dispatches; intermediate stays device-resident"
            ),
            acceptance=(
                "finite elementwise exact original BF16 Q on legal and stress, faster "
                "than fastest exact JAX control at local batches 16K and 32K"
            ),
        ),
        configurations=configs, preparation={}, input_scopes=[], measurements=[],
        timing_groups=[],
        target_decision=dict(winner_id=None, target_achieved=False,
                             jax_baseline_id={}, per_corpus_speedup={}),
        confirmation_decision=None,
    )
    checkpoint(result_path, report)
    try:
        prepared = _prepared_models(
            params, original_apply, architecture, weights, interpret=interpret,
        )
        report["preparation"] = prepared["preparation"]
        checkpoint(result_path, report)
        _execute_section(
            report, result_path, section="target", prepared=prepared,
            architecture=architecture, corpora=corpora, configs=configs,
            devices=devices, local_batch=target_local_batch,
            directory=directory, warmups=warmups, repeats=repeats,
            interpret=interpret,
        )
        report["target_decision"] = select_execution_winner(
            [row for row in report["measurements"] if row["section"] == "target"],
            corpus_names=tuple(corpora),
        )
        checkpoint(result_path, report)
        winner = report["target_decision"]["winner_id"]
        if winner is not None:
            confirmation_configs = _confirmation_configs(
                configs, report["target_decision"],
            )
            _execute_section(
                report, result_path, section="confirmation", prepared=prepared,
                architecture=architecture, corpora=corpora,
                configs=confirmation_configs, devices=devices,
                local_batch=confirmation_local_batch, directory=directory,
                warmups=warmups, repeats=repeats, interpret=interpret,
            )
            confirmation = select_execution_winner(
                [row for row in report["measurements"]
                 if row["section"] == "confirmation"],
                corpus_names=tuple(corpora),
            )
            confirmation["same_winner_as_target"] = confirmation["winner_id"] == winner
            confirmation["target_achieved"] = bool(
                confirmation["target_achieved"]
                and confirmation["same_winner_as_target"]
            )
            report["confirmation_decision"] = confirmation
            report["target_decision"]["target_achieved"] = bool(
                report["target_decision"]["target_achieved"]
                and confirmation["target_achieved"]
            )
        report["status"] = "complete"
        report["error_count"] = sum(
            row.get("status") == "error" for row in report["measurements"]
        )
        checkpoint(result_path, report)
        return report
    except Exception as exc:
        report.update(
            status="error", fatal_error_type=type(exc).__name__,
            fatal_error=str(exc), fatal_traceback=traceback.format_exc(),
        )
        checkpoint(result_path, report)
        raise


def _dataset_path(path):
    if path is not None:
        return path
    return next((candidate for candidate in (
        Path("/kaggle/input/cube555-tpu-artifacts"),
        Path("/kaggle/input/datasets/artgor/cube555-tpu-artifacts"),
    ) if candidate.is_dir()), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/kaggle/working/final_residual_ab"),
    )
    args = parser.parse_args()
    dataset = _dataset_path(args.dataset)
    if dataset is None:
        raise FileNotFoundError("attach artgor/cube555-tpu-artifacts")
    devices = jax.devices()
    if len(devices) < 8 or any(device.platform != "tpu" for device in devices[:8]):
        raise RuntimeError(f"requires eight TPU devices, found: {devices}")
    inventory = runtime_inventory()
    inventory["active_device_count"] = 8
    sys.path.insert(0, str(dataset))
    from jax_model import apply as original_apply, load_params_from_pt

    checkpoint_path = dataset / "q555_2k_BEST.pt"
    with jax.default_device(jax.local_devices()[0]):
        params = load_params_from_pt(checkpoint_path)
        architecture = Stream1Architecture.from_artgor_params(
            params, STATE_STORAGE_LEN=int(params["state_size"]),
        )
        contract = (
            architecture.STATE_LEN, architecture.NUM_CLASSES,
            architecture.EMBED_DIM, architecture.HIDDEN1,
            architecture.RESIDUAL_COUNT, architecture.MOVE_COUNT,
        )
        if contract != (150, 150, 24, 1024, 10, 30):
            raise ValueError("checkpoint is not the agreed Artgor Q ResMLP")
        weights = layernorm_stream1_weights_from_artgor_params(params, architecture)
        puzzle = load_puzzle(
            dataset / "puzzle_info.json", state_len=architecture.STATE_LEN,
            move_count=architecture.MOVE_COUNT,
        )
        max_rows = TARGET_DEVICE_COUNT * 32_768
        legal = make_legal_scrambles(puzzle, batch=max_rows, seed=42)
        stress = np.random.default_rng(43).integers(
            0, architecture.NUM_CLASSES,
            (max_rows, architecture.STATE_STORAGE_LEN), dtype=np.uint8,
        )
        corpora = dict(
            legal_scrambles=legal.states,
            categorical_stress=stress,
        )
        context = dict(
            source_commit=subprocess.check_output(
                ("git", "rev-parse", "HEAD"), text=True,
            ).strip(),
            runtime=inventory,
            checkpoint_sha256=sha256_file(checkpoint_path),
            original_source_sha256=sha256_file(dataset / "jax_model.py"),
            puzzle_sha256=puzzle.sha256,
            input_sha256={
                name: hashlib.sha256(value.tobytes()).hexdigest()
                for name, value in corpora.items()
            },
            seeds=dict(legal=42, stress=43),
            execution_ab_source_commit="88d6e42c4100578aa9478d3faf6b4f5d30adc01f",
            execution_ab_result=(
                "Pallas values are exact; graph context changes the sole final "
                "residual Dense schedule and rare Q bits"
            ),
        )
        print(json.dumps(context, indent=2), flush=True)
        report = run_suite(
            params, original_apply, architecture, weights, corpora,
            args.output, context=context,
        )
        print("RESULT", json.dumps(report["target_decision"], allow_nan=False), flush=True)
        print("CONFIRMATION", json.dumps(
            report["confirmation_decision"], allow_nan=False,
        ), flush=True)
        print("RESULT_PATH", args.output / "stream1_final_residual_ab.json", flush=True)


if __name__ == "__main__":
    main()
