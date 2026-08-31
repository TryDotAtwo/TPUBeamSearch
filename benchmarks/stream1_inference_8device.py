"""Inference-only exact Pallas/JAX comparison on one and eight TPU devices."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from benchmarks.diagnostic_timing import diagnostic_profile
from benchmarks.execution_boundary_ops import candidate_full, mismatch_witnesses
from benchmarks.layernorm_quality import (
    inverse_valid_mask, load_puzzle, make_legal_scrambles, tensor_metrics,
)
from benchmarks.stream1_layernorm_arithmetic import (
    checkpoint, quality, runtime_inventory, runtime_params, sha256_file,
)
from benchmarks.stream1_layernorm_followup import measure_comparison_group
from tpu_beam_search.sharding import make_sharded_inference
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_embedding_experimental import prepare_banked_embedding
from tpu_beam_search.stream1_layernorm_reference import (
    layernorm_stream1_weights_from_artgor_params,
    stream1_layernorm_reference_inference,
)


TARGET_DEVICE_COUNT = 8
ORIGINAL_ID = "original_runtime_jax"


def candidate_configs():
    """Static experiment matrix; every candidate retains the JAX ResMLP."""
    base = dict(network="unchanged_jax_resmlp", dense="jax", norm="jax")
    configs = [
        dict(base, id=ORIGINAL_ID, implementation="original_runtime_jax",
             control=True, bm=None, bank_dtype=None),
        dict(base, id="typed_runtime_jax", implementation="typed_runtime_jax",
             control=True, bm=None, bank_dtype=None),
        dict(base, id="jax_tiled", implementation="jax_tiled",
             control=False, bm=128, bank_dtype=None),
        dict(base, id="pallas_runtime_bm128", implementation="pallas_runtime",
             control=False, bm=128, bank_dtype="float32"),
    ]
    for bm in (64, 128, 256, 512, 1024, 2048):
        for dtype in ("bfloat16", "float32"):
            configs.append(dict(
                base,
                id=f"pallas_prepacked_bm{bm}_{dtype}",
                implementation="pallas_prepacked",
                control=False,
                bm=bm,
                bank_dtype=dtype,
            ))
    return configs


def _eligible(row, *, device_count, local_batch):
    median = row.get("timing", {}).get("median_ms")
    return (
        row.get("status") == "ok"
        and row.get("device_count") == device_count
        and row.get("local_batch") == local_batch
        and row.get("global_batch") == device_count * local_batch
        and row.get("timing_comparable") is True
        and row.get("exact_oracle_on_sample") is True
        and isinstance(median, (int, float))
        and not isinstance(median, bool)
        and math.isfinite(median)
        and median > 0
    )


def select_eight_device_winner(rows, *, corpus_names, local_batch):
    """Select only an exact candidate that beats original JAX per corpus."""
    names = tuple(corpus_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("corpus_names must be unique and nonempty")
    baseline = {
        row["corpus"]: row for row in rows
        if row.get("id") == ORIGINAL_ID
        and row.get("corpus") in names
        and _eligible(row, device_count=TARGET_DEVICE_COUNT, local_batch=local_batch)
    }
    if set(baseline) != set(names):
        return dict(winner_id=None, target_achieved=False, per_corpus_speedup={})

    identifiers = sorted({
        row.get("id") for row in rows
        if row.get("id") not in (None, ORIGINAL_ID) and not row.get("control", False)
    })
    accepted = []
    for identifier in identifiers:
        matching = {
            row["corpus"]: row for row in rows
            if row.get("id") == identifier
            and row.get("corpus") in names
            and _eligible(row, device_count=TARGET_DEVICE_COUNT, local_batch=local_batch)
        }
        if set(matching) != set(names):
            continue
        speedups = {
            name: baseline[name]["timing"]["median_ms"]
            / matching[name]["timing"]["median_ms"]
            for name in names
        }
        if not all(value > 1.0 for value in speedups.values()):
            continue
        accepted.append((statistics.mean(speedups.values()), identifier, speedups))
    if not accepted:
        return dict(winner_id=None, target_achieved=False, per_corpus_speedup={})
    _, identifier, speedups = max(accepted)
    return dict(winner_id=identifier, target_achieved=True,
                per_corpus_speedup=speedups)


def weak_scaling(one_device, eight_device):
    """Compute fixed-local-batch weak-scaling throughput and efficiency."""
    if one_device.get("device_count") != 1 or eight_device.get("device_count") != 8:
        raise ValueError("weak scaling requires one-device and eight-device rows")
    if one_device.get("local_batch") != eight_device.get("local_batch"):
        raise ValueError("weak scaling requires the same local batch")
    local_batch = one_device["local_batch"]
    if (one_device.get("global_batch") != local_batch
            or eight_device.get("global_batch") != 8 * local_batch):
        raise ValueError("global batch must equal device_count times local batch")
    one_ms = one_device["timing"]["median_ms"]
    eight_ms = eight_device["timing"]["median_ms"]
    if any(not isinstance(v, (int, float)) or isinstance(v, bool)
           or not math.isfinite(v) or v <= 0 for v in (one_ms, eight_ms)):
        raise ValueError("latencies must be finite positive numbers")
    one_throughput = local_batch * 1000.0 / one_ms
    eight_throughput = 8 * local_batch * 1000.0 / eight_ms
    speedup = eight_throughput / one_throughput
    return dict(
        one_device_states_per_second=one_throughput,
        eight_device_states_per_second=eight_throughput,
        throughput_speedup=speedup,
        parallel_efficiency=speedup / 8,
    )


def _tree_bytes(tree):
    return int(sum(value.size * value.dtype.itemsize for value in jax.tree.leaves(tree)))


def _array_sha256(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _compiled_call(mapped, arguments, directory, identifier):
    """Lower/compile once, retain both IR forms, then run one synchronized call."""
    jax.block_until_ready(arguments)
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    lowered = mapped.lower(*arguments)
    lowering_s = time.perf_counter() - start
    (directory / f"{identifier}.stablehlo.txt").write_text(
        str(lowered.compiler_ir(dialect="stablehlo")), encoding="utf-8"
    )
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - start
    (directory / f"{identifier}.compiled.txt").write_text(
        compiled.as_text(), encoding="utf-8"
    )
    start = time.perf_counter()
    output = jax.block_until_ready(compiled(*arguments))
    compilation = dict(
        lowering_s=lowering_s,
        compile_s=compile_s,
        first_execution_s=time.perf_counter() - start,
    )
    analysis = compiled.memory_analysis()
    if analysis is not None:
        compilation["static_memory_bytes_not_hardware_counters"] = {
            name: int(getattr(analysis, name))
            for name in (
                "argument_size_in_bytes", "output_size_in_bytes",
                "temp_size_in_bytes", "alias_size_in_bytes",
            )
        }
    return output, compiled, compilation


def _prepared_configs(params, original_apply, architecture, weights, configs,
                      *, interpret=False):
    payload, metadata = runtime_params(params)
    runtime_weights = weights._replace(
        embedding=jnp.asarray(params["embed"], jnp.float32)
    )
    bank_cache = {}
    banked_weight_cache = {}
    prepared = {}
    preparation = {}
    for config in configs:
        identifier = config["id"]
        implementation = config["implementation"]
        started = time.perf_counter()
        if implementation == "original_runtime_jax":
            call = lambda states, model: original_apply(
                {**metadata, **model}, states, dtype=jnp.bfloat16
            )
            model = payload
        elif implementation == "typed_runtime_jax":
            call = lambda states, model: stream1_layernorm_reference_inference(
                states, model, architecture
            )
            model = weights
        elif implementation in ("jax_tiled", "pallas_runtime"):
            embedding = "jax_tiled" if implementation == "jax_tiled" else "pallas_banked"
            full_config = {**config, "embedding": embedding}
            call = candidate_full(full_config, architecture, interpret=interpret)
            model = runtime_weights
        elif implementation == "pallas_prepacked":
            dtype_name = config["bank_dtype"]
            if dtype_name not in bank_cache:
                dtype = jnp.bfloat16 if dtype_name == "bfloat16" else jnp.float32
                banks = prepare_banked_embedding(
                    runtime_weights.embedding, storage_dtype=dtype
                )
                jax.block_until_ready(banks)
                bank_cache[dtype_name] = banks
            banks = bank_cache[dtype_name]
            if dtype_name not in banked_weight_cache:
                banked_weight_cache[dtype_name] = runtime_weights._replace(
                    embedding=banks
                )
            full_config = {**config, "embedding": "pallas_banked_prepacked"}
            call = candidate_full(full_config, architecture, interpret=interpret)
            model = banked_weight_cache[dtype_name]
        else:
            raise ValueError(f"unknown implementation: {implementation}")
        jax.block_until_ready(model)
        prepared[identifier] = (call, model)
        row = dict(
            id=identifier,
            implementation=implementation,
            one_time_preparation_s=time.perf_counter() - started,
            dynamic_argument_bytes=_tree_bytes(model),
        )
        if implementation == "pallas_prepacked":
            row.update(
                bank_dtype=config["bank_dtype"],
                bank_shapes=[list(banks.low.shape), list(banks.high.shape)],
                bank_sha256={"low": _array_sha256(banks.low),
                             "high": _array_sha256(banks.high)},
                bank_bytes=_tree_bytes(banks),
                conversion_order="checkpoint FP32 -> BF16 logical values -> selected bank storage",
            )
        preparation[identifier] = row
    return prepared, preparation


def _exact_ids(rows, configs, *, device_count, local_batch, corpus_names):
    names = set(corpus_names)
    result = []
    for config in configs:
        matching = [
            row for row in rows
            if row.get("id") == config["id"]
            and row.get("device_count") == device_count
            and row.get("local_batch") == local_batch
        ]
        if (len(matching) == len(names)
                and {row.get("corpus") for row in matching} == names
                and all(row.get("exact_oracle_on_sample") is True
                        and row.get("status") == "ok" for row in matching)):
            result.append(config["id"])
    return result


def run_suite(params, original_apply, architecture, weights, corpora, last_moves,
              inverse, directory, *, configs=None, screen_local_batch=16384,
              confirmation_local_batch=32768, device_counts=(1, 8),
              warmups=5, repeats=12, interpret=False, context=None):
    """Run exact full inference only; no move expansion, top-k, or beam state."""
    configs = candidate_configs() if configs is None else list(configs)
    corpus_names = tuple(corpora)
    if not corpus_names or set(corpus_names) != set(last_moves):
        raise ValueError("corpora and last_moves must have identical nonempty keys")
    if not configs or len({config["id"] for config in configs}) != len(configs):
        raise ValueError("config IDs must be unique and nonempty")
    if configs[0]["id"] != ORIGINAL_ID:
        raise ValueError("original_runtime_jax must be the first config")
    counts = (screen_local_batch, confirmation_local_batch, warmups, repeats)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
           for value in counts):
        raise ValueError("batch and timing counts must be positive integers")
    device_counts = tuple(device_counts)
    if (not device_counts or len(set(device_counts)) != len(device_counts)
            or any(value not in (1, 8) for value in device_counts)):
        raise ValueError("device_counts must be a unique subset of (1,8)")
    devices = jax.devices()
    if max(device_counts) > len(devices):
        raise RuntimeError(f"need {max(device_counts)} devices, found {len(devices)}")
    needed = max(screen_local_batch, confirmation_local_batch) * max(device_counts)
    for name, values in corpora.items():
        array = np.asarray(values)
        if (array.ndim != 2 or array.dtype != np.uint8
                or array.shape[1] != architecture.STATE_STORAGE_LEN
                or array.shape[0] < needed
                or np.any(array[:, :architecture.STATE_LEN] >= architecture.NUM_CLASSES)
                or len(last_moves[name]) != len(array)):
            raise ValueError(f"invalid corpus contract: {name}")

    directory = Path(directory)
    result_path = directory / "stream1_inference_8device.json"
    if result_path.exists():
        raise FileExistsError("use a new output directory")
    report = dict(
        status="running",
        context=context or {},
        architecture=asdict(architecture),
        protocol=dict(
            scope="full Q inference only; no beam-search stages",
            screen_local_batch=screen_local_batch,
            confirmation_local_batch=confirmation_local_batch,
            device_counts=list(device_counts),
            warmups=warmups,
            repeats=repeats,
            timing="paired forward/reverse, already compiled, synchronized global output",
            scaling="weak scaling with fixed local batch; global batch=device_count*local_batch",
            acceptance="elementwise exact finite Q on every corpus and faster than original JAX per corpus",
            weights="runtime replicated arguments; prepacked banks prepared once outside steady calls",
            output_semantics="MOVE_COUNT scores per state; lower Q is better",
        ),
        configurations=configs,
        preparation=[],
        input_scopes=[],
        measurements=[],
        timing_groups=[],
        winner_quality=[],
        profiles=[],
        weak_scaling=[],
        decision=dict(winner_id=None, target_achieved=False,
                      per_corpus_speedup={}),
    )
    checkpoint(result_path, report)
    host_outputs = {}
    profile_cases = {}

    try:
        prepared, preparation = _prepared_configs(
            params, original_apply, architecture, weights, configs,
            interpret=interpret,
        )
        report["preparation"] = [preparation[config["id"]] for config in configs]
        checkpoint(result_path, report)

        def execute(section, device_count, local_batch, selected):
            mesh = Mesh(np.asarray(devices[:device_count]), ("core",))
            state_sharding = NamedSharding(mesh, P("core", None))
            replicated = NamedSharding(mesh, P())
            global_batch = device_count * local_batch
            model_arg_cache = {}
            for corpus in corpus_names:
                host_states = np.asarray(corpora[corpus][:global_batch])
                report["input_scopes"].append(dict(
                    section=section, corpus=corpus,
                    device_count=device_count, local_batch=local_batch,
                    global_batch=global_batch,
                    global_input_sha256=hashlib.sha256(host_states.tobytes()).hexdigest(),
                    shard_input_sha256=[
                        hashlib.sha256(
                            host_states[index * local_batch:(index + 1) * local_batch].tobytes()
                        ).hexdigest()
                        for index in range(device_count)
                    ],
                ))
                states = jax.device_put(host_states, state_sharding)
                cases = {}
                rows = {}
                outputs = {}
                for config in selected:
                    identifier = config["id"]
                    row = dict(
                        section=section, id=identifier, corpus=corpus,
                        status="running", control=bool(config["control"]),
                        config=config, device_count=device_count,
                        local_batch=local_batch, global_batch=global_batch,
                    )
                    report["measurements"].append(row)
                    checkpoint(result_path, report)
                    print(
                        f"START {section} devices={device_count} batch={local_batch} "
                        f"corpus={corpus} id={identifier}", flush=True,
                    )
                    try:
                        local_call, model = prepared[identifier]
                        model_key = id(model)
                        if model_key not in model_arg_cache:
                            model_arg_cache[model_key] = jax.tree.map(
                                lambda value: jax.device_put(value, replicated),
                                model,
                            )
                        model_arg = model_arg_cache[model_key]
                        mapped = make_sharded_inference(
                            local_call, mesh=mesh, weights_example=model
                        )
                        hlo_id = f"{section}-d{device_count}-b{local_batch}-{corpus}-{identifier}"
                        output, compiled, compilation = _compiled_call(
                            mapped, (states, model_arg), directory / "hlo", hlo_id
                        )
                        host = np.asarray(output)
                        outputs[identifier] = host
                        cases[identifier] = (compiled, (states, model_arg))
                        host_outputs[(section, device_count, local_batch,
                                      corpus, identifier)] = host
                        if identifier == ORIGINAL_ID:
                            comparison = tensor_metrics(host, host)
                            witnesses = mismatch_witnesses(host, host)
                        else:
                            if ORIGINAL_ID not in outputs:
                                raise RuntimeError("original JAX oracle must compile first")
                            comparison = tensor_metrics(outputs[ORIGINAL_ID], host)
                            witnesses = mismatch_witnesses(outputs[ORIGINAL_ID], host)
                        exact = bool(
                            comparison["finite"]
                            and comparison["exact_fraction"] == 1.0
                        )
                        row.update(
                            status="ok", compilation=compilation,
                            comparison_vs_original=comparison,
                            mismatch_witnesses=witnesses,
                            exact_oracle_on_sample=exact,
                            argmin_agreement=float(np.mean(
                                np.argmin(outputs[ORIGINAL_ID], axis=1)
                                == np.argmin(host, axis=1)
                            )),
                            output_sha256=_array_sha256(host),
                        )
                        rows[identifier] = row
                    except Exception as exc:
                        row.update(
                            status="error", error_type=type(exc).__name__,
                            error=str(exc), traceback=traceback.format_exc(),
                            exact_oracle_on_sample=False,
                        )
                        print(row["traceback"], flush=True)
                    checkpoint(result_path, report)

                group = dict(
                    section=section, corpus=corpus,
                    device_count=device_count, local_batch=local_batch,
                    global_batch=global_batch,
                    **measure_comparison_group(
                        cases, warmups=warmups, repeats=repeats
                    ),
                )
                report["timing_groups"].append(group)
                for identifier, row in rows.items():
                    row["timing_comparable"] = group["comparison_valid"]
                    row["timing_label"] = group["label"]
                    if identifier in group["cases"]:
                        row["timing"] = group["cases"][identifier]
                        median_ms = row["timing"]["median_ms"]
                        row["states_per_second"] = (
                            global_batch * 1000.0 / median_ms
                        )
                    if (section == "target" and device_count == 8
                            and corpus == corpus_names[0]):
                        profile_cases[identifier] = cases[identifier]
                checkpoint(result_path, report)

        execute("screen", 1, screen_local_batch, configs)
        exact_screen = set(_exact_ids(
            report["measurements"], configs, device_count=1,
            local_batch=screen_local_batch, corpus_names=corpus_names,
        ))
        target_configs = [
            config for config in configs
            if config["control"] or config["id"] in exact_screen
        ]
        if 8 in device_counts:
            execute("target", 8, screen_local_batch, target_configs)
            target_rows = [
                row for row in report["measurements"]
                if row["section"] == "target"
            ]
            report["decision"] = select_eight_device_winner(
                target_rows, corpus_names=corpus_names,
                local_batch=screen_local_batch,
            )
        winner_id = report["decision"]["winner_id"]

        if winner_id is not None:
            chosen = [
                config for config in configs
                if config["id"] in (ORIGINAL_ID, "typed_runtime_jax", winner_id)
            ]
            for device_count in device_counts:
                execute(
                    "confirmation", device_count,
                    confirmation_local_batch, chosen,
                )
            if 8 in device_counts:
                confirmed = select_eight_device_winner(
                    [row for row in report["measurements"]
                     if row["section"] == "confirmation"],
                    corpus_names=corpus_names,
                    local_batch=confirmation_local_batch,
                )
                report["decision"]["confirmation"] = confirmed
                report["decision"]["target_achieved"] = bool(
                    confirmed["target_achieved"]
                    and confirmed["winner_id"] == winner_id
                )

            for section, batch in (("target", screen_local_batch),
                                   ("confirmation", confirmation_local_batch)):
                for device_count in (8,):
                    if device_count not in device_counts:
                        continue
                    for corpus in corpus_names:
                        key = (section, device_count, batch, corpus)
                        reference = host_outputs.get((*key, ORIGINAL_ID))
                        candidate = host_outputs.get((*key, winner_id))
                        if reference is None or candidate is None:
                            continue
                        mask = inverse_valid_mask(
                            np.asarray(last_moves[corpus][:device_count * batch]),
                            np.asarray(inverse),
                        )
                        report["winner_quality"].append(dict(
                            section=section, device_count=device_count,
                            local_batch=batch, global_batch=device_count * batch,
                            corpus=corpus, winner_id=winner_id,
                            q=quality(reference, candidate, mask),
                        ))

            if 1 in device_counts and 8 in device_counts:
                for identifier in (ORIGINAL_ID, winner_id):
                    for corpus in corpus_names:
                        for one_section, eight_section, batch, label in (
                            ("screen", "target", screen_local_batch, "target"),
                            ("confirmation", "confirmation",
                             confirmation_local_batch, "confirmation"),
                        ):
                            matches = {}
                            for row in report["measurements"]:
                                expected_section = (
                                    one_section if row["device_count"] == 1
                                    else eight_section
                                )
                                if (row["section"] == expected_section
                                        and row["id"] == identifier
                                        and row["corpus"] == corpus
                                        and row["local_batch"] == batch
                                        and row.get("timing_comparable") is True):
                                    matches[row["device_count"]] = row
                            if set(matches) == {1, 8}:
                                report["weak_scaling"].append(dict(
                                    section=label, id=identifier,
                                    corpus=corpus, local_batch=batch,
                                    **weak_scaling(matches[1], matches[8]),
                                ))

            if not interpret:
                for identifier in (ORIGINAL_ID, winner_id):
                    if identifier not in profile_cases:
                        continue
                    compiled, arguments = profile_cases[identifier]
                    try:
                        report["profiles"].append(dict(
                            id=identifier,
                            **diagnostic_profile(
                                compiled, *arguments,
                                directory=directory / "profiles" / identifier,
                                iterations=3,
                            ),
                        ))
                    except Exception as exc:
                        report["profiles"].append(dict(
                            id=identifier, status="error",
                            error_type=type(exc).__name__, error=str(exc),
                        ))

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
        default=Path("/kaggle/working/inference_8device"),
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
    inventory["experiment_active_device_counts"] = [1, 8]
    sys.path.insert(0, str(dataset))
    from jax_model import apply as original_apply, load_params_from_pt

    checkpoint_path = dataset / "q555_2k_BEST.pt"
    with jax.default_device(jax.local_devices()[0]):
        params = load_params_from_pt(checkpoint_path)
        architecture = Stream1Architecture.from_artgor_params(
            params, STATE_STORAGE_LEN=int(params["state_size"])
        )
        contract = (
            architecture.STATE_LEN, architecture.NUM_CLASSES,
            architecture.EMBED_DIM, architecture.HIDDEN1,
            architecture.RESIDUAL_COUNT, architecture.MOVE_COUNT,
        )
        if contract != (150, 150, 24, 1024, 10, 30):
            raise ValueError("checkpoint is not the agreed Artgor Q ResMLP")
        weights = layernorm_stream1_weights_from_artgor_params(
            params, architecture
        )
        puzzle = load_puzzle(
            dataset / "puzzle_info.json", state_len=architecture.STATE_LEN,
            move_count=architecture.MOVE_COUNT,
        )
        max_rows = 32768 * 8
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
                ("git", "rev-parse", "HEAD"), text=True
            ).strip(),
            runtime=inventory,
            checkpoint_sha256=sha256_file(checkpoint_path),
            original_source_sha256=sha256_file(dataset / "jax_model.py"),
            puzzle_sha256=puzzle.sha256,
            input_sha256={
                name: hashlib.sha256(value.tobytes()).hexdigest()
                for name, value in corpora.items()
            },
            input_prefix_32768_sha256={
                name: hashlib.sha256(value[:32768].tobytes()).hexdigest()
                for name, value in corpora.items()
            },
            seeds=dict(legal=42, stress=43),
            input_method=(
                "independent fixed local shards from legal random walks and "
                "categorical stress; not beam frontiers"
            ),
        )
        print(json.dumps(context, indent=2), flush=True)
        report = run_suite(
            params, original_apply, architecture, weights, corpora,
            dict(
                legal_scrambles=legal.last_moves,
                categorical_stress=np.full(max_rows, -1, np.int32),
            ),
            puzzle.inverse, args.output, context=context,
        )
        print("RESULT", json.dumps(report["decision"], allow_nan=False), flush=True)
        print("RESULT_PATH", args.output / "stream1_inference_8device.json", flush=True)


if __name__ == "__main__":
    main()
