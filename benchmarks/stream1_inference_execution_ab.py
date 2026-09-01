"""Localize rare Q drift and compare exact eight-TPU inference launch modes."""
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

from benchmarks.execution_boundary_ops import (
    candidate_encode, candidate_full, candidate_nodes, candidate_tail,
    mismatch_witnesses,
)
from benchmarks.layernorm_quality import load_puzzle, make_legal_scrambles, tensor_metrics
from benchmarks.stream1_layernorm_arithmetic import (
    checkpoint, runtime_inventory, runtime_params, sha256_file,
)
from tpu_beam_search.sharding import make_sharded_inference
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_embedding_experimental import prepare_banked_embedding
from tpu_beam_search.stream1_layernorm_reference import (
    layernorm_stream1_weights_from_artgor_params,
    stream1_layernorm_reference_inference,
)


ORACLE_ID = "original_shard_map"
TARGET_DEVICE_COUNT = 8


def execution_configs():
    """Frozen bundled matrix; BM2048 is the measured v1 lookup winner."""
    common = dict(bm=2048, dense="jax", norm="jax", boundary="none")
    rows = [
        dict(common, id=ORACLE_ID, role="jax_control", backend="shard_map",
             implementation="original", embedding="original", input_boundary="none"),
        dict(common, id="typed_shard_map", role="jax_control", backend="shard_map",
             implementation="typed", embedding="reference", input_boundary="none"),
    ]
    for boundary in ("none", "pre", "post", "both"):
        rows.append(dict(
            common, id=f"pallas_shard_{boundary}", role="candidate",
            backend="shard_map", implementation="pallas",
            embedding="pallas_banked_prepacked", input_boundary=boundary,
        ))
    rows.extend([
        dict(common, id="jax_split", role="jax_control", backend="split",
             implementation="typed", embedding="reference", input_boundary="none"),
        dict(common, id="pallas_split", role="candidate", backend="split",
             implementation="pallas", embedding="pallas_banked_prepacked",
             input_boundary="none"),
        dict(common, id="original_direct_jit", role="jax_control", backend="direct_jit",
             implementation="original", embedding="original", input_boundary="none"),
        dict(common, id="pallas_direct_jit", role="candidate", backend="direct_jit",
             implementation="pallas", embedding="pallas_banked_prepacked",
             input_boundary="none"),
        dict(common, id="original_pmap", role="jax_control", backend="pmap",
             implementation="original", embedding="original", input_boundary="none"),
        dict(common, id="pallas_pmap", role="candidate", backend="pmap",
             implementation="pallas", embedding="pallas_banked_prepacked",
             input_boundary="none"),
        dict(common, id="original_independent", role="jax_control", backend="independent",
             implementation="original", embedding="original", input_boundary="none"),
        dict(common, id="pallas_independent", role="candidate", backend="independent",
             implementation="pallas", embedding="pallas_banked_prepacked",
             input_boundary="none"),
    ])
    return rows


def owner_local_index(global_index, *, local_batch, device_count):
    """Map a flattened global row to its contiguous device shard."""
    values = (global_index, local_batch, device_count)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise TypeError("indices and sizes must be integers")
    if global_index < 0 or local_batch <= 0 or device_count <= 0:
        raise ValueError("index must be nonnegative and sizes positive")
    if global_index >= local_batch * device_count:
        raise ValueError("global index lies outside the sharded batch")
    return divmod(global_index, local_batch)


def _eligible(row):
    timing = row.get("timing", {}).get("median_ms")
    return (
        row.get("status") == "ok"
        and row.get("timing_comparable") is True
        and row.get("exact_oracle_on_sample") is True
        and isinstance(timing, (int, float))
        and not isinstance(timing, bool)
        and math.isfinite(timing)
        and timing > 0
    )


def select_execution_winner(rows, *, corpus_names):
    """Require a candidate to beat the fastest exact JAX arm per corpus."""
    names = tuple(corpus_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("corpus_names must be unique and nonempty")
    baselines = {}
    baseline_ids = {}
    for corpus in names:
        controls = [
            row for row in rows
            if row.get("corpus") == corpus
            and row.get("role") == "jax_control"
            and _eligible(row)
        ]
        if not controls:
            return dict(winner_id=None, target_achieved=False,
                        jax_baseline_id={}, per_corpus_speedup={})
        best = min(controls, key=lambda row: row["timing"]["median_ms"])
        baselines[corpus] = best
        baseline_ids[corpus] = best["id"]

    candidate_ids = sorted({
        row.get("id") for row in rows if row.get("role") == "candidate"
    })
    accepted = []
    for identifier in candidate_ids:
        matching = {
            row["corpus"]: row for row in rows
            if row.get("id") == identifier and row.get("corpus") in names
            and _eligible(row)
        }
        if set(matching) != set(names):
            continue
        speedups = {
            corpus: baselines[corpus]["timing"]["median_ms"]
            / matching[corpus]["timing"]["median_ms"]
            for corpus in names
        }
        if all(speedup > 1 for speedup in speedups.values()):
            accepted.append((statistics.mean(speedups.values()), identifier, speedups))
    if not accepted:
        return dict(winner_id=None, target_achieved=False,
                    jax_baseline_id=baseline_ids, per_corpus_speedup={})
    _, winner, speedups = max(accepted)
    return dict(winner_id=winner, target_achieved=True,
                jax_baseline_id=baseline_ids, per_corpus_speedup=speedups)


def _array_sha256(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _tree_bytes(tree):
    return int(sum(value.size * value.dtype.itemsize for value in jax.tree.leaves(tree)))


def measure_runner_group(cases, *, warmups=5, repeats=12):
    """Paired synchronized wall time for compiled or multi-stage runners."""
    if (not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 1
            or not isinstance(repeats, int) or isinstance(repeats, bool)
            or repeats < 1):
        raise ValueError("warmups and repeats must be positive integers")
    if not cases or any(not callable(call) for call in cases.values()):
        raise ValueError("cases must be a nonempty callable mapping")
    names = list(cases)
    for repeat in range(warmups):
        order = names if repeat % 2 == 0 else reversed(names)
        for name in order:
            jax.block_until_ready(cases[name]())
    samples = {name: [] for name in names}
    execution_order = []
    for repeat in range(repeats):
        order = names[:] if repeat % 2 == 0 else list(reversed(names))
        execution_order.append(order)
        for name in order:
            started = time.perf_counter()
            jax.block_until_ready(cases[name]())
            samples[name].append((time.perf_counter() - started) * 1000.0)
    return dict(
        label="paired_interleaved_synchronized_runner_wall_time",
        warmups=warmups,
        repeats=repeats,
        execution_order=execution_order,
        cases={
            name: dict(
                samples_ms=values,
                median_ms=float(statistics.median(values)),
                min_ms=float(min(values)),
                max_ms=float(max(values)),
            )
            for name, values in samples.items()
        },
    )


def _compile(mapped, arguments, directory, identifier):
    directory.mkdir(parents=True, exist_ok=True)
    jax.block_until_ready(arguments)
    started = time.perf_counter()
    lowered = mapped.lower(*arguments)
    lowering_s = time.perf_counter() - started
    (directory / f"{identifier}.stablehlo.txt").write_text(
        str(lowered.compiler_ir(dialect="stablehlo")), encoding="utf-8"
    )
    started = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - started
    compiled_text = compiled.as_text()
    (directory / f"{identifier}.compiled.txt").write_text(
        compiled_text, encoding="utf-8"
    )
    started = time.perf_counter()
    output = jax.block_until_ready(compiled(*arguments))
    result = dict(
        lowering_s=lowering_s,
        compile_s=compile_s,
        first_execution_s=time.perf_counter() - started,
        compiled_hlo_sha256=hashlib.sha256(compiled_text.encode()).hexdigest(),
    )
    analysis = compiled.memory_analysis()
    if analysis is not None:
        result["static_memory_bytes_not_hardware_counters"] = {
            name: int(getattr(analysis, name))
            for name in (
                "argument_size_in_bytes", "output_size_in_bytes",
                "temp_size_in_bytes", "alias_size_in_bytes",
            )
        }
    return compiled, output, result


def _prepared_models(params, original_apply, architecture, weights,
                     *, interpret=False):
    payload, metadata = runtime_params(params)
    runtime_weights = weights._replace(
        embedding=jnp.asarray(params["embed"], jnp.float32)
    )
    banks = prepare_banked_embedding(
        runtime_weights.embedding, storage_dtype=jnp.float32
    )
    jax.block_until_ready(banks)
    banked_weights = runtime_weights._replace(embedding=banks)

    def original(states, model):
        return original_apply({**metadata, **model}, states, dtype=jnp.bfloat16)

    typed = lambda states, model: stream1_layernorm_reference_inference(
        states, model, architecture
    )
    return dict(
        original=(original, payload),
        typed=(typed, weights),
        pallas=(None, banked_weights),
        preparation=dict(
            payload_bytes=_tree_bytes(payload),
            typed_weight_bytes=_tree_bytes(weights),
            banked_weight_bytes=_tree_bytes(banked_weights),
            bank_shapes=[list(banks.low.shape), list(banks.high.shape)],
            bank_sha256=dict(low=_array_sha256(banks.low), high=_array_sha256(banks.high)),
            bank_dtype=str(banks.low.dtype),
            bank_conversion=(
                "checkpoint FP32 -> BF16 logical values -> FP32 bank storage"
            ),
            interpret=bool(interpret),
        ),
    )


def _local_call_and_model(config, prepared, architecture, *, interpret=False):
    implementation = config["implementation"]
    if implementation in ("original", "typed"):
        return prepared[implementation]
    if implementation != "pallas":
        raise ValueError(f"unknown implementation: {implementation}")
    return (
        candidate_full(config, architecture, interpret=interpret),
        prepared["pallas"][1],
    )


def _pmap_value(value, *, device_count, sharding):
    host = np.broadcast_to(
        np.asarray(value), (device_count, *value.shape)
    )
    return jax.device_put(host, sharding)


def _build_case(config, *, prepared, architecture, states, host_states,
                mesh, devices, replicated_cache, pmap_cache,
                independent_cache, directory, prefix, interpret=False):
    identifier = config["id"]
    backend = config["backend"]
    local_call, model = _local_call_and_model(
        config, prepared, architecture, interpret=interpret
    )
    device_count = len(devices)
    local_batch = len(host_states) // device_count
    replicated = NamedSharding(mesh, P())

    def replicated_model(value):
        key = id(value)
        if key not in replicated_cache:
            replicated_cache[key] = jax.tree.map(
                lambda leaf: jax.device_put(leaf, replicated), value
            )
            jax.block_until_ready(replicated_cache[key])
        return replicated_cache[key]

    if backend == "shard_map":
        model_arg = replicated_model(model)
        mapped = make_sharded_inference(
            local_call, mesh=mesh, weights_example=model
        )
        args = (states, model_arg)
        compiled, output, compilation = _compile(
            mapped, args, directory, prefix
        )
        return dict(
            runner=lambda: compiled(*args),
            initial_output=output,
            host_output=lambda value: np.asarray(value).reshape(-1, architecture.MOVE_COUNT),
            compilation=compilation,
            orchestration="one shard_map SPMD dispatch",
        )

    if backend == "split":
        model_arg = replicated_model(model)
        encode = candidate_encode(config, architecture, interpret=interpret)
        tail = candidate_tail(config, architecture, interpret=interpret)
        encode_mapped = make_sharded_inference(
            encode, mesh=mesh, weights_example=model
        )
        encode_args = (states, model_arg)
        encode_compiled, encoded, encode_info = _compile(
            encode_mapped, encode_args, directory, f"{prefix}-encode"
        )
        tail_mapped = make_sharded_inference(
            tail, mesh=mesh, weights_example=model
        )
        tail_args = (encoded, model_arg)
        tail_compiled, output, tail_info = _compile(
            tail_mapped, tail_args, directory, f"{prefix}-tail"
        )

        def runner():
            fresh_encoded = encode_compiled(*encode_args)
            return tail_compiled(fresh_encoded, model_arg)

        return dict(
            runner=runner,
            initial_output=output,
            host_output=lambda value: np.asarray(value).reshape(-1, architecture.MOVE_COUNT),
            compilation=dict(encode=encode_info, tail=tail_info),
            orchestration="two compiled shard_map dispatches with device-resident BF16 boundary",
        )

    if backend == "direct_jit":
        model_arg = replicated_model(model)
        state_sharding = states.sharding
        weight_shardings = jax.tree.map(lambda _: replicated, model)
        mapped = jax.jit(
            local_call,
            in_shardings=(state_sharding, weight_shardings),
            out_shardings=state_sharding,
        )
        args = (states, model_arg)
        compiled, output, compilation = _compile(
            mapped, args, directory, prefix
        )
        return dict(
            runner=lambda: compiled(*args),
            initial_output=output,
            host_output=lambda value: np.asarray(value).reshape(-1, architecture.MOVE_COUNT),
            compilation=compilation,
            orchestration="direct jit with explicit global in/out shardings",
        )

    if backend == "pmap":
        state_shape = (
            device_count, local_batch, architecture.STATE_STORAGE_LEN
        )
        pmap_sharding = NamedSharding(mesh, P("core"))
        pmap_states = jax.device_put(
            host_states.reshape(state_shape), pmap_sharding
        )
        model_key = id(model)
        if model_key not in pmap_cache:
            pmap_cache[model_key] = jax.tree.map(
                lambda leaf: _pmap_value(
                    leaf, device_count=device_count, sharding=pmap_sharding
                ),
                model,
            )
            jax.block_until_ready(pmap_cache[model_key])
        model_arg = pmap_cache[model_key]
        axes = jax.tree.map(lambda _: 0, model_arg)
        mapped = jax.pmap(
            local_call, in_axes=(0, axes), out_axes=0, devices=devices
        )
        args = (pmap_states, model_arg)
        compiled, output, compilation = _compile(
            mapped, args, directory, prefix
        )
        return dict(
            runner=lambda: compiled(*args),
            initial_output=output,
            host_output=lambda value: np.asarray(value).reshape(-1, architecture.MOVE_COUNT),
            compilation=compilation,
            orchestration="pmap leading replica axis with preplaced per-replica weights",
        )

    if backend == "independent":
        calls = []
        args_list = []
        outputs = []
        compile_rows = []
        for index, device in enumerate(devices):
            model_key = (id(model), int(device.id))
            if model_key not in independent_cache:
                independent_cache[model_key] = jax.tree.map(
                    lambda leaf: jax.device_put(leaf, device), model
                )
                jax.block_until_ready(independent_cache[model_key])
            local_model = independent_cache[model_key]
            local_states = jax.device_put(
                host_states[index * local_batch:(index + 1) * local_batch],
                device,
            )
            mapped = jax.jit(local_call)
            args = (local_states, local_model)
            compiled, output, info = _compile(
                mapped, args, directory, f"{prefix}-core{index}"
            )
            calls.append(compiled)
            args_list.append(args)
            outputs.append(output)
            compile_rows.append(info)

        def runner():
            return tuple(call(*args) for call, args in zip(calls, args_list))

        return dict(
            runner=runner,
            initial_output=tuple(outputs),
            host_output=lambda values: np.concatenate(
                [np.asarray(value) for value in values], axis=0
            ).reshape(-1, architecture.MOVE_COUNT),
            compilation=dict(per_core=compile_rows),
            orchestration="eight independent one-partition AOT calls, sequential async host enqueue",
        )

    raise ValueError(f"unknown backend: {backend}")


def _measure_with_failure_record(runners, *, warmups, repeats):
    try:
        return dict(
            status="ok", comparison_valid=True,
            **measure_runner_group(runners, warmups=warmups, repeats=repeats),
        )
    except Exception as exc:
        result = dict(
            status="error", comparison_valid=False,
            label="unpaired_diagnostic_after_runner_group_failure",
            error=dict(type=type(exc).__name__, message=str(exc),
                       traceback=traceback.format_exc()),
            cases={}, case_errors={}, warmups=warmups, repeats=repeats,
        )
        for identifier, runner in runners.items():
            try:
                single = measure_runner_group(
                    {identifier: runner}, warmups=warmups, repeats=repeats
                )
                result["cases"][identifier] = single["cases"][identifier]
            except Exception as single_exc:
                result["case_errors"][identifier] = dict(
                    type=type(single_exc).__name__, message=str(single_exc)
                )
        return result


def _execute_section(report, result_path, *, section, prepared, architecture,
                     corpora, configs, devices, local_batch, directory,
                     warmups, repeats, interpret=False):
    device_count = len(devices)
    global_batch = device_count * local_batch
    mesh = Mesh(np.asarray(devices), ("core",))
    state_sharding = NamedSharding(mesh, P("core", None))
    replicated_cache = {}
    pmap_cache = {}
    independent_cache = {}
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
                    states=states, host_states=host_states, mesh=mesh,
                    devices=devices, replicated_cache=replicated_cache,
                    pmap_cache=pmap_cache, independent_cache=independent_cache,
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
            runners, warmups=warmups, repeats=repeats
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


def _single_device_replay(report, result_path, *, prepared, architecture,
                          corpora, directory, local_batch, interpret=False):
    """Replay v1 witness-owning shards on one TPU with the same local shape."""
    known = (
        ("legal_scrambles", 29_807),
        ("legal_scrambles", 50_224),
        ("categorical_stress", 29_369),
    )
    device = jax.devices()[0]
    mesh = Mesh(np.asarray([device]), ("core",))
    state_sharding = NamedSharding(mesh, P("core", None))
    configs_by_id = {config["id"]: config for config in execution_configs()}
    selected = [
        configs_by_id[ORACLE_ID], configs_by_id["typed_shard_map"],
        configs_by_id["pallas_shard_none"],
    ]
    for corpus, global_index in known:
        owner, local_index = owner_local_index(
            global_index, local_batch=local_batch,
            device_count=TARGET_DEVICE_COUNT,
        )
        start = owner * local_batch
        host_states = np.asarray(corpora[corpus][start:start + local_batch])
        state = host_states[local_index]
        scope = dict(
            corpus=corpus, global_index=global_index, owner=owner,
            local_index=local_index, shard_start=start,
            state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
            state_bytes_hex=state.tobytes().hex(),
        )
        states = jax.device_put(host_states, state_sharding)
        replicated_cache = {}
        outputs = {}
        row = dict(scope=scope, full_model=[])
        for config in selected:
            try:
                case = _build_case(
                    config, prepared=prepared, architecture=architecture,
                    states=states, host_states=host_states, mesh=mesh,
                    devices=[device], replicated_cache=replicated_cache,
                    pmap_cache={}, independent_cache={},
                    directory=directory / "hlo",
                    prefix=(f"replay-{corpus}-g{global_index}-{config['id']}"),
                    interpret=interpret,
                )
                host = case["host_output"](case["initial_output"])
                outputs[config["id"]] = host
                row["full_model"].append(dict(
                    id=config["id"], status="ok",
                    output_sha256=_array_sha256(host),
                    witness_q=[float(value) for value in host[local_index].astype(np.float32)],
                    compilation=case["compilation"],
                ))
            except Exception as exc:
                row["full_model"].append(dict(
                    id=config["id"], status="error",
                    error_type=type(exc).__name__, error=str(exc),
                    traceback=traceback.format_exc(),
                ))
        oracle = outputs.get(ORACLE_ID)
        if oracle is not None:
            for item in row["full_model"]:
                candidate = outputs.get(item["id"])
                if candidate is None:
                    continue
                item["comparison_vs_original"] = tensor_metrics(oracle, candidate)
                item["mismatch_witnesses"] = mismatch_witnesses(oracle, candidate)
                item["witness_comparison"] = mismatch_witnesses(
                    oracle[local_index:local_index + 1],
                    candidate[local_index:local_index + 1],
                )

        reference_config = dict(
            selected[1], embedding="reference", input_boundary="none"
        )
        pallas_config = dict(
            selected[2], embedding="pallas_banked_prepacked",
            input_boundary="none",
        )
        node_outputs = {}
        for label, config, model in (
            ("reference", reference_config, prepared["typed"][1]),
            ("pallas", pallas_config, prepared["pallas"][1]),
        ):
            call = candidate_nodes(
                config, architecture, sample_rows=(local_index,),
                interpret=interpret,
            )
            model_arg = jax.tree.map(lambda leaf: jax.device_put(leaf, device), model)
            mapped = jax.jit(call)
            args = (jax.device_put(host_states, device), model_arg)
            try:
                compiled, output, info = _compile(
                    mapped, args, directory / "hlo",
                    f"nodes-{corpus}-g{global_index}-{label}",
                )
                del compiled
                node_outputs[label] = jax.tree.map(np.asarray, output)
                row.setdefault("observed_nodes", {})[label] = dict(
                    status="ok", compilation=info,
                    output_sha256={key: _array_sha256(value)
                                   for key, value in node_outputs[label].items()},
                )
            except Exception as exc:
                row.setdefault("observed_nodes", {})[label] = dict(
                    status="error", error_type=type(exc).__name__,
                    error=str(exc), traceback=traceback.format_exc(),
                )
        if set(node_outputs) == {"reference", "pallas"}:
            row["node_comparisons"] = {
                key: dict(
                    metrics=tensor_metrics(
                        node_outputs["reference"][key], node_outputs["pallas"][key]
                    ),
                    witnesses=mismatch_witnesses(
                        node_outputs["reference"][key], node_outputs["pallas"][key]
                    ),
                )
                for key in node_outputs["reference"]
            }
        report["replay_diagnostics"].append(row)
        checkpoint(result_path, report)


def run_suite(params, original_apply, architecture, weights, corpora, directory,
              *, configs=None, target_local_batch=16_384,
              confirmation_local_batch=32_768, warmups=5, repeats=12,
              interpret=False, context=None):
    configs = execution_configs() if configs is None else list(configs)
    if not configs or configs[0]["id"] != ORACLE_ID:
        raise ValueError("canonical original_shard_map must be first")
    if len({config["id"] for config in configs}) != len(configs):
        raise ValueError("configuration IDs must be unique")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in (target_local_batch, confirmation_local_batch, warmups, repeats)
    ):
        raise ValueError("batch and timing counts must be positive integers")
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
    result_path = directory / "stream1_inference_execution_ab.json"
    if result_path.exists():
        raise FileExistsError("use a new output directory")
    report = dict(
        status="running", context=context or {}, architecture=asdict(architecture),
        protocol=dict(
            scope="full Q inference only; no beam-search work",
            oracle="original jax_model.apply through shard_map",
            target_devices=TARGET_DEVICE_COUNT,
            target_local_batch=target_local_batch,
            confirmation_local_batch=confirmation_local_batch,
            warmups=warmups, repeats=repeats,
            timing=(
                "paired forward/reverse, precompiled, device-resident, synchronized "
                "end-to-end runner wall time"
            ),
            acceptance=(
                "finite elementwise exact original Q on every corpus and faster "
                "than the fastest exact JAX control per corpus"
            ),
            split=(
                "two device-resident compiled dispatches; encoding output is an "
                "explicit HBM execution boundary"
            ),
            independent=(
                "diagnostic production option: eight one-partition executables, "
                "sequential asynchronous host enqueue, one final synchronization"
            ),
        ),
        configurations=configs, preparation={}, replay_diagnostics=[],
        input_scopes=[], measurements=[], timing_groups=[],
        target_decision=dict(winner_id=None, target_achieved=False,
                             jax_baseline_id={}, per_corpus_speedup={}),
        confirmation_decision=None,
    )
    checkpoint(result_path, report)
    try:
        prepared = _prepared_models(
            params, original_apply, architecture, weights, interpret=interpret
        )
        report["preparation"] = prepared["preparation"]
        checkpoint(result_path, report)
        _single_device_replay(
            report, result_path, prepared=prepared, architecture=architecture,
            corpora=corpora, directory=directory,
            local_batch=target_local_batch, interpret=interpret,
        )
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
            target_rows = [
                row for row in report["measurements"]
                if row["section"] == "target"
            ]
            control_ids = {
                config["id"] for config in configs
                if config["role"] == "jax_control"
                and all(any(
                    row["id"] == config["id"]
                    and row["corpus"] == corpus
                    and row.get("exact_oracle_on_sample") is True
                    and row.get("status") == "ok"
                    for row in target_rows
                ) for corpus in corpora)
            }
            selected_ids = {ORACLE_ID, winner, *control_ids}
            confirmation_configs = [
                config for config in configs if config["id"] in selected_ids
            ]
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
        default=Path("/kaggle/working/inference_execution_ab"),
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
            params, STATE_STORAGE_LEN=int(params["state_size"])
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
            seeds=dict(legal=42, stress=43),
            v1_witnesses=dict(
                legal_scrambles=[29_807, 50_224],
                categorical_stress=[29_369],
            ),
            v1_source_commit="d2159cb230ef77deeb5a4a2b6a42181a62dc027c",
        )
        print(json.dumps(context, indent=2), flush=True)
        report = run_suite(
            params, original_apply, architecture, weights, corpora,
            args.output, context=context,
        )
        print("RESULT", json.dumps(report["target_decision"], allow_nan=False), flush=True)
        print("CONFIRMATION", json.dumps(report["confirmation_decision"], allow_nan=False), flush=True)
        print("RESULT_PATH", args.output / "stream1_inference_execution_ab.json", flush=True)


if __name__ == "__main__":
    main()
