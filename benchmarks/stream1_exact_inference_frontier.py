"""Exact eight-TPU inference frontier after the accepted split winner.

This module keeps the already validated split path as an explicit control.
Experimental BM, head, and materialization arms may only be promoted after
elementwise-exact full-Q checks on every corpus.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import functools
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from benchmarks.diagnostic_timing import diagnostic_profile
from benchmarks.execution_boundary_ops import mismatch_witnesses
from benchmarks.layernorm_quality import (
    load_puzzle, make_legal_scrambles, tensor_metrics,
)
from benchmarks.stream1_inference_execution_ab import (
    _array_sha256, _compile, _measure_with_failure_record,
)
from benchmarks.stream1_layernorm_arithmetic import (
    checkpoint, runtime_inventory, runtime_params, sha256_file,
)
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_layernorm_exact import (
    prepare_exact_layernorm_inference_weights,
    stream1_layernorm_exact_head,
    stream1_layernorm_exact_prefix,
)
from tpu_beam_search.stream1_layernorm_pallas import pallas_layernorm_dense
from tpu_beam_search.stream1_layernorm_reference import (
    layernorm_stream1_weights_from_artgor_params,
    stream1_layernorm_reference_inference,
)
from tpu_beam_search.tpu_layout import pad_to_multiple


TARGET_DEVICE_COUNT = 8
ORIGINAL_ID = "original_shard_map"
TYPED_ID = "typed_monolithic"
ACCEPTED_ID = "exact_split_bm2048_jax_head"
JAX_HEAD_ID = "jax_head"
PREFIX_BMS = (2048, 4096, 8192, 16384)
IDENTITY_BMS = (128, 512, 2048)
HEAD_BMS = (128, 256, 512, 1024, 2048)
HEAD_BKS = (128, 256, 512, 1024)
HEAD_BN = 128
HEAD_ROUNDINGS = ("late", "bf16_before_bias")


def head_configs():
    configs = [dict(id=JAX_HEAD_ID, backend="jax", control=True)]
    for bm in HEAD_BMS:
        for bk in HEAD_BKS:
            for rounding in HEAD_ROUNDINGS:
                configs.append(dict(
                    id=f"pallas_head_bm{bm}_bk{bk}_bn{HEAD_BN}_{rounding}",
                    backend="pallas", control=False, bm=bm, bk=bk,
                    bn=HEAD_BN, dense_rounding=rounding,
                ))
    return configs


def _head_by_id():
    return {config["id"]: config for config in head_configs()}


def full_candidate_configs(*, exact_prefix_bms, promoted_head_ids,
                           accepted_prefix_bm=2048,
                           identity_bms=IDENTITY_BMS):
    """Build a bounded full-model matrix around the accepted exact control."""
    prefix_bms = tuple(exact_prefix_bms)
    if not prefix_bms or len(set(prefix_bms)) != len(prefix_bms):
        raise ValueError("exact_prefix_bms must be unique and nonempty")
    if accepted_prefix_bm not in prefix_bms:
        raise ValueError("the accepted prefix BM control must be retained")
    known_heads = _head_by_id()
    promoted = tuple(promoted_head_ids)
    if len(set(promoted)) != len(promoted) or any(
        identifier == JAX_HEAD_ID or identifier not in known_heads
        for identifier in promoted
    ):
        raise ValueError("promoted_head_ids must be unique Pallas head IDs")

    configs = [
        dict(id=ORIGINAL_ID, role="jax_control", backend="monolithic",
             implementation="original"),
        dict(id=TYPED_ID, role="jax_control", backend="monolithic",
             implementation="typed"),
        dict(id=ACCEPTED_ID, role="accepted_control", backend="split",
             implementation="exact", prefix_bm=accepted_prefix_bm,
             head_backend="jax", head_id=JAX_HEAD_ID),
    ]
    for bm in prefix_bms:
        if bm != accepted_prefix_bm:
            configs.append(dict(
                id=f"exact_split_bm{bm}_jax_head", role="candidate",
                backend="split", implementation="exact", prefix_bm=bm,
                head_backend="jax", head_id=JAX_HEAD_ID,
            ))
        for head_id in promoted:
            configs.append(dict(
                id=f"exact_split_bm{bm}_{head_id}", role="candidate",
                backend="split", implementation="exact", prefix_bm=bm,
                head_backend="pallas", head_id=head_id,
            ))
    for bm in identity_bms:
        configs.append(dict(
            id=f"exact_identity_bm{bm}_jax_head", role="candidate",
            backend="materialized_identity", implementation="exact",
            prefix_bm=accepted_prefix_bm, identity_bm=bm,
            head_backend="jax", head_id=JAX_HEAD_ID,
        ))
    return configs


def _eligible(row):
    timing = row.get("timing", {}).get("median_ms")
    return (
        row.get("status") == "ok"
        and row.get("exact_oracle_on_sample") is True
        and row.get("timing_comparable") is True
        and isinstance(timing, (int, float))
        and not isinstance(timing, bool)
        and math.isfinite(timing)
        and timing > 0
    )


def select_head_promotions(rows, *, corpus_names, limit=3):
    """Promote the fastest heads that are exact on every required corpus."""
    names = tuple(corpus_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("corpus_names must be unique and nonempty")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    baselines = {
        row["corpus"]: row for row in rows
        if row.get("id") == JAX_HEAD_ID
        and row.get("corpus") in names and _eligible(row)
    }
    identifiers = sorted({
        row.get("id") for row in rows
        if row.get("id") not in (None, JAX_HEAD_ID)
    })
    accepted = []
    rejected = []
    speedups = {}
    for identifier in identifiers:
        matching = {
            row["corpus"]: row for row in rows
            if row.get("id") == identifier
            and row.get("corpus") in names and _eligible(row)
        }
        if set(baselines) != set(names) or set(matching) != set(names):
            rejected.append(identifier)
            continue
        values = {
            corpus: baselines[corpus]["timing"]["median_ms"]
            / matching[corpus]["timing"]["median_ms"]
            for corpus in names
        }
        speedups[identifier] = values
        accepted.append((statistics.mean(values.values()), identifier))
    accepted.sort(key=lambda item: (-item[0], item[1]))
    selected = [identifier for _, identifier in accepted[:limit]]
    return dict(
        selected_ids=selected,
        rejected_ids=rejected,
        per_corpus_speedup={identifier: speedups[identifier]
                            for identifier in selected},
    )


def select_frontier_improvement(rows, *, corpus_names):
    """Require a new full path to beat the accepted exact split everywhere."""
    names = tuple(corpus_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("corpus_names must be unique and nonempty")
    baseline = {
        row["corpus"]: row for row in rows
        if row.get("id") == ACCEPTED_ID
        and row.get("corpus") in names and _eligible(row)
    }
    if set(baseline) != set(names):
        return dict(
            accepted_control_id=ACCEPTED_ID, selected_id=None,
            improvement_achieved=False, per_corpus_speedup={},
        )
    identifiers = sorted({
        row.get("id") for row in rows
        if row.get("id") not in (None, ACCEPTED_ID, ORIGINAL_ID, TYPED_ID)
    })
    winners = []
    for identifier in identifiers:
        matching = {
            row["corpus"]: row for row in rows
            if row.get("id") == identifier
            and row.get("corpus") in names and _eligible(row)
        }
        if set(matching) != set(names):
            continue
        values = {
            corpus: baseline[corpus]["timing"]["median_ms"]
            / matching[corpus]["timing"]["median_ms"]
            for corpus in names
        }
        if all(value > 1.0 for value in values.values()):
            winners.append((statistics.mean(values.values()), identifier, values))
    if not winners:
        return dict(
            accepted_control_id=ACCEPTED_ID, selected_id=ACCEPTED_ID,
            improvement_achieved=False, per_corpus_speedup={},
        )
    _, identifier, values = max(winners)
    return dict(
        accepted_control_id=ACCEPTED_ID, selected_id=identifier,
        improvement_achieved=True, per_corpus_speedup=values,
    )


def profile_stage_ids(decision, configs):
    """List compiled stages and composed runners needed for attribution."""
    by_id = {config["id"]: config for config in configs}
    if ACCEPTED_ID not in by_id:
        raise ValueError("profile plan requires the accepted control")
    selected_id = decision.get("selected_id") or ACCEPTED_ID
    if selected_id not in by_id:
        raise ValueError("selected profile candidate is missing")
    accepted = by_id[ACCEPTED_ID]
    compiled = [
        ORIGINAL_ID, TYPED_ID,
        f"prefix_bm{accepted['prefix_bm']}", JAX_HEAD_ID,
    ]
    composed = [ACCEPTED_ID]
    selected = by_id[selected_id]
    if selected_id != ACCEPTED_ID:
        composed.append(selected_id)
        if selected["backend"] == "split":
            compiled.extend([
                f"prefix_bm{selected['prefix_bm']}", selected["head_id"],
            ])
        elif selected["backend"] == "materialized_identity":
            compiled.append(selected_id)
    return dict(
        compiled=list(dict.fromkeys(compiled)),
        composed=list(dict.fromkeys(composed)),
    )


def _identity_kernel(input_ref, output_ref):
    output_ref[...] = input_ref[...]


def materialize_hidden(values, *, bm=128, interpret=False):
    """Insert a real Pallas value boundary without changing BF16 hidden bits."""
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("values must be a nonempty matrix")
    if not isinstance(bm, int) or isinstance(bm, bool) or bm <= 0:
        raise ValueError("bm must be a positive integer")
    rows, width = values.shape
    padded_rows = pad_to_multiple(rows, bm)
    padded = jnp.pad(values, ((0, padded_rows - rows), (0, 0)))
    spec = pl.BlockSpec((bm, width), lambda i: (i, 0))
    call = pl.pallas_call(
        _identity_kernel,
        grid=(padded_rows // bm,),
        in_specs=[spec], out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(padded.shape, padded.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="stream1_exact_hidden_materialization",
    )
    return call(padded)[:rows]


def head_call(config, architecture, *, interpret=False):
    """Return one standalone hidden-to-Q implementation from a head config."""
    backend = config.get("backend")
    if backend == "jax":
        return lambda hidden, weights: stream1_layernorm_exact_head(
            hidden, weights, architecture,
        )
    if backend != "pallas":
        raise ValueError(f"unknown head backend: {backend}")
    required = ("bm", "bk", "bn", "dense_rounding")
    if any(name not in config for name in required):
        raise ValueError("Pallas head config is incomplete")
    return lambda hidden, weights: pallas_layernorm_dense(
        hidden, weights.output.weight, weights.output.bias,
        bm=config["bm"], bk=config["bk"], bn=config["bn"],
        dense_rounding=config["dense_rounding"], interpret=interpret,
    )


def _mapped(call, *, mesh, weights_example, input_specs=P("core", None)):
    weight_specs = jax.tree.map(lambda _: P(), weights_example)
    return jax.jit(jax.shard_map(
        call,
        mesh=mesh,
        in_specs=(input_specs, weight_specs),
        out_specs=P("core", None),
        check_vma=False,
    ))


def _replicate(tree, mesh):
    sharding = NamedSharding(mesh, P())
    result = jax.tree.map(lambda value: jax.device_put(value, sharding), tree)
    jax.block_until_ready(result)
    return result


def _tree_bytes(tree):
    return int(sum(
        value.size * value.dtype.itemsize
        for value in jax.tree.leaves(tree)
        if hasattr(value, "size") and hasattr(value, "dtype")
    ))


def _error_record(identifier, section, exc):
    return dict(
        section=section, id=identifier, status="error",
        error_type=type(exc).__name__, error=str(exc),
        traceback=traceback.format_exc(),
    )


def _comparison(reference, candidate):
    metrics = tensor_metrics(reference, candidate)
    return dict(
        comparison_vs_original=metrics,
        mismatch_witnesses=mismatch_witnesses(reference, candidate),
        exact_oracle_on_sample=bool(
            metrics["finite"] and metrics["exact_fraction"] == 1.0
        ),
        argmin_agreement=float(np.mean(
            np.argmin(reference, axis=1) == np.argmin(candidate, axis=1)
        )),
        output_sha256=_array_sha256(candidate),
    )


def _apply_timing(rows, group):
    by_id = {row["id"]: row for row in rows if row.get("status") == "ok"}
    for identifier, row in by_id.items():
        row["timing_comparable"] = group["comparison_valid"]
        row["timing_label"] = group["label"]
        if identifier in group["cases"]:
            row["timing"] = group["cases"][identifier]


def _identity_full_call(architecture, *, prefix_bm, identity_bm, interpret):
    def call(states, weights):
        hidden = stream1_layernorm_exact_prefix(
            states, weights, architecture, bm=prefix_bm, interpret=interpret,
        )
        hidden = materialize_hidden(hidden, bm=identity_bm, interpret=interpret)
        return stream1_layernorm_exact_head(hidden, weights, architecture)
    return call


def _execute_batch(
    report,
    result_path,
    *,
    section,
    original_apply,
    metadata,
    payload,
    architecture,
    typed_weights,
    exact_weights,
    corpora,
    directory,
    device_count,
    local_batch,
    prefix_bms,
    accepted_prefix_bm,
    identity_bms,
    heads,
    head_promotion_limit,
    forced_head_ids,
    warmups,
    repeats,
    interpret,
    collect_profiles,
):
    devices = jax.devices()[:device_count]
    mesh = Mesh(np.asarray(devices), ("core",))
    state_sharding = NamedSharding(mesh, P("core", None))
    global_batch = device_count * local_batch
    corpus_names = tuple(corpora)
    first_corpus = corpus_names[0]
    states = {
        name: jax.device_put(np.asarray(values[:global_batch]), state_sharding)
        for name, values in corpora.items()
    }
    for name, value in states.items():
        host = np.asarray(corpora[name][:global_batch])
        report["input_scopes"].append(dict(
            section=section, corpus=name, device_count=device_count,
            local_batch=local_batch, global_batch=global_batch,
            global_input_sha256=hashlib.sha256(host.tobytes()).hexdigest(),
            shard_input_sha256=[
                hashlib.sha256(
                    host[index * local_batch:(index + 1) * local_batch].tobytes()
                ).hexdigest()
                for index in range(device_count)
            ],
        ))

    payload_arg = _replicate(payload, mesh)
    typed_arg = _replicate(typed_weights, mesh)
    exact_arg = _replicate(exact_weights, mesh)
    hlo_dir = directory / "hlo"

    original_call = lambda value, model: original_apply(
        {**metadata, **model}, value, dtype=jnp.bfloat16,
    )
    typed_call = lambda value, model: stream1_layernorm_reference_inference(
        value, model, architecture,
    )
    controls = {}
    for identifier, call, model, model_arg in (
        (ORIGINAL_ID, original_call, payload, payload_arg),
        (TYPED_ID, typed_call, typed_weights, typed_arg),
    ):
        mapped = _mapped(call, mesh=mesh, weights_example=model)
        compiled, output, info = _compile(
            mapped, (states[first_corpus], model_arg), hlo_dir,
            f"{section}-d{device_count}-b{local_batch}-{identifier}",
        )
        controls[identifier] = dict(
            compiled=compiled, model_arg=model_arg, compilation=info,
            first_output=output,
        )
        checkpoint(result_path, report)

    prefix_cases = {}
    for bm in prefix_bms:
        identifier = f"prefix_bm{bm}"
        try:
            call = lambda value, model, bm=bm: stream1_layernorm_exact_prefix(
                value, model, architecture, bm=bm, interpret=interpret,
            )
            mapped = _mapped(call, mesh=mesh, weights_example=exact_weights)
            compiled, output, info = _compile(
                mapped, (states[first_corpus], exact_arg), hlo_dir,
                f"{section}-d{device_count}-b{local_batch}-{identifier}",
            )
            prefix_cases[bm] = dict(
                compiled=compiled, compilation=info, first_output=output,
            )
        except Exception as exc:
            report["compile_errors"].append(_error_record(identifier, section, exc))
            if bm == accepted_prefix_bm:
                raise
        checkpoint(result_path, report)

    accepted_hidden = prefix_cases[accepted_prefix_bm]["first_output"]
    head_cases = {}
    head_by_id = {config["id"]: config for config in heads}
    for config in heads:
        identifier = config["id"]
        try:
            mapped = _mapped(
                head_call(config, architecture, interpret=interpret),
                mesh=mesh, weights_example=exact_weights,
            )
            compiled, output, info = _compile(
                mapped, (accepted_hidden, exact_arg), hlo_dir,
                f"{section}-d{device_count}-b{local_batch}-{identifier}",
            )
            head_cases[identifier] = dict(
                compiled=compiled, compilation=info, first_output=output,
            )
        except Exception as exc:
            report["compile_errors"].append(_error_record(identifier, section, exc))
            if identifier == JAX_HEAD_ID:
                raise
        checkpoint(result_path, report)

    hidden_by_corpus = {}
    for corpus in corpus_names:
        hidden_by_corpus[corpus] = {}
        for bm, case in prefix_cases.items():
            hidden_by_corpus[corpus][bm] = jax.block_until_ready(
                case["compiled"](states[corpus], exact_arg)
            )

    original_outputs = {
        corpus: np.asarray(jax.block_until_ready(
            controls[ORIGINAL_ID]["compiled"](
                states[corpus], controls[ORIGINAL_ID]["model_arg"],
            )
        )).reshape(-1, architecture.MOVE_COUNT)
        for corpus in corpus_names
    }

    head_rows = []
    for corpus in corpus_names:
        hidden = hidden_by_corpus[corpus][accepted_prefix_bm]
        jax_output = np.asarray(jax.block_until_ready(
            head_cases[JAX_HEAD_ID]["compiled"](hidden, exact_arg)
        )).reshape(-1, architecture.MOVE_COUNT)
        runners = {}
        corpus_rows = []
        for config in heads:
            identifier = config["id"]
            row = dict(
                section=section, corpus=corpus, id=identifier,
                config=config, status="running", device_count=device_count,
                local_batch=local_batch, global_batch=global_batch,
                hidden_sha256=_array_sha256(np.asarray(hidden)),
            )
            head_rows.append(row)
            report["head_measurements"].append(row)
            if identifier not in head_cases:
                row.update(
                    status="error", exact_oracle_on_sample=False,
                    error="head compilation failed; see compile_errors",
                )
                continue
            compiled = head_cases[identifier]["compiled"]
            output = np.asarray(jax.block_until_ready(
                compiled(hidden, exact_arg)
            )).reshape(-1, architecture.MOVE_COUNT)
            metrics = tensor_metrics(jax_output, output)
            row.update(
                status="ok", compilation=head_cases[identifier]["compilation"],
                comparison_vs_jax_head=metrics,
                mismatch_witnesses=mismatch_witnesses(jax_output, output),
                exact_oracle_on_sample=bool(
                    metrics["finite"] and metrics["exact_fraction"] == 1.0
                ),
                output_sha256=_array_sha256(output),
            )
            corpus_rows.append(row)
            runners[identifier] = functools.partial(compiled, hidden, exact_arg)
        timing = _measure_with_failure_record(
            runners, warmups=warmups, repeats=repeats,
        )
        timing.update(
            phase="head", section=section, corpus=corpus,
            device_count=device_count, local_batch=local_batch,
        )
        report["timing_groups"].append(timing)
        _apply_timing(corpus_rows, timing)
        checkpoint(result_path, report)

    head_decision = select_head_promotions(
        head_rows, corpus_names=corpus_names, limit=head_promotion_limit,
    )
    exact_head_ids = {
        identifier for identifier in head_by_id
        if all(any(
            row["id"] == identifier and row["corpus"] == corpus
            and row.get("exact_oracle_on_sample") is True
            for row in head_rows
        ) for corpus in corpus_names)
    }
    promoted_head_ids = list(head_decision["selected_ids"])
    for identifier in forced_head_ids:
        if identifier in exact_head_ids and identifier not in promoted_head_ids:
            promoted_head_ids.append(identifier)
    report["head_decisions"].append(dict(
        section=section, **head_decision,
        effective_promoted_ids=promoted_head_ids,
    ))
    checkpoint(result_path, report)

    prefix_rows = []
    prefix_runners = {corpus: {} for corpus in corpus_names}
    exact_prefix_bms = []
    for bm, case in prefix_cases.items():
        exact_everywhere = True
        for corpus in corpus_names:
            hidden = hidden_by_corpus[corpus][bm]
            q = np.asarray(jax.block_until_ready(
                head_cases[JAX_HEAD_ID]["compiled"](hidden, exact_arg)
            )).reshape(-1, architecture.MOVE_COUNT)
            row = dict(
                section=section, corpus=corpus, id=f"prefix_bm{bm}",
                bm=bm, status="ok", device_count=device_count,
                local_batch=local_batch, global_batch=global_batch,
                compilation=case["compilation"],
                hidden_comparison_vs_accepted=tensor_metrics(
                    np.asarray(hidden_by_corpus[corpus][accepted_prefix_bm]),
                    np.asarray(hidden),
                ),
                **_comparison(original_outputs[corpus], q),
            )
            prefix_rows.append(row)
            report["prefix_measurements"].append(row)
            exact_everywhere &= row["exact_oracle_on_sample"]
            prefix_runners[corpus][f"prefix_bm{bm}"] = functools.partial(
                case["compiled"], states[corpus], exact_arg,
            )
        if exact_everywhere:
            exact_prefix_bms.append(bm)
    if accepted_prefix_bm not in exact_prefix_bms:
        raise RuntimeError("accepted exact prefix failed the full-Q oracle gate")

    for corpus, runners in prefix_runners.items():
        timing = _measure_with_failure_record(
            runners, warmups=warmups, repeats=repeats,
        )
        timing.update(
            phase="prefix", section=section, corpus=corpus,
            device_count=device_count, local_batch=local_batch,
        )
        report["timing_groups"].append(timing)
        _apply_timing(
            [row for row in prefix_rows if row["corpus"] == corpus], timing,
        )
    checkpoint(result_path, report)

    identity_cases = {}
    for bm in identity_bms:
        identifier = f"exact_identity_bm{bm}_jax_head"
        try:
            mapped = _mapped(
                _identity_full_call(
                    architecture, prefix_bm=accepted_prefix_bm,
                    identity_bm=bm, interpret=interpret,
                ),
                mesh=mesh, weights_example=exact_weights,
            )
            compiled, output, info = _compile(
                mapped, (states[first_corpus], exact_arg), hlo_dir,
                f"{section}-d{device_count}-b{local_batch}-{identifier}",
            )
            identity_cases[bm] = dict(
                compiled=compiled, compilation=info, first_output=output,
            )
        except Exception as exc:
            report["compile_errors"].append(_error_record(identifier, section, exc))
        checkpoint(result_path, report)

    configs = full_candidate_configs(
        exact_prefix_bms=tuple(exact_prefix_bms),
        promoted_head_ids=tuple(promoted_head_ids),
        accepted_prefix_bm=accepted_prefix_bm,
        identity_bms=tuple(identity_bms),
    )
    report["full_configurations"].extend(
        dict(section=section, **config) for config in configs
    )

    full_rows = []
    full_runner_cache = {}
    for corpus in corpus_names:
        runners = {}
        rows = []
        outputs = {}
        for config in configs:
            identifier = config["id"]
            row = dict(
                section=section, corpus=corpus, id=identifier,
                role=config["role"], config=config, status="running",
                device_count=device_count, local_batch=local_batch,
                global_batch=global_batch,
            )
            full_rows.append(row)
            rows.append(row)
            report["full_measurements"].append(row)
            try:
                if identifier in (ORIGINAL_ID, TYPED_ID):
                    case = controls[identifier]
                    runner = functools.partial(
                        case["compiled"], states[corpus], case["model_arg"],
                    )
                    compilation = case["compilation"]
                elif config["backend"] == "split":
                    prefix = prefix_cases[config["prefix_bm"]]["compiled"]
                    head = head_cases[config["head_id"]]["compiled"]

                    def runner(prefix=prefix, head=head, state=states[corpus]):
                        return head(prefix(state, exact_arg), exact_arg)

                    compilation = dict(
                        prefix=prefix_cases[config["prefix_bm"]]["compilation"],
                        head=head_cases[config["head_id"]]["compilation"],
                    )
                else:
                    case = identity_cases[config["identity_bm"]]
                    runner = functools.partial(
                        case["compiled"], states[corpus], exact_arg,
                    )
                    compilation = case["compilation"]
                output = np.asarray(jax.block_until_ready(runner())).reshape(
                    -1, architecture.MOVE_COUNT,
                )
                outputs[identifier] = output
                row.update(
                    status="ok", compilation=compilation,
                    orchestration=(
                        "two compiled device-resident shard_map dispatches"
                        if config["backend"] == "split"
                        else "one compiled shard_map dispatch"
                    ),
                    **_comparison(original_outputs[corpus], output),
                )
                runners[identifier] = runner
                if corpus == first_corpus:
                    full_runner_cache[identifier] = runner
            except Exception as exc:
                row.update(_error_record(identifier, section, exc))
                row["corpus"] = corpus
                row["exact_oracle_on_sample"] = False
            checkpoint(result_path, report)
        timing = _measure_with_failure_record(
            runners, warmups=warmups, repeats=repeats,
        )
        timing.update(
            phase="full", section=section, corpus=corpus,
            device_count=device_count, local_batch=local_batch,
        )
        report["timing_groups"].append(timing)
        _apply_timing(rows, timing)
        for row in rows:
            if "timing" in row:
                row["states_per_second"] = (
                    global_batch * 1000.0 / row["timing"]["median_ms"]
                )
        checkpoint(result_path, report)

    decision = select_frontier_improvement(
        full_rows, corpus_names=corpus_names,
    )
    if collect_profiles:
        available_profiles = {
            ORIGINAL_ID: (
                controls[ORIGINAL_ID]["compiled"],
                (states[first_corpus], controls[ORIGINAL_ID]["model_arg"]),
            ),
            TYPED_ID: (
                controls[TYPED_ID]["compiled"],
                (states[first_corpus], controls[TYPED_ID]["model_arg"]),
            ),
        }
        available_profiles.update({
            f"prefix_bm{bm}": (
                case["compiled"], (states[first_corpus], exact_arg),
            )
            for bm, case in prefix_cases.items()
        })
        available_profiles.update({
            identifier: (
                case["compiled"],
                (hidden_by_corpus[first_corpus][accepted_prefix_bm], exact_arg),
            )
            for identifier, case in head_cases.items()
        })
        available_profiles.update({
            f"exact_identity_bm{bm}_jax_head": (
                case["compiled"], (states[first_corpus], exact_arg),
            )
            for bm, case in identity_cases.items()
        })
        profile_plan = profile_stage_ids(decision, configs)
        for identifier in profile_plan["compiled"]:
            compiled, arguments = available_profiles[identifier]
            try:
                report["profiles"].append(dict(
                    section=section, id=identifier, profile_scope="compiled_stage",
                    **diagnostic_profile(
                        compiled, *arguments,
                        directory=directory / "profiles" / section / identifier,
                        iterations=3,
                    ),
                ))
            except Exception as exc:
                report["profiles"].append(dict(
                    section=section, id=identifier, status="error",
                    error_type=type(exc).__name__, error=str(exc),
                ))
        for identifier in profile_plan["composed"]:
            try:
                trace_dir = directory / "profiles" / section / f"{identifier}_composed"
                trace_dir.mkdir(parents=True, exist_ok=True)
                with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
                    for _ in range(3):
                        jax.block_until_ready(full_runner_cache[identifier]())
                report["profiles"].append(dict(
                    section=section, id=identifier, status="ok",
                    profile_scope="composed_runner",
                    label="diagnostic_composed_runner_trace",
                    directory=str(trace_dir), iterations=3,
                ))
            except Exception as exc:
                report["profiles"].append(dict(
                    section=section, id=identifier, status="error",
                    profile_scope="composed_runner",
                    error_type=type(exc).__name__, error=str(exc),
                ))
        checkpoint(result_path, report)
    return dict(
        decision=decision,
        full_rows=full_rows,
        full_configs=configs,
        head_decision=head_decision,
    )


def _confirmation_matrix(screen, *, accepted_prefix_bm, heads_by_id):
    selected = screen["decision"]["selected_id"] or ACCEPTED_ID
    config = next(
        item for item in screen["full_configs"] if item["id"] == selected
    )
    prefix_bms = [accepted_prefix_bm]
    identity_bms = []
    selected_heads = [heads_by_id[JAX_HEAD_ID]]
    forced_heads = []
    if config["backend"] == "split":
        if config["prefix_bm"] not in prefix_bms:
            prefix_bms.append(config["prefix_bm"])
        if config["head_id"] != JAX_HEAD_ID:
            selected_heads.append(heads_by_id[config["head_id"]])
            forced_heads.append(config["head_id"])
    elif config["backend"] == "materialized_identity":
        identity_bms.append(config["identity_bm"])
    return dict(
        selected_id=selected, prefix_bms=tuple(prefix_bms),
        identity_bms=tuple(identity_bms), heads=selected_heads,
        forced_head_ids=tuple(forced_heads),
    )


def run_suite(
    params,
    original_apply,
    architecture,
    weights,
    corpora,
    directory,
    *,
    screen_local_batch=16_384,
    confirmation_local_batch=32_768,
    device_count=TARGET_DEVICE_COUNT,
    prefix_bms=PREFIX_BMS,
    accepted_prefix_bm=2048,
    identity_bms=IDENTITY_BMS,
    heads=None,
    head_promotion_limit=3,
    warmups=5,
    repeats=12,
    interpret=False,
    collect_profiles=True,
    context=None,
):
    """Run a staged exactness-first frontier; never beam-search work."""
    prefix_bms = tuple(prefix_bms)
    identity_bms = tuple(identity_bms)
    heads = head_configs() if heads is None else list(heads)
    counts = (
        screen_local_batch, confirmation_local_batch, device_count,
        head_promotion_limit, warmups, repeats,
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in counts
    ):
        raise ValueError("batch, device, promotion, and timing counts must be positive")
    if device_count > len(jax.devices()):
        raise RuntimeError(f"need {device_count} devices, found {len(jax.devices())}")
    if (not prefix_bms or len(set(prefix_bms)) != len(prefix_bms)
            or accepted_prefix_bm not in prefix_bms
            or any(not isinstance(value, int) or value <= 0 for value in prefix_bms)):
        raise ValueError("prefix_bms must be unique, positive, and retain accepted BM")
    if (len(set(identity_bms)) != len(identity_bms)
            or any(not isinstance(value, int) or value <= 0 for value in identity_bms)):
        raise ValueError("identity_bms must be unique positive integers")
    if (not heads or heads[0].get("id") != JAX_HEAD_ID
            or heads[0].get("backend") != "jax"
            or len({config.get("id") for config in heads}) != len(heads)):
        raise ValueError("heads must be unique and start with jax_head")
    corpus_names = tuple(corpora)
    if not corpus_names or len(corpus_names) != len(set(corpus_names)):
        raise ValueError("corpora must be unique and nonempty")
    required = device_count * max(screen_local_batch, confirmation_local_batch)
    for name, values in corpora.items():
        array = np.asarray(values)
        if (array.dtype != np.uint8 or array.ndim != 2
                or array.shape[1] != architecture.STATE_STORAGE_LEN
                or len(array) < required
                or np.any(array[:, :architecture.STATE_LEN] >= architecture.NUM_CLASSES)):
            raise ValueError(f"invalid corpus contract: {name}")

    directory = Path(directory)
    result_path = directory / "stream1_exact_inference_frontier.json"
    if result_path.exists():
        raise FileExistsError("use a new output directory")
    payload, metadata = runtime_params(params)
    exact_weights = prepare_exact_layernorm_inference_weights(weights, architecture)
    report = dict(
        status="running", context=context or {}, architecture=asdict(architecture),
        protocol=dict(
            scope="full Q inference only; no move expansion, top-k, or beam search",
            target_devices=device_count,
            screen_local_batch=screen_local_batch,
            confirmation_local_batch=confirmation_local_batch,
            warmups=warmups, repeats=repeats,
            timing="paired alternating synchronized calls; compiled execution only",
            acceptance=(
                "finite elementwise-exact original BF16 Q on every corpus; "
                "a new winner must beat accepted split BM control everywhere"
            ),
            accepted_control=ACCEPTED_ID,
            intermediate="BF16 hidden remains device-resident between dispatches",
            profiles="diagnostic only; never an acceptance substitute",
        ),
        configurations=dict(
            prefix_bms=list(prefix_bms), identity_bms=list(identity_bms),
            heads=heads, accepted_prefix_bm=accepted_prefix_bm,
        ),
        preparation=dict(
            payload_bytes=_tree_bytes(payload),
            typed_weight_bytes=_tree_bytes(weights),
            exact_weight_bytes=_tree_bytes(exact_weights),
            exact_bank_shapes=[
                list(exact_weights.embedding.low.shape),
                list(exact_weights.embedding.high.shape),
            ],
            exact_bank_dtype=str(exact_weights.embedding.low.dtype),
        ),
        input_scopes=[], compile_errors=[], head_measurements=[],
        head_decisions=[], prefix_measurements=[], full_configurations=[],
        full_measurements=[], timing_groups=[], profiles=[],
        screen_decision=None, confirmation_decision=None,
    )
    checkpoint(result_path, report)
    try:
        screen = _execute_batch(
            report, result_path, section="screen",
            original_apply=original_apply, metadata=metadata, payload=payload,
            architecture=architecture, typed_weights=weights,
            exact_weights=exact_weights, corpora=corpora, directory=directory,
            device_count=device_count, local_batch=screen_local_batch,
            prefix_bms=prefix_bms, accepted_prefix_bm=accepted_prefix_bm,
            identity_bms=identity_bms, heads=heads,
            head_promotion_limit=head_promotion_limit, forced_head_ids=(),
            warmups=warmups, repeats=repeats, interpret=interpret,
            collect_profiles=collect_profiles,
        )
        report["screen_decision"] = screen["decision"]
        checkpoint(result_path, report)
        heads_by_id = {config["id"]: config for config in heads}
        confirmation_matrix = _confirmation_matrix(
            screen, accepted_prefix_bm=accepted_prefix_bm,
            heads_by_id=heads_by_id,
        )
        confirmation = _execute_batch(
            report, result_path, section="confirmation",
            original_apply=original_apply, metadata=metadata, payload=payload,
            architecture=architecture, typed_weights=weights,
            exact_weights=exact_weights, corpora=corpora, directory=directory,
            device_count=device_count, local_batch=confirmation_local_batch,
            prefix_bms=confirmation_matrix["prefix_bms"],
            accepted_prefix_bm=accepted_prefix_bm,
            identity_bms=confirmation_matrix["identity_bms"],
            heads=confirmation_matrix["heads"],
            head_promotion_limit=1,
            forced_head_ids=confirmation_matrix["forced_head_ids"],
            warmups=warmups, repeats=repeats, interpret=interpret,
            collect_profiles=collect_profiles,
        )
        confirmation_decision = confirmation["decision"]
        selected = confirmation_matrix["selected_id"]
        confirmation_decision["screen_selected_id"] = selected
        confirmation_decision["confirmed"] = bool(
            confirmation_decision["selected_id"] == selected
            and (
                selected == ACCEPTED_ID
                or confirmation_decision["improvement_achieved"]
            )
        )
        report["confirmation_decision"] = confirmation_decision
        report["status"] = "complete"
        report["error_count"] = len(report["compile_errors"]) + sum(
            row.get("status") == "error" for row in report["full_measurements"]
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
        return Path(path)
    return next((candidate for candidate in (
        Path("/kaggle/input/cube555-tpu-artifacts"),
        Path("/kaggle/input/datasets/artgor/cube555-tpu-artifacts"),
    ) if candidate.is_dir()), None)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/kaggle/working/exact_inference_frontier"),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dataset = _dataset_path(args.dataset)
    if dataset is None or not dataset.is_dir():
        raise FileNotFoundError("attach artgor/cube555-tpu-artifacts")
    devices = jax.devices()
    if (len(devices) < TARGET_DEVICE_COUNT
            or any(device.platform != "tpu" for device in devices[:TARGET_DEVICE_COUNT])):
        raise RuntimeError(f"requires eight TPU devices, found: {devices}")
    inventory = runtime_inventory()
    inventory["active_device_count"] = TARGET_DEVICE_COUNT
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
        weights = layernorm_stream1_weights_from_artgor_params(
            params, architecture,
        )
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
            accepted_source_commit=(
                "7865ac455b0d4dfbb3d6e8b68430164790fc076c"
            ),
            accepted_result=(
                "exact split after final residual block; prepacked Pallas "
                "embedding prefix and separate device-resident JAX head"
            ),
        )
        print(json.dumps(context, indent=2), flush=True)
        report = run_suite(
            params, original_apply, architecture, weights, corpora,
            args.output, context=context,
        )
        print("SCREEN", json.dumps(
            report["screen_decision"], allow_nan=False,
        ), flush=True)
        print("CONFIRMATION", json.dumps(
            report["confirmation_decision"], allow_nan=False,
        ), flush=True)
        print(
            "RESULT_PATH", args.output / "stream1_exact_inference_frontier.json",
            flush=True,
        )


if __name__ == "__main__":
    main()
