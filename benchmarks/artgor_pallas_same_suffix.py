"""Causal same-suffix attribution for isolated all-Pallas ResMLP blocks."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import traceback

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from benchmarks.artgor_exact_notebook_validation import (
    _array_sha256, _dataset_path, _make_original_inference, _replicate,
    _tensor_comparison, checkpoint,
)
from benchmarks.artgor_pallas_exact_diagnostic import _make_states
from benchmarks.artgor_input_trace import (
    TRACE_NAMES, MEAN_ORDERS, input_trace, save_mismatch_rows, mean_buffer, external_mean_ln,
)
from benchmarks.layernorm_quality import load_puzzle
from benchmarks.stream1_layernorm_arithmetic import runtime_inventory, sha256_file
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_layernorm_pallas_exact import (
    _exact_dense,
    PallasExactConfig, pallas_exact_embedding, pallas_exact_head,
    pallas_exact_input_block, pallas_exact_input_dense,
    pallas_exact_layer_norm_activation, pallas_exact_residual_block,
    prepare_pallas_exact_weights,
    stream1_layernorm_pallas_exact_inference,
    pallas_preactivation_ln,
)
from tpu_beam_search.stream1_layernorm_pallas import pallas_layernorm_dense
from tpu_beam_search.stream1_layernorm_reference import (
    layer_norm_reference, layernorm_stream1_weights_from_artgor_params,
)


TARGET_DEVICE_COUNT = 8
LOCAL_BATCH = 256
RESULT_NAME = "artgor_pallas_same_suffix.json"
CASES = (("legal_seed_42", "legal", 42), ("stress_seed_43", "stress", 43))


def config() -> PallasExactConfig:
    return PallasExactConfig(
        embedding_bm=256, input_bm=128, input_bk=128, input_bn=256,
        residual_bm=128, residual_bk=128, residual_bn=256,
        head_bm=256, head_bk=1024, head_bn=128,
        dense_rounding="late", layernorm_arithmetic="monolithic_fp32_variance",
        skip_layernorm_arithmetic="monolithic_fp32_variance_late_skip",
    )


def _normalized(values, layer, *, epsilon, relu):
    values = values @ layer.dense.weight + layer.dense.bias
    values = layer_norm_reference(values, layer.normalization, epsilon=epsilon)
    return jax.nn.relu(values) if relu else values


def reference_hidden_after_depth(states, weights, architecture, depth: int):
    """Return the input-stack output plus `depth` complete residual blocks."""
    if not 0 <= depth <= architecture.RESIDUAL_COUNT:
        raise ValueError("depth is outside the residual stack")
    logical = states[:, :architecture.STATE_LEN]
    hidden = weights.embedding[logical.astype(jnp.int32)].reshape(
        states.shape[0], architecture.STATE_LEN * architecture.EMBED_DIM,
    )
    hidden = _normalized(
        hidden, weights.input, epsilon=architecture.LAYER_NORM_EPSILON, relu=True,
    )
    for block in weights.residuals[:depth]:
        skip = hidden
        branch = _normalized(
            hidden, block.first, epsilon=architecture.LAYER_NORM_EPSILON, relu=True,
        )
        branch = _normalized(
            branch, block.second, epsilon=architecture.LAYER_NORM_EPSILON, relu=False,
        )
        hidden = jax.nn.relu(skip + branch)
    return hidden


def reference_embedding(states, weights, architecture):
    logical = states[:, :architecture.STATE_LEN]
    return weights.embedding[logical.astype(jnp.int32)].reshape(
        states.shape[0], architecture.STATE_LEN * architecture.EMBED_DIM,
    )


def reference_input_dense(embedded, weights):
    return embedded @ weights.input.dense.weight + weights.input.dense.bias


def reference_input_ln(dense, weights, architecture):
    return jax.nn.relu(layer_norm_reference(
        dense, weights.input.normalization, epsilon=architecture.LAYER_NORM_EPSILON,
    ))


def reference_residual(hidden, block, architecture):
    branch = _normalized(hidden, block.first, epsilon=architecture.LAYER_NORM_EPSILON, relu=True)
    branch = _normalized(branch, block.second, epsilon=architecture.LAYER_NORM_EPSILON, relu=False)
    return jax.nn.relu(hidden + branch)


def residual_operator(values, skip, block, architecture, *, stage,
                      pallas_config=None, interpret=False):
    """One residual operator; both backends consume the same explicit operands."""
    if stage not in range(4):
        raise ValueError("residual stage must be 0..3")
    layer = block.first if stage < 2 else block.second
    if stage in (0, 2):
        if pallas_config is None:
            return values @ layer.dense.weight + layer.dense.bias
        return _exact_dense(
            values, layer.dense, bm=pallas_config.residual_bm,
            bk=pallas_config.residual_bk, bn=pallas_config.residual_bn,
            rounding=pallas_config.dense_rounding, interpret=interpret,
        )
    if pallas_config is None:
        normalized = layer_norm_reference(
            values, layer.normalization, epsilon=architecture.LAYER_NORM_EPSILON,
        )
        return jax.nn.relu(skip + normalized if stage == 3 else normalized)
    return pallas_exact_layer_norm_activation(
        values, layer.normalization.scale, layer.normalization.bias,
        skip=skip if stage == 3 else None, add_skip=stage == 3, relu=True,
        epsilon=architecture.LAYER_NORM_EPSILON, bm=pallas_config.residual_bm,
        arithmetic=pallas_config.layernorm_arithmetic, interpret=interpret,
    )


def residual0_suffix(pair, weights, architecture, stage):
    values, skip = pair
    for following in range(stage + 1, 4):
        values = residual_operator(values, skip, weights.residuals[0], architecture, stage=following)
    return reference_suffix(values, weights, architecture, 1)


def residual0_prefix(hidden, weights, architecture, stage):
    values = hidden
    for current in range(stage + 1):
        values = residual_operator(values, hidden, weights.residuals[0], architecture, stage=current)
    return values


def residual0_runner_bundle(mesh, weights, architecture, cfg):
    """Dense/LN controls plus factorial variance/skip-rounding A/B."""
    references, prefixes, suffixes, candidates = [], [], [], []
    spec = P("core", None)
    for stage in range(4):
        references.append(_mapped(
            lambda pair, w, s=stage: residual_operator(*pair, w.residuals[0], architecture, stage=s),
            mesh=mesh, input_spec=(spec, spec), weights_example=weights,
        ))
        prefixes.append(_mapped(
            lambda h, w, s=stage: residual0_prefix(h, w, architecture, s),
            mesh=mesh, input_spec=spec, weights_example=weights,
        ))
        suffixes.append(_mapped(
            lambda pair, w, s=stage: residual0_suffix(pair, w, architecture, s),
            mesh=mesh, input_spec=(spec, spec), weights_example=weights,
        ))
        variants = (
            [(f"bk{bk}", dataclasses.replace(cfg, residual_bk=bk)) for bk in (128, 1024)]
            if stage in (0, 2) else
            [(arithmetic, dataclasses.replace(cfg, layernorm_arithmetic=arithmetic))
             for arithmetic in ("monolithic_fp32_variance", "hlo_mixed", "legacy_bf16")]
        )
        if stage == 3:
            variants.extend(
                (arithmetic, dataclasses.replace(cfg, layernorm_arithmetic=arithmetic))
                for arithmetic in ("hlo_mixed_late_skip", "monolithic_fp32_variance_late_skip")
            )
        candidates.append({name: _mapped(
            lambda pair, w, s=stage, c=variant: residual_operator(
                *pair, w.residuals[0], architecture, stage=s, pallas_config=c,
            ), mesh=mesh, input_spec=(spec, spec), weights_example=weights,
        ) for name, variant in variants})
    return references, prefixes, suffixes, candidates


def reference_suffix_from_input_dense(dense, weights, architecture):
    hidden = layer_norm_reference(
        dense, weights.input.normalization, epsilon=architecture.LAYER_NORM_EPSILON,
    )
    return reference_suffix(jax.nn.relu(hidden), weights, architecture, 0)


def reference_suffix_from_embedding(embedded, weights, architecture):
    return reference_suffix_from_input_dense(
        reference_input_dense(embedded, weights), weights, architecture,
    )


def reference_suffix(hidden, weights, architecture, start_depth: int):
    """Run one shared JAX suffix from an externally supplied hidden boundary."""
    if not 0 <= start_depth <= architecture.RESIDUAL_COUNT:
        raise ValueError("start_depth is outside the residual stack")
    for block in weights.residuals[start_depth:]:
        skip = hidden
        branch = _normalized(
            hidden, block.first, epsilon=architecture.LAYER_NORM_EPSILON, relu=True,
        )
        branch = _normalized(
            branch, block.second, epsilon=architecture.LAYER_NORM_EPSILON, relu=False,
        )
        hidden = jax.nn.relu(skip + branch)
    return hidden @ weights.output.weight + weights.output.bias


def _mapped(call, *, mesh, input_spec, weights_example):
    specs = jax.tree.map(lambda _: P(), weights_example)
    return jax.jit(jax.shard_map(
        call, mesh=mesh, in_specs=(input_spec, specs), out_specs=P("core", None),
        check_vma=False,
    ))


def _metrics(reference, candidate):
    return _tensor_comparison(jax.block_until_ready(reference), jax.block_until_ready(candidate))


def compare_isolated_operator(values, *, reference_op, candidate_op, suffix,
                              monolithic, prefix_output):
    """Both operators consume the identical runtime tensor; prefix is diagnostic."""
    reference = jax.block_until_ready(reference_op(values))
    candidate = jax.block_until_ready(candidate_op(values))
    control_q = jax.block_until_ready(suffix(reference))
    candidate_q = jax.block_until_ready(suffix(candidate))
    return {
        "boundary": _metrics(reference, candidate),
        "candidate_vs_same_suffix": _metrics(control_q, candidate_q),
        "same_suffix_control_vs_monolithic": _metrics(monolithic, control_q),
        "isolated_reference_vs_prefix": _metrics(prefix_output, reference),
        "zero_replacement": _metrics(control_q, suffix(reference)),
    }


def run_residual0_ab(hidden, weights_d, bundle, monolithic, *, hlo_output=None):
    references, prefixes, suffixes, candidates = bundle
    rows = []
    values = hidden
    names = ("dense1", "layernorm1_relu", "dense2", "layernorm2_skip_relu")
    if hlo_output is not None:
        hlo_output.mkdir(parents=True, exist_ok=True)
    for stage, name in enumerate(names):
        pair = (values, hidden)
        prefix = jax.block_until_ready(prefixes[stage](hidden, weights_d))
        reference = references[stage]
        suffix_call = suffixes[stage]
        for variant, candidate in candidates[stage].items():
            try:
                comparison = compare_isolated_operator(
                    pair, reference_op=lambda x: reference(x, weights_d),
                    candidate_op=lambda x: candidate(x, weights_d),
                    suffix=lambda x: suffix_call((x, hidden), weights_d),
                    monolithic=monolithic, prefix_output=prefix,
                )
                rows.append({"operator": name, "variant": variant, **comparison})
                if hlo_output is not None:
                    for label, call in (("jax", reference), (variant, candidate)):
                        lowered = call.lower(pair, weights_d)
                        (hlo_output / f"{name}-{label}.stablehlo.txt").write_text(
                            str(lowered.compiler_ir(dialect="stablehlo")), encoding="utf-8",
                        )
                        (hlo_output / f"{name}-{label}.compiled.txt").write_text(
                            lowered.compile().as_text(), encoding="utf-8",
                        )
            except Exception as error:
                rows.append({"operator": name, "variant": variant,
                             "error_type": type(error).__name__, "error": str(error)})
        values = jax.block_until_ready(reference(pair, weights_d))
    return rows


def run(*, dataset: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    path = output / RESULT_NAME
    report = {"schema_version": 1, "status": "running", "cases": {}}
    checkpoint(path, report)
    try:
        devices = jax.devices()[:TARGET_DEVICE_COUNT]
        inventory = runtime_inventory()
        inventory.update(
            active_device_count=len(devices),
            all_devices_are_tpu=len(devices) == 8 and all(d.platform == "tpu" for d in devices),
        )
        if not inventory["all_devices_are_tpu"]:
            raise RuntimeError(f"requires eight TPU devices, found {devices}")
        checkpoint_path = dataset / "q555_2k_BEST.pt"
        sys.path.insert(0, str(dataset))
        from jax_model import apply as original_apply, load_params_from_pt
        with jax.default_device(jax.local_devices()[0]):
            params = load_params_from_pt(checkpoint_path)
        architecture = Stream1Architecture.from_artgor_params(params, STATE_STORAGE_LEN=150)
        weights = layernorm_stream1_weights_from_artgor_params(params, architecture)
        pweights = prepare_pallas_exact_weights(weights, architecture)
        cfg = config()
        mesh = Mesh(np.asarray(devices), ("core",))
        state_spec = P("core", None)
        state_sharding = NamedSharding(mesh, state_spec)
        original, original_weights_d = _make_original_inference(original_apply, params, mesh)
        weights_d = _replicate(weights, mesh)
        pweights_d = _replicate(pweights, mesh)

        boundary = [
            _mapped(
                lambda states, w, d=depth: reference_hidden_after_depth(states, w, architecture, d),
                mesh=mesh, input_spec=state_spec, weights_example=weights,
            ) for depth in range(architecture.RESIDUAL_COUNT + 1)
        ]
        suffix = [
            _mapped(
                lambda hidden, w, d=depth: reference_suffix(hidden, w, architecture, d),
                mesh=mesh, input_spec=state_spec, weights_example=weights,
            ) for depth in range(architecture.RESIDUAL_COUNT + 1)
        ]
        pinput = _mapped(
            lambda states, w: pallas_exact_input_block(states, w, architecture, config=cfg),
            mesh=mesh, input_spec=state_spec, weights_example=pweights,
        )
        pblocks = [
            _mapped(
                lambda hidden, w, i=index: pallas_exact_residual_block(
                    hidden, w.residuals[i], architecture, config=cfg,
                ), mesh=mesh, input_spec=state_spec, weights_example=pweights,
            ) for index in range(architecture.RESIDUAL_COUNT)
        ]
        phead = _mapped(
            lambda hidden, w: pallas_exact_head(hidden, w, config=cfg),
            mesh=mesh, input_spec=state_spec, weights_example=pweights,
        )
        pfull = _mapped(
            lambda states, w: stream1_layernorm_pallas_exact_inference(
                states, w, architecture, config=cfg,
            ), mesh=mesh, input_spec=state_spec, weights_example=pweights,
        )
        ref_embedding = _mapped(
            lambda states, w: reference_embedding(states, w, architecture),
            mesh=mesh, input_spec=state_spec, weights_example=weights,
        )
        ref_input_dense = _mapped(
            reference_input_dense, mesh=mesh, input_spec=state_spec, weights_example=weights,
        )
        suffix_embedding = _mapped(
            lambda embedded, w: reference_suffix_from_embedding(embedded, w, architecture),
            mesh=mesh, input_spec=state_spec, weights_example=weights,
        )
        suffix_input_dense = _mapped(
            lambda dense, w: reference_suffix_from_input_dense(dense, w, architecture),
            mesh=mesh, input_spec=state_spec, weights_example=weights,
        )
        pembedding = _mapped(
            lambda states, w: pallas_exact_embedding(states, w, architecture, config=cfg),
            mesh=mesh, input_spec=state_spec, weights_example=pweights,
        )
        pinput_dense = _mapped(
            lambda embedded, w: pallas_exact_input_dense(embedded, w, config=cfg),
            mesh=mesh, input_spec=state_spec, weights_example=pweights,
        )
        pinput_ln = _mapped(
            lambda dense, w: pallas_exact_layer_norm_activation(
                dense, w.input.normalization.scale, w.input.normalization.bias,
                relu=True, epsilon=architecture.LAYER_NORM_EPSILON,
                bm=cfg.input_bm, arithmetic=cfg.layernorm_arithmetic,
            ), mesh=mesh, input_spec=state_spec, weights_example=pweights,
        )
        ref_ln = _mapped(
            lambda dense, w: reference_input_ln(dense, w, architecture),
            mesh=mesh, input_spec=state_spec, weights_example=weights,
        )
        ref_blocks = [
            _mapped(
                lambda h, w, i=i: reference_residual(h, w.residuals[i], architecture),
                mesh=mesh, input_spec=state_spec, weights_example=weights,
            ) for i in range(architecture.RESIDUAL_COUNT)
        ]
        operator_bundle = residual0_runner_bundle(mesh, weights, architecture, cfg)
        raw_dense_runners = {rounding: _mapped(
            lambda x, w, r=rounding: pallas_layernorm_dense(
                x, w.input.dense.weight, w.input.dense.bias,
                bm=cfg.input_bm, bk=cfg.input_bk, bn=cfg.input_bn,
                dense_rounding=r, output_dtype=jnp.float32,
            ), mesh=mesh, input_spec=state_spec, weights_example=weights,
        ) for rounding in ("late", "bf16_before_bias")}
        mean_runners = {source: _mapped(
            lambda raw, w, s=source: pallas_preactivation_ln(
                raw, w.input.normalization.scale, w.input.normalization.bias,
                mean_source=s, bm=cfg.input_bm,
                epsilon=architecture.LAYER_NORM_EPSILON,
            ), mesh=mesh, input_spec=state_spec, weights_example=weights,
        ) for source in ("fp32", "bf16")}
        raw_bk1024 = _mapped(
            lambda x, w: pallas_layernorm_dense(
                x, w.input.dense.weight, w.input.dense.bias,
                bm=cfg.input_bm, bk=1024, bn=cfg.input_bn,
                dense_rounding="late", output_dtype=jnp.float32,
            ), mesh=mesh, input_spec=state_spec, weights_example=weights,
        )
        trace_runners = {use_pallas: jax.jit(jax.shard_map(
            lambda raw, w, p=use_pallas: input_trace(
                raw, w.input.normalization.scale, w.input.normalization.bias,
                epsilon=architecture.LAYER_NORM_EPSILON, pallas=p, bm=cfg.input_bm),
            mesh=mesh, in_specs=(state_spec, jax.tree.map(lambda _: P(), weights)),
            out_specs=tuple(P("core", None) for _ in TRACE_NAMES), check_vma=False,
        )) for use_pallas in (False, True)}
        puzzle = load_puzzle(dataset / "puzzle_info.json", state_len=150, move_count=30)
        materialized_means = {p: _mapped(
            lambda raw, w, p=p: mean_buffer(raw, pallas=p, bm=cfg.input_bm),
            mesh=mesh, input_spec=state_spec, weights_example=weights,
        ) for p in (False, True)}
        external_remainder = _mapped(
            lambda pair, w: external_mean_ln(
                pair[0], pair[1], w.input.normalization.scale, w.input.normalization.bias,
                bm=cfg.input_bm, epsilon=architecture.LAYER_NORM_EPSILON),
            mesh=mesh, input_spec=(state_spec, state_spec), weights_example=weights,
        )
        order_runners = {order: _mapped(
            lambda raw, w, order=order: mean_buffer(raw, pallas=True, bm=cfg.input_bm, order=order),
            mesh=mesh, input_spec=state_spec, weights_example=weights,
        ) for order in MEAN_ORDERS}
        for case_name, kind, seed in CASES:
            host = _make_states(puzzle, kind, seed, TARGET_DEVICE_COUNT * LOCAL_BATCH)
            states = jax.device_put(host, state_sharding)
            monolithic = jax.block_until_ready(original(states, original_weights_d))
            full_pallas = jax.block_until_ready(pfull(states, pweights_d))
            if case_name == CASES[0][0]:
                for label, runner, arguments in (
                    ("original_full", original, (states, original_weights_d)),
                    ("pallas_full", pfull, (states, pweights_d)),
                    ("jax_input_prefix", boundary[0], (states, weights_d)),
                    ("pallas_input_prefix", pinput, (states, pweights_d)),
                ):
                    lowered = runner.lower(*arguments)
                    (output / f"{label}.compiled.txt").write_text(lowered.compile().as_text(), encoding="utf-8")
                    (output / f"{label}.stablehlo.txt").write_text(str(lowered.compiler_ir(dialect="stablehlo")), encoding="utf-8")
            hidden = [jax.block_until_ready(call(states, weights_d)) for call in boundary]
            operator_ab = run_residual0_ab(
                hidden[0], weights_d, operator_bundle, monolithic,
                hlo_output=output / "residual0_hlo" if case_name == CASES[0][0] else None,
            )
            rows = []

            embedded = jax.block_until_ready(ref_embedding(states, weights_d))
            candidate_embedded = jax.block_until_ready(pembedding(states, pweights_d))
            embedded_control_q = jax.block_until_ready(suffix_embedding(embedded, weights_d))
            embedded_candidate_q = jax.block_until_ready(suffix_embedding(candidate_embedded, weights_d))
            rows.append({
                "operator": "embedding", "boundary": _metrics(embedded, candidate_embedded),
                "candidate_vs_same_suffix": _metrics(embedded_control_q, embedded_candidate_q),
                "same_suffix_control_vs_monolithic": _metrics(monolithic, embedded_control_q),
            })
            dense = jax.block_until_ready(ref_input_dense(embedded, weights_d))
            input_mean_ab = []
            prefix_q = jax.block_until_ready(suffix[0](hidden[0], weights_d))
            raw128 = jax.block_until_ready(raw_dense_runners["late"](embedded, weights_d))
            jm = jax.block_until_ready(materialized_means[False](raw128, weights_d))
            pm = jax.block_until_ready(materialized_means[True](raw128, weights_d))
            native = jax.block_until_ready(mean_runners["fp32"](raw128, weights_d))
            external_mean_ab = {"jax_vs_pallas_mean": _metrics(jm, pm), "cases": []}
            np.savez_compressed(output / f"{case_name}_mean_bits.npz",
                                jax_mean_bits=np.asarray(jm).view(np.uint16)[:, 0],
                                pallas_mean_bits=np.asarray(pm).view(np.uint16)[:, 0])
            for label, mean in (("jax", jm), ("pallas", pm)):
                candidate = jax.block_until_ready(external_remainder((raw128, mean), weights_d))
                external_mean_ab["cases"].append({
                    "source": label, "vs_native": _metrics(native, candidate),
                    "prefix": _metrics(hidden[0], candidate),
                    "same_suffix": _metrics(prefix_q, suffix[0](candidate, weights_d)),
                    "original_q": _metrics(monolithic, suffix[0](candidate, weights_d)),
                })
                save_mismatch_rows(output / f"{case_name}_{label}_mean_prefix.npz", raw128,
                                   hidden[0].astype(jnp.float32), candidate.astype(jnp.float32))
            reduction_order_ab = []
            for order, runner in order_runners.items():
                phase = "compile"
                try:
                    compiled = runner.lower(raw128, weights_d).compile()
                    if case_name == CASES[0][0]:
                        (output / f"mean_order_{order}.compiled.txt").write_text(compiled.as_text(), encoding="utf-8")
                    phase = "execute"
                    mean = jax.block_until_ready(compiled(raw128, weights_d))
                    candidate = jax.block_until_ready(external_remainder((raw128, mean), weights_d))
                    q = jax.block_until_ready(suffix[0](candidate, weights_d))
                    np.savez_compressed(output / f"{case_name}_mean_order_{order}.npz",
                                        mean_bits=np.asarray(mean).view(np.uint16)[:, 0])
                    reduction_order_ab.append({
                        "order": order, "status": "complete",
                        "finite": bool(np.isfinite(np.asarray(candidate, np.float32)).all()),
                        "mean_vs_jax": _metrics(jm, mean), "vs_native": _metrics(native, candidate),
                        "prefix": _metrics(hidden[0], candidate),
                        "same_suffix": _metrics(prefix_q, q), "original_q": _metrics(monolithic, q),
                    })
                except Exception as error:
                    reduction_order_ab.append({"order": order, "status": "error", "phase": phase,
                                               "error_type": type(error).__name__, "error": str(error)})
            if case_name == CASES[0][0]:
                for label, call, arg in (("materialized_mean_jax", materialized_means[False], raw128),
                                         ("materialized_mean_pallas", materialized_means[True], raw128),
                                         ("external_mean_remainder", external_remainder, (raw128, pm))):
                    (output / f"{label}.compiled.txt").write_text(
                        call.lower(arg, weights_d).compile().as_text(), encoding="utf-8")
            raw1024 = jax.block_until_ready(raw_bk1024(embedded, weights_d))
            input_trace_ab = {"raw_bk128_vs_bk1024": _metrics(raw128, raw1024), "cases": []}
            for bk, raw in ((128, raw128), (1024, raw1024)):
                jt = jax.block_until_ready(trace_runners[False](raw, weights_d))
                pt = jax.block_until_ready(trace_runners[True](raw, weights_d))
                np.savez_compressed(output / f"{case_name}_bk{bk}_row_sums.npz",
                                    jax_sum=np.asarray(jt[0])[:, 0],
                                    pallas_sum=np.asarray(pt[0])[:, 0])
                value = jax.block_until_ready(mean_runners["fp32"](raw, weights_d))
                trace_output = pt[-1].astype(jnp.bfloat16)
                artifact = f"{case_name}_bk{bk}_prefix_mismatches.npz"
                save_mismatch_rows(output / artifact, raw,
                                   hidden[0].astype(jnp.float32), value.astype(jnp.float32))
                input_trace_ab["cases"].append({
                    "bk": bk, "rounded_dense": _metrics(dense, raw.astype(jnp.bfloat16)),
                    "jax_vs_pallas_trace": {name: _metrics(j, p) for name, j, p in zip(TRACE_NAMES, jt, pt)},
                    "trace_output_vs_uninstrumented": _metrics(value, trace_output),
                    "prefix_boundary": _metrics(hidden[0], value),
                    "same_suffix": _metrics(prefix_q, suffix[0](value, weights_d)),
                    "original_q": _metrics(monolithic, suffix[0](value, weights_d)),
                    "mismatch_artifact": artifact,
                })
                if case_name == CASES[0][0]:
                    for label, call, arg in (("raw_dense_bk1024", raw_bk1024, embedded),
                                             ("trace_jax", trace_runners[False], raw),
                                             ("trace_pallas", trace_runners[True], raw)):
                        (output / f"{label}.compiled.txt").write_text(
                            call.lower(arg, weights_d).compile().as_text(), encoding="utf-8")
            for rounding, raw_call in raw_dense_runners.items():
                raw = jax.block_until_ready(raw_call(embedded, weights_d))
                for mean_source, ln_call in mean_runners.items():
                    value = jax.block_until_ready(ln_call(raw, weights_d))
                    q = jax.block_until_ready(suffix[0](value, weights_d))
                    input_mean_ab.append({
                        "dense_rounding": rounding, "mean_source": mean_source,
                        "rounded_dense_vs_reference": _metrics(dense, raw.astype(jnp.bfloat16)),
                        "prefix_boundary": _metrics(hidden[0], value),
                        "standalone_ln_control": _metrics(ref_ln(raw.astype(jnp.bfloat16), weights_d), value),
                        "same_suffix": _metrics(prefix_q, q),
                        "original_q": _metrics(monolithic, q),
                    })
                    if case_name == CASES[0][0]:
                        for label, call, arg in ((f"raw_dense_{rounding}", raw_call, embedded),
                                                 (f"raw_mean_{mean_source}", ln_call, raw)):
                            lowered = call.lower(arg, weights_d)
                            (output / f"{label}.compiled.txt").write_text(lowered.compile().as_text(), encoding="utf-8")
            candidate_dense = jax.block_until_ready(pinput_dense(embedded, pweights_d))
            dense_control_q = jax.block_until_ready(suffix_input_dense(dense, weights_d))
            dense_candidate_q = jax.block_until_ready(suffix_input_dense(candidate_dense, weights_d))
            rows.append({
                "operator": "input.dense", "boundary": _metrics(dense, candidate_dense),
                "candidate_vs_same_suffix": _metrics(dense_control_q, dense_candidate_q),
                "same_suffix_control_vs_monolithic": _metrics(monolithic, dense_control_q),
            })
            rows.append({
                "operator": "input.layernorm_relu",
                **compare_isolated_operator(
                    dense, reference_op=lambda x: ref_ln(x, weights_d),
                    candidate_op=lambda x: pinput_ln(x, pweights_d),
                    suffix=lambda x: suffix[0](x, weights_d),
                    monolithic=monolithic, prefix_output=hidden[0],
                ),
            })

            candidate = jax.block_until_ready(pinput(states, pweights_d))
            control_q = jax.block_until_ready(suffix[0](hidden[0], weights_d))
            candidate_q = jax.block_until_ready(suffix[0](candidate, weights_d))
            rows.append({
                "operator": "input_stack", "boundary": _metrics(hidden[0], candidate),
                "candidate_vs_same_suffix": _metrics(control_q, candidate_q),
                "same_suffix_control_vs_monolithic": _metrics(monolithic, control_q),
            })
            for index, call in enumerate(pblocks):
                rows.append({
                    "operator": f"residual.{index}",
                    **compare_isolated_operator(
                        hidden[index], reference_op=lambda x: ref_blocks[index](x, weights_d),
                        candidate_op=lambda x: call(x, pweights_d),
                        suffix=lambda x: suffix[index + 1](x, weights_d),
                        monolithic=monolithic, prefix_output=hidden[index + 1],
                    ),
                })
            candidate_q = jax.block_until_ready(phead(hidden[-1], pweights_d))
            control_q = jax.block_until_ready(suffix[-1](hidden[-1], weights_d))
            rows.append({
                "operator": "head", "boundary": _metrics(control_q, candidate_q),
                "candidate_vs_same_suffix": _metrics(control_q, candidate_q),
                "same_suffix_control_vs_monolithic": _metrics(monolithic, control_q),
            })
            report["cases"][case_name] = {
                "kind": kind, "seed": seed, "input_sha256": _array_sha256(host),
                "operators": rows,
                "residual0_operator_ab": operator_ab,
                "full_composition_vs_original": _metrics(monolithic, full_pallas),
                "input_mean_ab": input_mean_ab,
                "input_trace_ab": input_trace_ab,
                "external_mean_ab": external_mean_ab,
                "reduction_order_ab": reduction_order_ab,
            }
            checkpoint(path, report)
        report.update(
            status="complete",
            context={
                "source_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
                "runtime": inventory, "config": dataclasses.asdict(cfg),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "model_source_sha256": sha256_file(dataset / "jax_model.py"),
            },
        )
        checkpoint(path, report)
        return report
    except Exception as error:
        report.update(status="error", fatal_error_type=type(error).__name__, fatal_error=str(error), traceback=traceback.format_exc())
        checkpoint(path, report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(dataset=_dataset_path(args.dataset), output=args.output)
    print(json.dumps({"status": result["status"], "result": str(args.output / RESULT_NAME)}, indent=2))


if __name__ == "__main__":
    main()
