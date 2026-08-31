"""One-device arithmetic/layout diagnostics; no production default changes."""

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
import numpy as np

from benchmarks.diagnostic_timing import (
    diagnostic_profile, paired_interleaved_measure, queued_measure,
)
from benchmarks.layernorm_quality import (
    inverse_valid_mask, load_puzzle, make_legal_scrambles, tensor_metrics,
)
from benchmarks.stream1_layernorm_arithmetic import (
    checkpoint, quality, reference_block, reference_prefix, reference_suffix,
    runtime_inventory, runtime_params, sha256_file,
)
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_layernorm_reference import (
    layer_norm_reference, layernorm_stream1_weights_from_artgor_params,
    stream1_layernorm_reference_inference,
)


def experiment_configs():
    base = dict(bm=128, bk=256, bn=512, control=False)
    return [
        dict(base, id="jax-graph-control", dense="jax", norm="jax", control=True),
        dict(base, id="late-dense-jax-ln", dense="late", norm="jax"),
        dict(base, id="jax-dense-legacy-unmasked", dense="jax", norm="experimental",
             arithmetic="legacy_bf16", mask_mode="none"),
        dict(base, id="jax-dense-legacy-fp32-where", dense="jax", norm="experimental",
             arithmetic="legacy_bf16", mask_mode="fp32_where"),
        dict(base, id="late-dense-legacy-unmasked", dense="late", norm="experimental",
             arithmetic="legacy_bf16", mask_mode="none"),
        dict(base, id="jax-dense-mixed", dense="jax", norm="experimental",
             arithmetic="hlo_mixed", mask_mode="fp32_where"),
        dict(base, id="late-dense-mixed", dense="late", norm="experimental",
             arithmetic="hlo_mixed", mask_mode="fp32_where"),
    ]


def candidate_block(hidden, block, epsilon, config, *, interpret=False):
    from tpu_beam_search.stream1_layernorm_experimental import experimental_layer_norm
    from tpu_beam_search.stream1_layernorm_pallas import pallas_layernorm_dense

    if config["dense"] not in ("jax", "late") or config["norm"] not in ("jax", "experimental"):
        raise ValueError("unknown diagnostic Dense/LN implementation")

    def layer(x, weights):
        if config["dense"] == "late":
            dense = pallas_layernorm_dense(
                x, weights.dense.weight, weights.dense.bias,
                bm=config["bm"], bk=config["bk"], bn=config["bn"],
                dense_rounding="late", interpret=interpret)
        else:
            dense = x @ weights.dense.weight + weights.dense.bias
        if config["norm"] == "jax":
            return layer_norm_reference(dense, weights.normalization, epsilon=epsilon)
        return experimental_layer_norm(
            dense, weights.normalization.scale, weights.normalization.bias,
            epsilon=epsilon, bm=config["bm"], arithmetic=config["arithmetic"],
            mask_mode=config["mask_mode"], interpret=interpret)

    first = jax.nn.relu(layer(hidden, block.first))
    second = layer(first, block.second)
    return jax.nn.relu(hidden + second)


def full_call(config, architecture, *, interpret=False):
    def call(states, weights):
        # One outer JIT, original JAX embedding/input and output head. Only
        # residual Dense/LN operators are replaced, never individual child MLPs.
        hidden = reference_prefix(states, weights, architecture)
        for block in weights.residuals:
            hidden = candidate_block(hidden, block, architecture.LAYER_NORM_EPSILON,
                                     config, interpret=interpret)
        return hidden @ weights.output.weight + weights.output.bias
    return call


def compile_case(call, arguments, directory, identifier):
    jax.block_until_ready(arguments)
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    lowered = jax.jit(call).lower(*arguments)
    lowering_s = time.perf_counter() - start
    # Keep source IR even if Mosaic rejects compilation. This is generated
    # experiment output, not a source-file edit.
    (directory / f"{identifier}.stablehlo.txt").write_text(
        str(lowered.compiler_ir(dialect="stablehlo")), encoding="utf-8")
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - start
    (directory / f"{identifier}.compiled.txt").write_text(compiled.as_text(), encoding="utf-8")
    start = time.perf_counter()
    out = jax.block_until_ready(compiled(*arguments))
    return out, compiled, dict(lowering_s=lowering_s, compile_s=compile_s,
                               first_execution_s=time.perf_counter() - start)


def evaluate_full_case(reference, output, mask, *, compiled, arguments, directory,
                       profile, profile_function=diagnostic_profile):
    q = quality(reference, output, mask)
    exact = bool(q["unmasked"]["finite"] and q["unmasked"]["exact_fraction"] == 1.)
    result = dict(q=q, exact_oracle_on_sample=exact, eligible_speedup=None,
                  eligible_speedup_vs_typed=None)
    if profile:
        try:
            result["profile"] = profile_function(compiled, *arguments,
                directory=directory, iterations=3)
        except Exception as exc:
            result["profile_error"] = dict(type=type(exc).__name__, message=str(exc))
    return result


def promotion_candidates(configs, rows, corpus_names):
    accepted = []
    for config in configs:
        if config.get("control", False):
            continue
        matching = [r for r in rows if r["id"] == config["id"]]
        if (len(matching) == len(corpus_names)
                and {r["corpus"] for r in matching} == set(corpus_names)
                and all(r.get("exact_oracle_on_sample") and _comparable_timing(r)
                        for r in matching)):
            accepted.append((statistics.mean(r["timing"]["median_ms"] for r in matching), config))
    return [c for _, c in sorted(accepted, key=lambda p: p[0])[:2]]


def _comparable_timing(row):
    median_ms = row.get("timing", {}).get("median_ms")
    return (row.get("status") == "ok" and row.get("timing_comparable") is True
            and median_ms is not None and math.isfinite(median_ms) and median_ms > 0)


def measure_comparison_group(cases, *, warmups, repeats):
    """Keep a failed comparison explicit; salvaged singletons are diagnostic only.

    One failing executable invalidates the entire paired comparison. Retrying
    each existing executable separately may salvage diagnostic timing, but can
    never restore comparable timing or make a candidate eligible for promotion.
    """
    try:
        measured = paired_interleaved_measure(cases, warmups=warmups, repeats=repeats)
        return dict(measured, status="ok", comparison_valid=True)
    except Exception as exc:
        result = dict(status="error", comparison_valid=False,
            label="unpaired_diagnostic_after_group_failure",
            error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()),
            warmups=warmups, repeats=repeats, cases={}, case_errors={}, execution_order=[])
    for identifier, case in cases.items():
        try:
            single = paired_interleaved_measure({identifier: case}, warmups=warmups, repeats=repeats)
            result["cases"][identifier] = single["cases"][identifier]
            result["execution_order"].extend(single["execution_order"])
        except Exception as exc:
            result["case_errors"][identifier] = dict(type=type(exc).__name__, message=str(exc))
    return result


def finalize_eligible_speedups(configs, rows, baseline_rows, corpus_names, *, batch):
    """Set speedups only after this batch's complete corpus gate is known.

    Per-corpus exactness is not sufficient. A missing/nonexact corpus or an
    invalid paired comparison keeps all eligible ratios null for that config.
    Controls retain null eligible ratios even when they match the oracle.
    """
    names = set(corpus_names)
    for row in rows:
        if row.get("batch") == batch:
            row.update(eligible_speedup=None, eligible_speedup_vs_typed=None,
                       exact_across_corpora=False)
    baselines = {(row["corpus"], row["id"]): row for row in baseline_rows
                 if row.get("batch") == batch}
    for config in configs:
        matching = [row for row in rows if row["id"] == config["id"] and row.get("batch") == batch]
        exact = (len(matching) == len(names) and {row["corpus"] for row in matching} == names
                 and all(row.get("status") == "ok" and row.get("exact_oracle_on_sample")
                         for row in matching))
        for row in matching:
            row["exact_across_corpora"] = exact
        if config.get("control", False) or not exact or not all(_comparable_timing(row) for row in matching):
            continue
        if not all(_comparable_timing(baselines.get((name, "original_runtime"), {})) for name in names):
            continue
        typed_valid = all(_comparable_timing(baselines.get((name, "typed_runtime"), {}))
            and baselines[(name, "typed_runtime")].get("quality_vs_original", {}).get("unmasked", {}).get("finite")
            and baselines[(name, "typed_runtime")].get("quality_vs_original", {}).get("unmasked", {}).get("exact_fraction") == 1.
            for name in names)
        for row in matching:
            name = row["corpus"]
            row["eligible_speedup"] = baselines[(name, "original_runtime")]["timing"]["median_ms"] / row["timing"]["median_ms"]
            if typed_valid:
                row["eligible_speedup_vs_typed"] = baselines[(name, "typed_runtime")]["timing"]["median_ms"] / row["timing"]["median_ms"]


def run_suite(params, original_apply, architecture, weights, corpora, last_moves,
              inverse, directory, **options):
    directory = Path(directory)
    path = directory / "stream1_layernorm_followup.json"
    try:
        return _run_suite(params, original_apply, architecture, weights, corpora,
                          last_moves, inverse, directory, **options)
    except Exception as exc:
        report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        report.update(status="error", fatal_error_type=type(exc).__name__,
                      fatal_error=str(exc), fatal_traceback=traceback.format_exc())
        checkpoint(path, report)
        raise


def _run_suite(params, original_apply, architecture, weights, corpora, last_moves,
               inverse, directory, *, configs=None, screen_batch=4096,
               full_batch=16384, promotion_batch=32768, warmups=5, repeats=12,
               queue_depth=8, queue_repeats=5, interpret=False,
               synthetic_probes=True, context=None):
    if min(screen_batch, full_batch, promotion_batch, repeats, queue_depth, queue_repeats, warmups) < 1:
        raise ValueError("batch/repeat/queue/warmup sizes must be positive")
    if not corpora or any(len(x) < max(screen_batch, full_batch, promotion_batch) for x in corpora.values()):
        raise ValueError("every corpus must cover all requested batches")
    configs = experiment_configs() if configs is None else configs
    if len({c["id"] for c in configs}) != len(configs):
        raise ValueError("configuration ids must be unique")
    path = directory / "stream1_layernorm_followup.json"
    report = dict(status="running", context=context or {}, architecture=asdict(architecture),
        configurations=configs, protocol=dict(
            acceptance="finite elementwise exact original monolithic Q on both corpora; not full beam/replay",
            weights="runtime FP32 original and runtime BF16 typed/candidates; captured named separately",
            selection="minimize Q; parent-major/move-minor; stable score/flat-id topK proxy",
            no_backtrack=False, mask="inverse mask additional diagnostic only",
            timing="already compiled device-resident synchronized calls; alternating forward/reverse ordering",
            queued="same executable with retained outputs, not real 128-chunk scan",
            profiles="diagnostic, independent of correctness; unsuccessful compile has no runtime timing",
            warmups=warmups, repeats=repeats, queue_depth=queue_depth, queue_repeats=queue_repeats,
            screen_batch=screen_batch, full_batch=full_batch, promotion_batch=promotion_batch),
        synthetic=[], operators=[], screen=[], controls=[], full_baselines=[], full=[], promotion=[],
        timing_groups=[], corpus_statistics={})
    checkpoint(path, report)

    def record(section, identifier, action, **details):
        row = dict(id=identifier, status="running", **details)
        report[section].append(row)
        checkpoint(path, report)
        print(f"START {section}/{identifier} {details.get('corpus', '')} {details.get('batch', '')}", flush=True)
        try:
            row.update(action())
            row["status"] = "ok"
        except Exception as exc:
            row.update(status="error", error_type=type(exc).__name__, error=str(exc),
                       traceback=traceback.format_exc())
            print(row["traceback"], flush=True)
        checkpoint(path, report)
        return row

    timing_options = dict(warmups=warmups, repeats=repeats)

    def measure_probe(call, args, expected, key):
        out, compiled, compilation = compile_case(call, args, directory / "hlo", key)
        timing = paired_interleaved_measure({key: (compiled, args)}, **timing_options)["cases"][key]
        return dict(comparison=tensor_metrics(expected, out), timing=timing, compilation=compilation,
                    timing_label="synthetic_diagnostic_singleton", timing_comparable=False)

    def time_and_queue_group(scope, corpus, batch, cases, rows):
        group = dict(scope=scope, corpus=corpus, batch=batch,
                     **measure_comparison_group(cases, **timing_options))
        report["timing_groups"].append(group)
        checkpoint(path, report)
        for identifier, (compiled, args) in cases.items():
            row = rows.get(identifier)
            if row is not None:
                row["timing_comparable"] = group["comparison_valid"]
                row["timing_label"] = group["label"]
                if identifier in group["cases"]:
                    row["timing"] = group["cases"][identifier]
                if not group["comparison_valid"]:
                    row["timing_error"] = group["case_errors"].get(identifier, group["error"])
            try:
                queued = queued_measure(compiled, *args, queue_depth=queue_depth,
                                        warmups=warmups, repeats=queue_repeats)
                group.setdefault("queued", {})[identifier] = queued
                if row is not None:
                    row["queued"] = queued
            except Exception as exc:
                error = dict(type=type(exc).__name__, message=str(exc))
                group.setdefault("queue_errors", {})[identifier] = error
                if row is not None:
                    row["queue_error"] = error
            checkpoint(path, report)
        return group

    if synthetic_probes:
        run_synthetic_probes(record, measure_probe, interpret=interpret)

    payload, metadata = runtime_params(params)
    raw = lambda x, p: original_apply({**metadata, **p}, x, dtype=jnp.bfloat16)
    typed = lambda x, w: stream1_layernorm_reference_inference(x, w, architecture)
    epsilon = architecture.LAYER_NORM_EPSILON

    for name, host_states in corpora.items():
        full_states = np.asarray(host_states[:full_batch])
        report["corpus_statistics"][name] = dict(
            batch=full_batch, unique_states=int(np.unique(full_states, axis=0).shape[0]),
            unique_state_last_move=int(np.unique(np.column_stack((full_states, last_moves[name][:full_batch])), axis=0).shape[0]),
            input_sha256=hashlib.sha256(np.asarray(host_states).tobytes()).hexdigest())
        x = jax.device_put(np.asarray(host_states[:screen_batch]))
        mask = inverse_valid_mask(np.asarray(last_moves[name][:screen_batch]), np.asarray(inverse))
        hidden = jax.jit(lambda s, w: reference_prefix(s, w, architecture))(x, weights)
        block = weights.residuals[0]
        block_reference, block_compiled, _ = compile_case(
            lambda h, b: reference_block(h, b, epsilon), (hidden, block), directory / "hlo", f"{name}-block-jax")
        suffix_args = (weights.residuals[1:], weights.output)
        control_q, suffix, suffix_compilation = compile_case(
            lambda h, b, o: reference_suffix(h, b, o, epsilon),
            (block_reference, *suffix_args), directory / "hlo", f"{name}-same-suffix")
        monolithic = jax.jit(raw)(x, payload)
        record("controls", "same-suffix-zero-replacement", lambda: dict(
            same_suffix=quality(control_q, suffix(block_reference, *suffix_args), mask),
            boundary_vs_monolithic=quality(monolithic, control_q, mask),
            suffix_compilation=suffix_compilation), corpus=name, batch=screen_batch)

        # Exact same Dense output feeds every LayerNorm arm in this section.
        first = block.first
        dense = jax.jit(lambda h, w, b: h @ w + b)(hidden, first.dense.weight, first.dense.bias)
        norm = first.normalization
        expected_ln = jax.jit(lambda v, n: layer_norm_reference(v, n, epsilon=epsilon))(dense, norm)
        from tpu_beam_search.stream1_layernorm_experimental import experimental_layer_norm
        from tpu_beam_search.stream1_layernorm_pallas import pallas_layer_norm, pallas_layernorm_dense
        operator_cases = {"dense": {}, "layernorm": {}}
        operator_rows = {}

        def compile_probe(call, args, expected, key, scope):
            out, compiled, compilation = compile_case(call, args, directory / "hlo", key)
            operator_cases[scope][key] = (compiled, args)
            return dict(comparison=tensor_metrics(expected, out), compilation=compilation)

        for identifier, call, args, expected in (
            ("ln-jax-reference", lambda v, n: layer_norm_reference(v, n, epsilon=epsilon), (dense, norm), expected_ln),
            ("ln-legacy-v1", lambda v, n: pallas_layer_norm(v, n.scale, n.bias,
                epsilon=epsilon, bm=128 if not interpret else 2, fp32_statistics=False,
                mean_mode="sum_div", interpret=interpret), (dense, norm), expected_ln),
            ("dense-jax", lambda h, w, b: h @ w + b, (hidden, first.dense.weight, first.dense.bias), dense),
            ("dense-late", lambda h, w, b: pallas_layernorm_dense(h, w, b,
                bm=128 if not interpret else 2, bk=256, bn=512, dense_rounding="late", interpret=interpret),
                (hidden, first.dense.weight, first.dense.bias), dense),
        ):
            key = f"{name}-{identifier}"
            scope = "dense" if identifier.startswith("dense-") else "layernorm"
            operator_rows[key] = record("operators", key,
                lambda c=call, a=args, e=expected, k=key, s=scope: compile_probe(c, a, e, k, s),
                corpus=name, batch=screen_batch)
        for mode in ("all", "none", "input", "center", "output", "fp32_where", "direct_2d"):
            for arithmetic in ("legacy_bf16", "hlo_mixed"):
                key = f"{name}-ln-{arithmetic}-{mode}"
                call = lambda v, n, a=arithmetic, m=mode: experimental_layer_norm(
                    v, n.scale, n.bias, epsilon=epsilon, arithmetic=a, mask_mode=m,
                    bm=128 if not interpret else 2, interpret=interpret)
                operator_rows[key] = record("operators", key,
                    lambda c=call, k=key: compile_probe(c, (dense, norm), expected_ln, k, "layernorm"),
                    corpus=name, batch=screen_batch, logical_width=int(dense.shape[1]), mask_mode=mode, arithmetic=arithmetic)

        # Compile all arms before timing. Production Dense and LN comparisons
        # each include their JAX reference in alternating-order paired rounds.
        for scope, cases in operator_cases.items():
            time_and_queue_group(scope, name, screen_batch, cases, operator_rows)

        block_cases = {"jax-reference": (block_compiled, (hidden, block))}
        block_rows = {}
        for config in configs:
            key = f"{name}-block-{config['id']}"
            def action(c=config, k=key):
                call = lambda h, b: candidate_block(h, b, epsilon, c, interpret=interpret)
                out, compiled, compilation = compile_case(call, (hidden, block), directory / "hlo", k)
                block_cases[c["id"]] = (compiled, (hidden, block))
                q = suffix(out, *suffix_args)
                return dict(hidden=tensor_metrics(block_reference, out),
                            same_suffix_q=quality(control_q, q, mask),
                            hybrid_vs_monolithic=quality(monolithic, q, mask), compilation=compilation)
            block_rows[config["id"]] = record("screen", config["id"], action, corpus=name, batch=screen_batch)
        time_and_queue_group("block", name, screen_batch, block_cases, block_rows)

    def full_batch_run(batch, selected, section):
        for name, host_states in corpora.items():
            x = jax.device_put(np.asarray(host_states[:batch]))
            mask = inverse_valid_mask(np.asarray(last_moves[name][:batch]), np.asarray(inverse))
            cases, outputs, rows = {}, {}, {}
            for identifier, call, args in (
                ("original_runtime", raw, (x, payload)), ("typed_runtime", typed, (x, weights)),
                ("captured_control", lambda s: original_apply(params, s, dtype=jnp.bfloat16), (x,))):
                def action(i=identifier, c=call, a=args):
                    out, compiled, compilation = compile_case(c, a, directory / "hlo", f"{section}-{batch}-{name}-{i}")
                    cases[i], outputs[i] = (compiled, a), out
                    result = dict(compilation=compilation)
                    if i != "original_runtime":
                        result["quality_vs_original"] = quality(outputs["original_runtime"], out, mask)
                    return result
                rows[identifier] = record("full_baselines", identifier, action, corpus=name, batch=batch)
                if identifier == "original_runtime" and rows[identifier]["status"] != "ok":
                    break
            if "original_runtime" not in outputs:
                continue
            for config in selected:
                def action(c=config):
                    args = (x, weights)
                    out, compiled, compilation = compile_case(full_call(c, architecture, interpret=interpret),
                        args, directory / "hlo", f"{section}-{batch}-{name}-{c['id']}")
                    cases[c["id"]], outputs[c["id"]] = (compiled, args), out
                    result = evaluate_full_case(outputs["original_runtime"], out, mask,
                        compiled=compiled, arguments=args, directory=directory, profile=False)
                    result["compilation"] = compilation
                    if "typed_runtime" in outputs:
                        result["typed_comparison"] = tensor_metrics(outputs["typed_runtime"], out)
                    return result
                rows[config["id"]] = record(section, config["id"], action, corpus=name, batch=batch, config=config)
            time_and_queue_group(section, name, batch, cases, rows)
            for identifier, (compiled, args) in cases.items():
                row = rows[identifier]
                median_ms = row.get("timing", {}).get("median_ms")
                row["states_per_second"] = batch * 1000 / median_ms if median_ms is not None and median_ms > 0 else None
                # Profiles are taken after timing, on legal16K only to bound
                # output size. Include baseline and compiled negative controls.
                if not interpret and section == "full" and name == "legal_scrambles":
                    try:
                        row["profile"] = diagnostic_profile(compiled, *args,
                            directory=directory / "profiles" / f"{batch}-{name}-{identifier}", iterations=3)
                    except Exception as exc:
                        row["profile_error"] = dict(type=type(exc).__name__, message=str(exc))
                checkpoint(path, report)
        finalize_eligible_speedups(selected, report[section], report["full_baselines"], tuple(corpora), batch=batch)
        checkpoint(path, report)

    full_batch_run(full_batch, configs, "full")
    winners = promotion_candidates(configs, report["full"], tuple(corpora))
    if winners and promotion_batch != full_batch:
        full_batch_run(promotion_batch, winners, "promotion")
    confirmed = promotion_candidates(winners, report["promotion"], tuple(corpora))
    report["promotion_decision"] = dict(selected_for_larger_batch=[c["id"] for c in winners],
        exact_at_larger_batch=[c["id"] for c in confirmed],
        reason="only exact non-control candidates on all16K corpora qualify;32K confirmation separate")
    if not any(r["status"] == "ok" for r in report["full"]):
        raise RuntimeError("no full case executed successfully")
    report["status"] = "complete"
    report["error_count"] = sum(r.get("status") == "error" for v in report.values()
                                if isinstance(v, list) for r in v if isinstance(r, dict))
    checkpoint(path, report)
    return report


def run_synthetic_probes(record, measure_probe, *, interpret):
    from tpu_beam_search.stream1_layernorm_experimental import experimental_layer_norm, minimal_predicate_select
    from tpu_beam_search.stream1_architecture import LayerNormWeights
    rng = np.random.default_rng(451)
    fixtures = {}
    for width in (1024, 130):
        values = jnp.asarray(rng.normal(size=(256, width)), jnp.bfloat16)
        norm = LayerNormWeights(jnp.asarray(rng.normal(size=width), jnp.bfloat16),
                               jnp.asarray(rng.normal(size=width), jnp.bfloat16))
        expected_ln = jax.jit(lambda v, n: layer_norm_reference(v, n, epsilon=1e-5))(values, norm)
        fixtures[width] = (values, norm, expected_ln)
    for bm in (128, 256):
        for width in (1024, 130):
            values, norm, expected_ln = fixtures[width]
            for dtype in ("bf16", "fp32"):
                for layout in ("broadcast", "direct_2d"):
                    key = f"predicate-bm{bm}-w{width}-{dtype}-{layout}"
                    cast = values.astype(jnp.float32 if dtype == "fp32" else jnp.bfloat16)
                    expected = jnp.where(jnp.arange(width)[None, :] % 2 == 0, cast, 0)
                    call = lambda v, d=dtype, l=layout, b=bm: minimal_predicate_select(
                        v, operand_dtype=d, predicate_layout=l, bm=b, interpret=interpret)
                    record("synthetic", key, lambda c=call, k=key: measure_probe(c, (values,), expected, k))
            modes = ("all", "none", "input", "center", "output", "fp32_where", "direct_2d") if width == 1024 else ("all", "fp32_where", "direct_2d")
            for arithmetic in ("legacy_bf16", "hlo_mixed"):
                for mode in modes:
                    key = f"ln-bm{bm}-w{width}-{arithmetic}-{mode}"
                    call = lambda v, n, a=arithmetic, m=mode, b=bm: experimental_layer_norm(
                        v, n.scale, n.bias, bm=b, arithmetic=a, mask_mode=m, interpret=interpret)
                    record("synthetic", key, lambda c=call, k=key: measure_probe(c, (values, norm), expected_ln, k))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/arithmetic_followup"))
    args = parser.parse_args()
    dataset = args.dataset or next((p for p in (
        Path("/kaggle/input/cube555-tpu-artifacts"),
        Path("/kaggle/input/datasets/artgor/cube555-tpu-artifacts")) if p.is_dir()), None)
    if dataset is None:
        raise FileNotFoundError("attach artgor/cube555-tpu-artifacts")
    inventory = runtime_inventory()
    if jax.local_devices()[0].platform != "tpu":
        raise RuntimeError("requires TPU; use interpreter tests locally")
    print(json.dumps(inventory, indent=2), flush=True)
    sys.path.insert(0, str(dataset))
    from jax_model import apply as original_apply, load_params_from_pt
    with jax.default_device(jax.local_devices()[0]):
        params = load_params_from_pt(dataset / "q555_2k_BEST.pt")
        arch = Stream1Architecture.from_artgor_params(params, STATE_STORAGE_LEN=int(params["state_size"]))
        if (arch.STATE_LEN, arch.NUM_CLASSES, arch.EMBED_DIM, arch.HIDDEN1, arch.RESIDUAL_COUNT, arch.MOVE_COUNT) != (150, 150, 24, 1024, 10, 30):
            raise ValueError("checkpoint is not the agreed Artgor Q ResMLP")
        weights = layernorm_stream1_weights_from_artgor_params(params, arch)
        puzzle = load_puzzle(dataset / "puzzle_info.json", state_len=arch.STATE_LEN, move_count=arch.MOVE_COUNT)
        legal = make_legal_scrambles(puzzle, batch=32768, seed=42)
        stress = np.random.default_rng(43).integers(0, arch.NUM_CLASSES, (32768, arch.STATE_LEN), dtype=np.uint8)
        corpora = dict(legal_scrambles=legal.states, categorical_stress=stress)
        context = dict(source_commit=subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
            runtime=inventory, checkpoint_sha256=sha256_file(dataset / "q555_2k_BEST.pt"),
            original_source_sha256=sha256_file(dataset / "jax_model.py"), puzzle_sha256=puzzle.sha256,
            input_sha256={name: hashlib.sha256(x.tobytes()).hexdigest() for name, x in corpora.items()},
            seeds=dict(legal=42, stress=43), move_names=list(puzzle.move_names),
            input_method="unchanged v1 legal stratified random walks; iid categorical stress",
            scramble_depth_counts={str(int(d)): int(np.count_nonzero(legal.lengths == d)) for d in np.unique(legal.lengths)},
            limitation="walk length is not true distance; no recorded frontiers; active device count1")
        run_suite(params, original_apply, arch, weights, corpora,
            dict(legal_scrambles=legal.last_moves, categorical_stress=np.full(32768, -1, np.int32)),
            puzzle.inverse, args.output, context=context)


if __name__ == "__main__":
    main()
