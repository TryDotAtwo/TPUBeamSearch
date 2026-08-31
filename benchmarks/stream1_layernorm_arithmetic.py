"""Controlled arithmetic and full-model TPU experiment (not a deployment gate)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.layernorm_quality import (
    inverse_valid_mask, load_puzzle, make_legal_scrambles,
    minimizing_q_metrics, tensor_metrics,
)
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_layernorm_reference import (
    layer_norm_reference, layernorm_stream1_weights_from_artgor_params,
    stream1_layernorm_reference_inference,
)


def runtime_params(params):
    """Keep source FP32 tensors runtime inputs, strings/shape metadata static."""
    array_keys = ("embed", "input_stack", "res_blocks", "head_w", "head_b")
    payload = {key: params[key] for key in array_keys}
    metadata = {key: params[key] for key in ("encoding", "num_classes", "state_size")}
    return payload, metadata


def reference_layer(hidden, layer, epsilon):
    dense = hidden @ layer.dense.weight + layer.dense.bias
    return layer_norm_reference(dense, layer.normalization, epsilon=epsilon)


def reference_prefix(states, weights, architecture):
    logical = states[:, :architecture.STATE_LEN].astype(jnp.int32)
    hidden = weights.embedding[logical].reshape(states.shape[0], -1)
    return jax.nn.relu(reference_layer(hidden, weights.input, architecture.LAYER_NORM_EPSILON))


def reference_block(hidden, block, epsilon):
    branch = jax.nn.relu(reference_layer(hidden, block.first, epsilon))
    branch = reference_layer(branch, block.second, epsilon)
    return jax.nn.relu(hidden + branch)


def reference_suffix(hidden, blocks, output, epsilon):
    for block in blocks:
        hidden = reference_block(hidden, block, epsilon)
    return hidden @ output.weight + output.bias


def compare_same_suffix(call, reference_hidden, candidate_hidden, arguments, segmented):
    """Both sides use the very same compiled suffix and runtime weights."""
    control = jax.block_until_ready(call(reference_hidden, *arguments))
    candidate = jax.block_until_ready(call(candidate_hidden, *arguments))
    return {
        "candidate_vs_same_suffix": tensor_metrics(control, candidate),
        "jax_control_vs_segmented": tensor_metrics(segmented, control),
        "candidate_vs_segmented": tensor_metrics(segmented, candidate),
    }


def measure(call, *args, warmups, repeats, hlo_path=None):
    if warmups < 0 or repeats < 1:
        raise ValueError("need nonnegative warmups and positive repeats")
    jax.block_until_ready(args)
    started = time.perf_counter()
    lowered = jax.jit(call).lower(*args)
    lowering_s = time.perf_counter() - started
    started = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - started
    if hlo_path is not None:
        hlo_path.parent.mkdir(parents=True, exist_ok=True)
        hlo_path.write_text(compiled.as_text(), encoding="utf-8")
    started = time.perf_counter()
    output = jax.block_until_ready(compiled(*args))
    first_s = time.perf_counter() - started
    for _ in range(warmups):
        jax.block_until_ready(compiled(*args))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        samples.append(time.perf_counter() - started)
    timing = {
        "lowering_s": lowering_s, "compile_s": compile_s,
        "first_execution_s": first_s, "warmups": warmups,
        "samples_s": samples, "median_s": statistics.median(samples),
        "min_s": min(samples), "max_s": max(samples),
        "scope": "synchronized device-resident single-device call; no transfers",
    }
    return output, timing, compiled


def checkpoint(path, result):
    # Serialize before opening the temporary file: invalid JSON never damages
    # the last checkpoint. This is experiment output, not a source-file edit.
    serialized = json.dumps(result, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def quality(reference, candidate, mask):
    """Keep rank identities in helper tests; summarize large arrays in reports."""
    results = {}
    for name, valid in (("unmasked", None), ("inverse_mask_diagnostic", mask)):
        count = reference.size if valid is None else int(np.count_nonzero(valid))
        if count == 0:
            results[name] = {"eligible": False, "reason": "no valid candidates"}
            continue
        value = minimizing_q_metrics(reference, candidate, valid_mask=valid,
                                     k=min(reference.shape[0], count))
        for key in ("reference_topk_flat_ids", "candidate_topk_flat_ids", "valid_counts"):
            value.pop(key, None)
        for ranking_key in ("reference_ranking", "candidate_ranking"):
            ranking = value.get(ranking_key)
            if ranking is None:
                continue
            for key in ("best_second_gaps", "best_tie_counts"):
                rows = ranking.pop(key)
                finite = np.asarray([v for v in rows if v is not None], dtype=float)
                ranking[key + "_summary"] = {
                    "missing_count": len(rows) - finite.size,
                    "zero_count": int(np.count_nonzero(finite == 0)),
                    "quantiles_0_10_50_90_100": np.quantile(finite, [0, .1, .5, .9, 1]).tolist() if finite.size else None,
                }
        results[name] = value
    return results


def cross_block(hidden, block, epsilon, config, *, replace="both", interpret=False):
    # Import lazily so the pure measurement/control tests need no candidate API.
    from tpu_beam_search.stream1_layernorm_pallas import pallas_layernorm_dense, pallas_layer_norm
    def layer(values, weights, index):
        selected = replace == "both" or replace == index
        if selected and config["dense"] == "pallas":
            dense = pallas_layernorm_dense(
                values, weights.dense.weight, weights.dense.bias,
                bm=config["bm"], bk=config["bk"], bn=config["bn"],
                dense_rounding=config["dense_rounding"], interpret=interpret)
        else:
            dense = values @ weights.dense.weight + weights.dense.bias
        if selected and config["norm"] == "pallas":
            return pallas_layer_norm(
                dense, weights.normalization.scale, weights.normalization.bias,
                bm=config["bm"], epsilon=epsilon,
                mean_mode=config["mean_mode"], fp32_statistics=config["fp32_statistics"],
                interpret=interpret)
        return layer_norm_reference(dense, weights.normalization, epsilon=epsilon)
    branch = jax.nn.relu(layer(hidden, block.first, "first"))
    branch = layer(branch, block.second, "second")
    return jax.nn.relu(hidden + branch)


def block_call(config, epsilon, *, interpret=False):
    from tpu_beam_search.stream1_layernorm_pallas import pallas_fused_dense_layer_norm, pallas_fused_residual_block
    if config["fusion"] == "cross":
        return lambda x, b: cross_block(x, b, epsilon, config, replace=config.get("replace", "both"), interpret=interpret)
    def call(x, block):
        options = {key: config[key] for key in ("bm", "bk", "bn", "dense_rounding", "mean_mode", "fp32_statistics")}
        options.update(epsilon=epsilon, interpret=interpret)
        if config["fusion"] == "per_block":
            return pallas_fused_residual_block(x, block, **options)
        def layer(values, weights, **extra):
            return pallas_fused_dense_layer_norm(
                values, weights.dense.weight, weights.dense.bias,
                weights.normalization.scale, weights.normalization.bias,
                relu=True, **extra, **options)
        first = layer(x, block.first)
        return layer(first, block.second, skip=x, add_skip=True)
    return call


def full_call(config, architecture, *, interpret=False):
    from tpu_beam_search.stream1_layernorm_pallas import stream1_layernorm_pallas_inference
    if config["fusion"] == "cross":
        def call(states, weights):
            # Explicit hybrid: JAX input/head, independent Pallas/JAX trunk ops.
            hidden = reference_prefix(states, weights, architecture)
            for block in weights.residuals:
                hidden = cross_block(hidden, block, architecture.LAYER_NORM_EPSILON,
                                     config, interpret=interpret)
            return hidden @ weights.output.weight + weights.output.bias
        return call
    return lambda states, weights: stream1_layernorm_pallas_inference(
        states, weights, architecture, bm=config["bm"],
        bk_input=config["bk"], bn_input=config["bn"],
        bk_hidden=config["bk"], bn_hidden=config["bn"],
        bk_output=config["bk"], bn_output=128,
        layernorm_fusion=config["fusion"], dense_rounding=config["dense_rounding"],
        mean_mode=config["mean_mode"], fp32_statistics=config["fp32_statistics"],
        interpret=interpret)


def experiment_configs():
    base = dict(bm=128, bk=256, bn=512, dense_rounding="late",
                mean_mode="sum_div", fp32_statistics=False)
    configs = []
    # 12 independent Dense x LN choices, with identical tiles.
    for dense, norm in itertools.product(("jax", "late", "bf16_before_bias"),
                                        ("jax", "sum_div", "mean_jax", "fp32")):
        configs.append({**base, "id": f"cross-{dense}-{norm}", "fusion": "cross",
                        "dense": "jax" if dense == "jax" else "pallas",
                        "dense_rounding": "late" if dense == "jax" else dense,
                        "norm": "jax" if norm == "jax" else "pallas",
                        "mean_mode": "jax" if norm == "mean_jax" else "sum_div",
                        "fp32_statistics": norm == "fp32"})
    # Localize each replacement to first/second residual sublayer as well.
    for dense, norm, replace in itertools.product(("jax", "bf16_before_bias"),
                                                 ("jax", "mean_jax"), ("first", "second")):
        if dense == "jax" and norm == "jax":
            continue
        source = next(c for c in configs if c["id"] == f"cross-{dense}-{norm}")
        configs.append({**source, "id": source["id"] + f"-{replace}", "replace": replace})
    # Matched BM128 boundaries + the historical larger per-layer tile. Do not
    # retry the known BM256 full-block VMEM overflow as a supposed optimization.
    for rounding, stats, (fusion, bm) in itertools.product(
        ("late", "bf16_before_bias"), (False, True),
        (("per_layer", 128), ("per_block", 128), ("per_layer", 256)),
    ):
        configs.append({**base, "id": f"{fusion}-bm{bm}-{rounding}-fp32{int(stats)}",
                        "fusion": fusion, "bm": bm, "dense_rounding": rounding,
                        "mean_mode": "jax", "fp32_statistics": stats})
    full = [
        {**base, "id": "legacy-separate", "fusion": "separate"},
        {**base, "id": "early-separate", "fusion": "separate",
         "dense_rounding": "bf16_before_bias", "mean_mode": "jax"},
    ]
    full += [c for c in configs if c["fusion"] != "cross"
             and c["dense_rounding"] == "bf16_before_bias"
             and (not c["fp32_statistics"] or c["fusion"] == "per_block")]
    full += [c for c in configs if c["id"] in
             ("cross-bf16_before_bias-jax", "cross-jax-mean_jax")]
    return configs, full


def run_suite(params, original_apply, architecture, weights, corpora, last_moves,
              inverse, directory, **options):
    try:
        return _run_suite(params, original_apply, architecture, weights, corpora,
                          last_moves, inverse, directory, **options)
    except Exception as exc:
        path = directory / "stream1_layernorm_arithmetic.json"
        report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        report.update(status="error", fatal_error_type=type(exc).__name__,
                      fatal_error=str(exc), fatal_traceback=traceback.format_exc())
        checkpoint(path, report)
        raise


def _run_suite(params, original_apply, architecture, weights, corpora, last_moves,
              inverse, directory, *, screen_configs=None, full_configs=None,
              screen_batch=4096, full_batch=16384, promotion_batch=32768,
              warmups=3, repeats=7, interpret=False, context=None):
    """One job, incremental results, runtime weights, no silent promotion."""
    if screen_configs is None or full_configs is None:
        default_screen, default_full = experiment_configs()
        screen_configs = default_screen if screen_configs is None else screen_configs
        full_configs = default_full if full_configs is None else full_configs
    minimum = max(screen_batch, full_batch, promotion_batch)
    if not corpora or any(len(states) < minimum for states in corpora.values()):
        raise ValueError("every corpus must cover all requested batch sizes")
    path = directory / "stream1_layernorm_arithmetic.json"
    report = {"status": "running", "context": context or {}, "architecture": asdict(architecture),
              "contract": {
                  "selection": "minimizing parent-major/move-minor global topK proxy",
                  "not_full_beam": "owner quotas, packed scores, receiver dedup/history not simulated",
                  "recorded_run_no_backtrack": False,
                  "inverse_mask": "additional diagnostic, not the saved notebook configuration",
                  "acceptance": "exact original Q output on these samples only; not a replay gate",
                  "timing_failed_quality": "diagnostic only, excluded from optimization winners",
                  "weights": "runtime arrays except explicitly named captured source control",
              }, "baseline_controls": [], "full_baselines": [], "controls": [],
              "operators": [], "screen": [], "full": [], "promotion": []}
    checkpoint(path, report)
    epsilon = architecture.LAYER_NORM_EPSILON
    payload, metadata = runtime_params(params)
    raw_runtime = lambda x, p: original_apply({**metadata, **p}, x, dtype=jnp.bfloat16)
    typed_runtime = lambda x, w: stream1_layernorm_reference_inference(x, w, architecture)
    suffix = jax.jit(lambda x, blocks, output: reference_suffix(x, blocks, output, epsilon))
    prefix = jax.jit(lambda x, w: reference_prefix(x, w, architecture))
    block_jax = jax.jit(lambda x, b: reference_block(x, b, epsilon))
    masks = {name: inverse_valid_mask(np.asarray(last_moves[name] if isinstance(last_moves, dict) else last_moves),
                                     np.asarray(inverse)) for name in corpora}
    timing_options = dict(warmups=warmups, repeats=repeats)

    def record(section, name, action, **info):
        entry = {"id": name, **info, "status": "running"}
        report[section].append(entry)
        checkpoint(path, report)
        print(json.dumps({"stage": section, "id": name, **info}), flush=True)
        try:
            entry.update(action())
            entry["status"] = "ok"
        except Exception as exc:
            entry.update(status="error", error_type=type(exc).__name__, error=str(exc),
                         traceback=traceback.format_exc())
            print(entry["traceback"], flush=True)
        checkpoint(path, report)
        return entry

    for corpus_name, host_states in corpora.items():
        x = jax.device_put(np.asarray(host_states[:screen_batch]))
        mask = masks[corpus_name][:screen_batch]
        # Original source and typed model are both runtime-parameter calls.
        measured_controls = {}
        for name, call, arguments in (
            ("original_runtime", raw_runtime, (x, payload)),
            ("typed_runtime", typed_runtime, (x, weights)),
            ("captured_source", lambda z: original_apply(params, z, dtype=jnp.bfloat16), (x,)),
        ):
            def action():
                out, timing, _ = measure(call, *arguments, **timing_options)
                measured_controls[name] = (out, timing)
                return {"timing": timing}
            record("baseline_controls", name, action, corpus=corpus_name, batch=screen_batch)
        if "original_runtime" not in measured_controls:
            continue  # No oracle: cannot evaluate this corpus' candidates.
        original, original_timing = measured_controls["original_runtime"]
        typed, typed_timing = measured_controls.get("typed_runtime", (None, None))
        captured, captured_timing = measured_controls.get("captured_source", (None, None))
        hiddens = []
        def build_segmented():
            hiddens.append(jax.block_until_ready(prefix(x, weights)))
            for block in weights.residuals:
                hiddens.append(jax.block_until_ready(block_jax(hiddens[-1], block)))
            return {"boundaries": len(hiddens)}
        segmented_status = record("baseline_controls", "segmented_jax", build_segmented, corpus=corpus_name)
        if segmented_status["status"] != "ok":
            continue
        segmented = jax.block_until_ready(suffix(hiddens[-1], (), weights.output))
        control = jax.block_until_ready(suffix(hiddens[1], weights.residuals[1:], weights.output))
        for depth in sorted(set((0, 1, min(3, architecture.RESIDUAL_COUNT), architecture.RESIDUAL_COUNT))):
            arguments = (weights.residuals[depth:], weights.output)
            out = jax.block_until_ready(suffix(hiddens[depth], *arguments))
            report["controls"].append({
                "corpus": corpus_name, "depth": depth,
                "same_suffix": compare_same_suffix(suffix, hiddens[depth], hiddens[depth], arguments, segmented),
                "suffix_vs_monolithic": quality(original, out, mask),
                "segmented_vs_monolithic": quality(original, segmented, mask),
                "typed_vs_original_runtime": quality(original, typed, mask) if typed is not None else None,
                "captured_vs_original_runtime": quality(original, captured, mask) if captured is not None else None,
                "original_runtime_timing": original_timing, "typed_runtime_timing": typed_timing,
                "captured_timing": captured_timing,
            })
        checkpoint(path, report)
        if not interpret:
            run_operator_probes(record, hiddens[0], weights.residuals[0], epsilon,
                                corpus_name, directory, timing_options)
        for config in screen_configs:
            def action(c=config):
                hidden, timing, _ = measure(block_call(c, epsilon, interpret=interpret),
                                            hiddens[0], weights.residuals[0], **timing_options)
                hybrid = jax.block_until_ready(suffix(hidden, weights.residuals[1:], weights.output))
                return {"hidden": tensor_metrics(hiddens[1], hidden),
                        "same_suffix_q": quality(control, hybrid, mask),
                        "hybrid_vs_original_q": quality(original, hybrid, mask),
                        "timing": timing, "states_per_second": screen_batch / timing["median_s"]}
            record("screen", config["id"], action, config=config, corpus=corpus_name, batch=screen_batch)

    def full_batch_run(batch, configs, section):
        for corpus_name, host_states in corpora.items():
            x = jax.device_put(np.asarray(host_states[:batch]))
            mask = masks[corpus_name][:batch]
            full_references = {}
            def source_baseline():
                out, timing, _ = measure(raw_runtime, x, payload, **timing_options)
                full_references["original"] = out
                return {"timing": timing, "states_per_second": batch / timing["median_s"]}
            baseline_entry = record("full_baselines", "original_runtime", source_baseline, corpus=corpus_name, batch=batch)
            if baseline_entry["status"] != "ok":
                continue
            original = full_references["original"]
            def typed_baseline():
                out, timing, _ = measure(typed_runtime, x, weights, **timing_options)
                full_references["typed"] = out
                q = quality(original, out, mask)
                baseline_entry["typed_quality_vs_original"] = q
                return {"timing": timing, "quality_vs_original": q,
                        "states_per_second": batch / timing["median_s"]}
            typed_entry = record("full_baselines", "typed_runtime", typed_baseline, corpus=corpus_name, batch=batch)
            for config in configs:
                def action(c=config):
                    out, measured, compiled = measure(full_call(c, architecture, interpret=interpret),
                                                       x, weights, **timing_options)
                    q = quality(original, out, mask)
                    exact = bool(q["unmasked"]["finite"] and q["unmasked"]["exact_fraction"] == 1.0)
                    # Recheck baseline after each candidate: report drift instead
                    # of choosing from one early baseline against a late sweep.
                    _, paired, _ = measure(raw_runtime, x, payload, **timing_options)
                    result = {"q": q, "timing": measured, "paired_baseline_timing": paired,
                              "states_per_second": batch / measured["median_s"],
                              "exact_oracle_on_sample": exact,
                              "eligible_speedup": paired["median_s"] / measured["median_s"] if exact else None}
                    if typed_entry["status"] == "ok":
                        typed_q = typed_entry["quality_vs_original"]["unmasked"]
                        _, paired_typed, _ = measure(typed_runtime, x, weights, **timing_options)
                        result.update(
                            typed_q=tensor_metrics(full_references["typed"], out),
                            paired_typed_baseline_timing=paired_typed,
                            eligible_speedup_vs_typed=(paired_typed["median_s"] / measured["median_s"]
                                if exact and typed_q["finite"] and typed_q["exact_fraction"] == 1.0 else None),
                        )
                    if exact and not interpret:
                        trace_dir = directory / "profiles" / f"{section}-{batch}-{corpus_name}-{c['id']}"
                        try:
                            with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
                                for _ in range(3):
                                    jax.block_until_ready(compiled(x, weights))
                            result["profile"] = str(trace_dir.relative_to(directory))
                        except Exception as exc:
                            result["profile_error"] = str(exc)
                    return result
                record(section, config["id"], action, config=config, corpus=corpus_name, batch=batch)
    full_batch_run(full_batch, full_configs, "full")
    # Larger-batch promotion requires exact full output in *every* corpus.
    # Approximate candidates remain diagnostic and cannot silently win on speed.
    accepted = []
    for config in full_configs:
        rows = [row for row in report["full"] if row["id"] == config["id"]]
        if len(rows) == len(corpora) and all(row.get("exact_oracle_on_sample", False) for row in rows):
            accepted.append((statistics.mean(row["timing"]["median_s"] for row in rows), config))
    winners = [config for _, config in sorted(accepted, key=lambda item: item[0])[:2]]
    if winners and promotion_batch != full_batch:
        full_batch_run(promotion_batch, winners, "promotion")
    validated = []
    for config in winners:
        rows = [r for r in report["promotion"] if r["id"] == config["id"]]
        if len(rows) == len(corpora) and all(r.get("exact_oracle_on_sample", False) for r in rows):
            validated.append(config["id"])
    report["promotion_decision"] = {
        "selected_for_larger_batch": [c["id"] for c in winners],
        "exact_at_larger_batch": validated,
        "reason": "exact output in all screening full-model corpora; inspect larger batch separately" if winners else "no exact full-model candidate; no promotion",
    }
    if not any(row.get("status") == "ok" for row in report["full"]):
        raise RuntimeError("no full-model candidate executed successfully; inspect baseline/candidate errors")
    report["status"] = "complete"
    report["error_count"] = sum(row.get("status") == "error" for value in report.values()
                                if isinstance(value, list) for row in value if isinstance(row, dict))
    checkpoint(path, report)
    return report


def run_operator_probes(record, hidden, block, epsilon, corpus_name, directory, timing_options):
    from tpu_beam_search.stream1_layernorm_pallas import pallas_layernorm_dense, pallas_layer_norm
    first = jax.jit(lambda x, l: jax.nn.relu(reference_layer(x, l, epsilon)))(hidden, block.first)
    for name, x, layer in (("first", hidden, block.first), ("second", first, block.second)):
        dense_jax = jax.jit(lambda v, w, b: v @ w + b)
        expected_dense = dense_jax(x, layer.dense.weight, layer.dense.bias)
        expected_norm = jax.jit(lambda v, n: layer_norm_reference(v, n, epsilon=epsilon))(
            expected_dense, layer.normalization)
        for rounding in ("jax", "late", "bf16_before_bias"):
            def action(mode=rounding):
                call = dense_jax if mode == "jax" else lambda v, w, b: pallas_layernorm_dense(
                    v, w, b, bm=128, bk=256, bn=512, dense_rounding=mode)
                output, timing, _ = measure(call, x, layer.dense.weight, layer.dense.bias,
                    hlo_path=directory / "hlo" / f"{corpus_name}-{name}-dense-{mode}.txt", **timing_options)
                return {"comparison": tensor_metrics(expected_dense, output), "timing": timing}
            record("operators", f"{name}-dense-{rounding}", action, corpus=corpus_name)
        for mean_mode, fp32 in (("jax_reference", False), ("sum_div", False), ("jax", False), ("jax", True)):
            def action(mode=mean_mode, stats=fp32):
                if mode == "jax_reference":
                    call = lambda v, n: layer_norm_reference(v, n, epsilon=epsilon)
                else:
                    call = lambda v, n: pallas_layer_norm(v, n.scale, n.bias, bm=128,
                        epsilon=epsilon, mean_mode=mode, fp32_statistics=stats)
                output, timing, _ = measure(call, expected_dense, layer.normalization,
                    hlo_path=directory / "hlo" / f"{corpus_name}-{name}-ln-{mode}-{stats}.txt", **timing_options)
                return {"comparison": tensor_metrics(expected_norm, output), "timing": timing}
            record("operators", f"{name}-ln-{mean_mode}-fp32{int(fp32)}", action, corpus=corpus_name)


def runtime_inventory():
    versions = {}
    for package in ("jax", "jaxlib", "libtpu", "numpy", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version, "platform": platform.platform(), "versions": versions,
        "devices": [{"id": device.id, "kind": device.device_kind,
                     "platform": device.platform, "process_index": device.process_index}
                    for device in jax.devices()],
        "process_count": jax.process_count(), "process_index": jax.process_index(),
        "local_device_count": jax.local_device_count(),
        "active_device_count": 1, "default_matmul_precision": jax.config.jax_default_matmul_precision,
        "x64_enabled": jax.config.jax_enable_x64,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/arithmetic_ab"))
    args = parser.parse_args()
    dataset = args.dataset
    if dataset is None:
        dataset = next((path for path in (
            Path("/kaggle/input/cube555-tpu-artifacts"),
            Path("/kaggle/input/datasets/artgor/cube555-tpu-artifacts"),
        ) if path.is_dir()), None)
    if dataset is None:
        raise FileNotFoundError("attach artgor/cube555-tpu-artifacts")
    inventory = runtime_inventory()
    if jax.devices()[0].platform != "tpu":
        raise RuntimeError("this benchmark requires a TPU; use CPU interpreter tests for local validation")
    print(json.dumps(inventory, indent=2), flush=True)
    sys.path.insert(0, str(dataset))
    from jax_model import apply as original_apply, load_params_from_pt
    checkpoint_path = dataset / "q555_2k_BEST.pt"
    with jax.default_device(jax.local_devices()[0]):
        params = load_params_from_pt(checkpoint_path)
        architecture = Stream1Architecture.from_artgor_params(params, STATE_STORAGE_LEN=int(params["state_size"]))
        if (architecture.STATE_LEN, architecture.NUM_CLASSES, architecture.EMBED_DIM,
                architecture.HIDDEN1, architecture.RESIDUAL_COUNT, architecture.MOVE_COUNT) != (150, 150, 24, 1024, 10, 30):
            raise ValueError("checkpoint is not the agreed Artgor cube555 Q ResMLP")
        weights = layernorm_stream1_weights_from_artgor_params(params, architecture)
        puzzle = load_puzzle(dataset / "puzzle_info.json",
                             state_len=architecture.STATE_LEN, move_count=architecture.MOVE_COUNT)
        legal = make_legal_scrambles(puzzle, batch=32768, seed=42)
        stress = np.random.default_rng(43).integers(0, architecture.NUM_CLASSES,
                                                   (32768, architecture.STATE_LEN), dtype=np.uint8)
        corpora = {"legal_scrambles": legal.states, "categorical_stress": stress}
        source_sha = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
        context = {
            "source_commit": source_sha, "runtime": inventory,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "original_source_sha256": sha256_file(dataset / "jax_model.py"),
            "puzzle_sha256": puzzle.sha256, "move_names": list(puzzle.move_names),
            "input_method": "legal: stratified random walks from solved, immediate inverse allowed; stress: iid categories",
            "seeds": {"legal": 42, "stress": 43},
            "input_sha256": {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in corpora.items()},
            "scramble_depth_counts": {str(int(d)): int(np.count_nonzero(legal.lengths == d)) for d in np.unique(legal.lengths)},
            "input_limit": "scramble length is an upper bound, not true distance; no recorded search frontiers",
        }
        run_suite(params, original_apply, architecture, weights, corpora,
                  {"legal_scrambles": legal.last_moves, "categorical_stress": np.full(32768, -1, dtype=np.int32)},
                  puzzle.inverse, args.output, context=context)


if __name__ == "__main__":
    main()
