"""Dense boundaries/tiles, LN predicates and exact flat embedding in one TPU job."""
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
import numpy as np

from benchmarks.diagnostic_timing import diagnostic_profile, queued_measure
from benchmarks.execution_boundary_ops import (
    candidate_dense, candidate_full, dense_configs, full_configs,
    jax_ln_observe, mismatch_witnesses, node_summaries, validate_provenance,
)
from benchmarks.layernorm_quality import inverse_valid_mask, load_puzzle, make_legal_scrambles, tensor_metrics
from benchmarks.stream1_layernorm_arithmetic import (
    checkpoint, quality, reference_prefix, runtime_inventory, runtime_params, sha256_file,
)
from benchmarks.stream1_layernorm_followup import (
    compile_case, finalize_eligible_speedups, measure_comparison_group, promotion_candidates,
)
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_embedding_experimental import flat_embedding
from tpu_beam_search.stream1_layernorm_experimental import experimental_layer_norm
from tpu_beam_search.stream1_layernorm_reference import (
    layer_norm_reference, layernorm_stream1_weights_from_artgor_params,
    stream1_layernorm_reference_inference,
)


def run_suite(params, original_apply, architecture, weights, corpora, last_moves,
              inverse, directory, *, configs=None, dense_cases=None, embedding_cases=None,
              screen_batch=4096, full_batch=16384, promotion_batch=32768,
              warmups=5, repeats=12, queue_depth=8, queue_repeats=5,
              interpret=False, context=None):
    configs = full_configs() if configs is None else configs
    dense_cases = dense_configs() if dense_cases is None else dense_cases
    embedding_cases = ["jax_flat", "jax_tiled", "pallas_banked"] if embedding_cases is None else embedding_cases
    counts = (screen_batch, full_batch, promotion_batch, warmups, repeats, queue_depth, queue_repeats)
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in counts):
        raise ValueError("batch/timing counts must be positive integers")
    for group in (configs, dense_cases):
        if not group or len({c["id"] for c in group}) != len(group):
            raise ValueError("configuration IDs must be unique and nonempty")
    if len(set(embedding_cases)) != len(embedding_cases) or any(
            k not in ("jax_flat", "jax_tiled", "pallas_banked") for k in embedding_cases):
        raise ValueError("embedding IDs must be unique known implementations")
    if not corpora or set(corpora) != set(last_moves):
        raise ValueError("corpora and last-move keys must match")
    for name, values in corpora.items():
        x = np.asarray(values)
        if (x.ndim != 2 or x.dtype != np.uint8 or x.shape[1] != architecture.STATE_LEN
                or x.shape[0] < max(screen_batch, full_batch, promotion_batch)
                or np.any(x >= architecture.NUM_CLASSES)
                or len(last_moves[name]) != len(x)):
            raise ValueError("corpus shape, category domain or last moves invalid")
    directory = Path(directory)
    path = directory / "stream1_execution_boundary.json"
    if path.exists():
        raise FileExistsError("use a new output directory; do not overwrite experiment evidence")
    report = dict(status="running", context=context or {}, architecture=asdict(architecture),
        configurations=configs, dense_configurations=dense_cases,
        protocol=dict(screen_batch=screen_batch, full_batch=full_batch, promotion_batch=promotion_batch,
            warmups=warmups, repeats=repeats, queue_depth=queue_depth, queue_repeats=queue_repeats,
            acceptance="finite exact monolithic Q on both16K corpora; actual32K separate",
            consumer="minimize Q; global topK is proxy, not distributed beam",
            timing="paired already-compiled resident calls; queue retains outputs, not real128chunk scan",
            candidate_parameters="runtime FP32 embedding and BF16 remaining weights; typed control allBF16",
            observations="instrumented nodes may change fusion; output observer-effect checks mandatory"),
        embedding=[], dense=[], dense_ln=[], layernorm=[], observations=[], controls=[],
        full_baselines=[], full=[], promotion=[], timing_groups=[], corpus_statistics={})
    checkpoint(path, report)

    def record(section, identifier, action, **details):
        row = dict(id=identifier, status="running", **details)
        report[section].append(row)
        checkpoint(path, report)
        print(f"START {section}/{identifier} {details.get('corpus','')} {details.get('batch','')}", flush=True)
        try:
            row.update(action())
            row["status"] = "ok"
        except Exception as exc:
            row.update(status="error", error_type=type(exc).__name__, error=str(exc),
                       traceback=traceback.format_exc())
            print(row["traceback"], flush=True)
        checkpoint(path, report)
        return row

    def compare(ref, out):
        return dict(comparison=tensor_metrics(ref, out), mismatch_witnesses=mismatch_witnesses(ref, out))

    def compile_into(cases, outputs, key, call, args, hlo_key):
        out, compiled, compilation = compile_case(call, args, directory / "hlo", hlo_key)
        cases[key], outputs[key] = (compiled, args), out
        analysis = compiled.memory_analysis()
        if analysis is not None:
            compilation["static_memory_bytes_not_hardware_counters"] = {
                name: int(getattr(analysis, name)) for name in
                ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")}
        return out, compilation

    def measure(scope, name, batch, cases, rows, profile=False):
        if not cases:
            return
        group = dict(scope=scope, corpus=name, batch=batch,
                     **measure_comparison_group(cases, warmups=warmups, repeats=repeats))
        report["timing_groups"].append(group)
        for key, (compiled, args) in cases.items():
            row = rows[key]
            row.update(timing_comparable=group["comparison_valid"], timing_label=group["label"])
            if key in group["cases"]:
                row["timing"] = group["cases"][key]
                ms = row["timing"]["median_ms"]
                row["states_per_second"] = batch * 1000 / ms if ms > 0 else None
            if not group["comparison_valid"]:
                row["timing_error"] = group["case_errors"].get(key, group["error"])
            try:
                row["queued"] = queued_measure(compiled, *args, queue_depth=queue_depth,
                    warmups=warmups, repeats=queue_repeats)
            except Exception as exc:
                row["queue_error"] = dict(type=type(exc).__name__, message=str(exc))
            if profile and not interpret:
                try:
                    row["profile"] = diagnostic_profile(compiled, *args,
                        directory=directory / "profiles" / f"{scope}-{batch}-{name}-{key}", iterations=3)
                except Exception as exc:
                    row["profile_error"] = dict(type=type(exc).__name__, message=str(exc))
            checkpoint(path, report)

    payload, metadata = runtime_params(params)
    raw = lambda s, p: original_apply({**metadata, **p}, s, dtype=jnp.bfloat16)
    typed = lambda s, w: stream1_layernorm_reference_inference(s, w, architecture)
    # Do not silently charge a pretyped embedding to the raw-parameter candidate.
    candidate_weights = weights._replace(embedding=jnp.asarray(params["embed"], jnp.float32))
    epsilon = architecture.LAYER_NORM_EPSILON

    def operators():
        for name, host_states in corpora.items():
            states = jax.device_put(np.asarray(host_states[:screen_batch]))
            xfull = np.asarray(host_states[:full_batch])
            report["corpus_statistics"][name] = dict(
                input_sha256=hashlib.sha256(np.asarray(host_states).tobytes()).hexdigest(),
                unique_states=int(np.unique(xfull, axis=0).shape[0]), batch=full_batch)
            hidden = jax.jit(lambda s, w: reference_prefix(s, w, architecture))(states, weights)
            layer = weights.residuals[0].first
            dense_ref_fn = jax.jit(lambda x, p: x @ p.dense.weight + p.dense.bias)
            dense_ref = dense_ref_fn(hidden, layer)
            ln_compiled = jax.jit(lambda x, n: layer_norm_reference(x, n, epsilon=epsilon)).lower(
                dense_ref, layer.normalization).compile()
            ln_ref = ln_compiled(dense_ref, layer.normalization)
            dense_ln_ref = jax.jit(lambda x, p: layer_norm_reference(
                x @ p.dense.weight + p.dense.bias, p.normalization, epsilon=epsilon))(hidden, layer)
            record("controls", "separate-dense-ln-vs-composed-jax", lambda: compare(dense_ln_ref, ln_ref),
                   corpus=name, batch=screen_batch)

            for scope in ("dense", "dense_ln"):
                cases, outputs, rows = {}, {}, {}
                for config in dense_cases:
                    def call(x, p, c=config):
                        value = candidate_dense(x, p.dense.weight, p.dense.bias, c, interpret=interpret)
                        return value if scope == "dense" else layer_norm_reference(value, p.normalization, epsilon=epsilon)
                    def action(c=config, fn=call):
                        out, compilation = compile_into(cases, outputs, c["id"], fn, (hidden, layer),
                                                        f"{scope}-{name}-{c['id']}")
                        result = dict(compilation=compilation, **compare(dense_ref if scope == "dense" else dense_ln_ref, out))
                        if scope == "dense":
                            result["same_compiled_ln"] = compare(ln_ref, ln_compiled(out, layer.normalization))
                        return result
                    rows[config["id"]] = record(scope, config["id"], action, corpus=name,
                                                  batch=screen_batch, config=config)
                measure(scope, name, screen_batch, cases, rows)

            cases, outputs, rows = {}, {}, {}
            ln_arms = [("reference", None, None), ("legacy-unmasked", "legacy_bf16", "none"),
                       ("mixed-unmasked", "hlo_mixed", "none"), ("mixed-direct2d", "hlo_mixed", "direct_2d"),
                       ("mixed-masked", "hlo_mixed", "fp32_where")]
            for key, arithmetic, mode in ln_arms:
                def call(x, n, a=arithmetic, m=mode):
                    if a is None:
                        return layer_norm_reference(x, n, epsilon=epsilon)
                    return experimental_layer_norm(x, n.scale, n.bias, epsilon=epsilon,
                        arithmetic=a, mask_mode=m, bm=8 if interpret else 128,
                        alignment=x.shape[1] if interpret else 128, interpret=interpret)
                def action(k=key, fn=call):
                    out, compilation = compile_into(cases, outputs, k, fn, (dense_ref, layer.normalization), f"ln-{name}-{k}")
                    return dict(compilation=compilation, **compare(ln_ref, out))
                rows[key] = record("layernorm", key, action, corpus=name, batch=screen_batch)
            if "mixed-unmasked" in outputs and "mixed-direct2d" in outputs:
                record("controls", "mixed-direct2d-vs-unmasked", lambda: compare(outputs["mixed-unmasked"], outputs["mixed-direct2d"]),
                       corpus=name, batch=screen_batch)
            measure("layernorm", name, screen_batch, cases, rows)

            def observe(kind):
                if kind == "ln":
                    fn = lambda x, p: jax_ln_observe(x, p.normalization, epsilon)
                    args, expected = (dense_ref, layer), ln_ref
                else:
                    def fn(x, p):
                        d = x @ p.dense.weight + p.dense.bias
                        return dict(dense=d, **jax_ln_observe(d, p.normalization, epsilon))
                    args, expected = (hidden, layer), dense_ln_ref
                out, _, compilation = compile_case(fn, args, directory / "hlo", f"observe-{kind}-{name}")
                return dict(compilation=compilation, output_observer_effect=compare(expected, out["output"]),
                    node_summaries=node_summaries(out),
                    attribution="observed JAX graph only; not Pallas internal statistics or original machine arithmetic")
            for kind in ("ln", "dense-ln"):
                record("observations", kind, lambda k=kind: observe(k), corpus=name, batch=screen_batch)

            cases, outputs, rows = {}, {}, {}
            embed_ref = jax.jit(lambda s, e: e.astype(jnp.bfloat16)[s.astype(jnp.int32)].reshape(s.shape[0], -1))(
                states, candidate_weights.embedding)
            for key in ["reference", "typed_reference", *embedding_cases]:
                table = weights.embedding if key == "typed_reference" else candidate_weights.embedding
                def call(s, e, k=key):
                    if k in ("reference", "typed_reference"):
                        return e.astype(jnp.bfloat16)[s.astype(jnp.int32)].reshape(s.shape[0], -1)
                    return flat_embedding(s, e, implementation=k, bm=8 if interpret else 128, interpret=interpret)
                def action(k=key, fn=call, e=table):
                    out, compilation = compile_into(cases, outputs, k, fn, (states, e), f"embedding-{name}-{k}")
                    return dict(compilation=compilation, parameter_dtype=str(e.dtype), **compare(embed_ref, out))
                rows[key] = record("embedding", key, action, corpus=name, batch=screen_batch)
            measure("embedding", name, screen_batch, cases, rows)

    def full(batch, selected, section):
        for name, host_states in corpora.items():
            states = jax.device_put(np.asarray(host_states[:batch]))
            mask = inverse_valid_mask(np.asarray(last_moves[name][:batch]), np.asarray(inverse))
            cases, outputs, rows = {}, {}, {}
            for key, fn, args in (("original_runtime", raw, (states, payload)), ("typed_runtime", typed, (states, weights))):
                def action(k=key, f=fn, a=args):
                    out, compilation = compile_into(cases, outputs, k, f, a, f"{section}-{batch}-{name}-{k}")
                    result = dict(compilation=compilation)
                    if k != "original_runtime":
                        result["quality_vs_original"] = quality(outputs["original_runtime"], out, mask)
                    return result
                rows[key] = record("full_baselines", key, action, corpus=name, batch=batch)
                if key == "original_runtime" and rows[key]["status"] != "ok":
                    raise RuntimeError("original monolithic oracle failed")
            for config in selected:
                def action(c=config):
                    out, compilation = compile_into(cases, outputs, c["id"], candidate_full(c, architecture, interpret=interpret),
                        (states, candidate_weights), f"{section}-{batch}-{name}-{c['id']}")
                    q = quality(outputs["original_runtime"], out, mask)
                    return dict(compilation=compilation, q=q, exact_oracle_on_sample=bool(
                        q["unmasked"]["finite"] and q["unmasked"]["exact_fraction"] == 1.),
                        mismatch_witnesses=mismatch_witnesses(outputs["original_runtime"], out),
                        eligible_speedup=None, eligible_speedup_vs_typed=None)
                rows[config["id"]] = record(section, config["id"], action, corpus=name, batch=batch, config=config)
            measure(section, name, batch, cases, rows, profile=section == "full" and name == "legal_scrambles")
        finalize_eligible_speedups(selected, report[section], report["full_baselines"], tuple(corpora), batch=batch)
        checkpoint(path, report)

    try:
        operators()
        full(full_batch, configs, "full")
        if not any(row["status"] == "ok" for row in report["full"]):
            raise RuntimeError("no full case executed successfully")
        selected = promotion_candidates(configs, report["full"], tuple(corpora))
        if selected and promotion_batch != full_batch:
            full(promotion_batch, selected, "promotion")
        confirmed = promotion_candidates(selected, report["promotion"], tuple(corpora))
        report["promotion_decision"] = dict(selected_for_larger_batch=[c["id"] for c in selected],
            exact_at_larger_batch=[c["id"] for c in confirmed],
            reason="all-corpus exact Q and comparable timing; controls excluded;32K separate")
        report["status"] = "complete"
        report["error_count"] = sum(r.get("status") == "error" for values in report.values()
            if isinstance(values, list) for r in values if isinstance(r, dict))
        checkpoint(path, report)
        return report
    except Exception as exc:
        report.update(status="error", fatal_error=str(exc), fatal_error_type=type(exc).__name__,
                      fatal_traceback=traceback.format_exc())
        checkpoint(path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/execution_boundary"))
    args = parser.parse_args()
    dataset = args.dataset or next((p for p in (Path("/kaggle/input/cube555-tpu-artifacts"),
        Path("/kaggle/input/datasets/artgor/cube555-tpu-artifacts")) if p.is_dir()), None)
    if dataset is None:
        raise FileNotFoundError("attach artgor/cube555-tpu-artifacts")
    inventory = runtime_inventory()
    if jax.local_devices()[0].platform != "tpu":
        raise RuntimeError("requires TPU; local validation uses interpreter tests")
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
        corpora = dict(legal_scrambles=legal.states, categorical_stress=np.random.default_rng(43).integers(
            0, arch.NUM_CLASSES, (32768, arch.STATE_LEN), dtype=np.uint8))
        context = dict(source_commit=subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
            runtime=inventory, checkpoint_sha256=sha256_file(dataset / "q555_2k_BEST.pt"),
            original_source_sha256=sha256_file(dataset / "jax_model.py"), puzzle_sha256=puzzle.sha256,
            input_sha256={name: hashlib.sha256(x.tobytes()).hexdigest() for name, x in corpora.items()},
            seeds=dict(legal=42, stress=43), move_names=list(puzzle.move_names),
            active_device_count=1, input_method="unchanged legal walks and categorical stress, not real frontiers")
        prior = Path(__file__).resolve().parents[1] / "test_results/kaggle_layernorm_followup_v1/arithmetic_followup/stream1_layernorm_followup.json"
        previous = json.loads(prior.read_text(encoding="utf-8"))["context"]
        validate_provenance(context, previous)
        print(json.dumps(context, indent=2), flush=True)
        run_suite(params, original_apply, arch, weights, corpora,
            dict(legal_scrambles=legal.last_moves, categorical_stress=np.full(32768, -1, np.int32)),
            puzzle.inverse, args.output, context=context)


if __name__ == "__main__":
    main()
