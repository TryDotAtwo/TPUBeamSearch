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
from benchmarks.layernorm_quality import load_puzzle
from benchmarks.stream1_layernorm_arithmetic import runtime_inventory, sha256_file
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_layernorm_pallas_exact import (
    PallasExactConfig, pallas_exact_head, pallas_exact_input_block,
    pallas_exact_residual_block, prepare_pallas_exact_weights,
)
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
        puzzle = load_puzzle(dataset / "puzzle_info.json", state_len=150, move_count=30)
        for case_name, kind, seed in CASES:
            host = _make_states(puzzle, kind, seed, TARGET_DEVICE_COUNT * LOCAL_BATCH)
            states = jax.device_put(host, state_sharding)
            monolithic = jax.block_until_ready(original(states, original_weights_d))
            hidden = [jax.block_until_ready(call(states, weights_d)) for call in boundary]
            rows = []

            candidate = jax.block_until_ready(pinput(states, pweights_d))
            control_q = jax.block_until_ready(suffix[0](hidden[0], weights_d))
            candidate_q = jax.block_until_ready(suffix[0](candidate, weights_d))
            rows.append({
                "operator": "input_stack", "boundary": _metrics(hidden[0], candidate),
                "candidate_vs_same_suffix": _metrics(control_q, candidate_q),
                "same_suffix_control_vs_monolithic": _metrics(monolithic, control_q),
            })
            for index, call in enumerate(pblocks):
                candidate = jax.block_until_ready(call(hidden[index], pweights_d))
                control_q = jax.block_until_ready(suffix[index + 1](hidden[index + 1], weights_d))
                candidate_q = jax.block_until_ready(suffix[index + 1](candidate, weights_d))
                rows.append({
                    "operator": f"residual.{index}",
                    "boundary": _metrics(hidden[index + 1], candidate),
                    "candidate_vs_same_suffix": _metrics(control_q, candidate_q),
                    "same_suffix_control_vs_monolithic": _metrics(monolithic, control_q),
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
    result = run(dataset=args.dataset or _dataset_path(), output=args.output)
    print(json.dumps({"status": result["status"], "result": str(args.output / RESULT_NAME)}, indent=2))


if __name__ == "__main__":
    main()
