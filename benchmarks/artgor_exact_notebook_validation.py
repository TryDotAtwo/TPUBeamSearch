"""Publication gate for the exact accelerated Artgor Cube555 TPU notebook."""
from __future__ import annotations

import argparse
import csv
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

from benchmarks.layernorm_quality import load_puzzle, make_legal_scrambles
from benchmarks.stream1_layernorm_arithmetic import (
    runtime_inventory,
    runtime_params,
    sha256_file,
)
from tpu_beam_search.artgor_exact_inference import ArtgorExactConfig
from tpu_beam_search.artgor_staged_beam import (
    StagedDepthConfig,
    beam_solve_v_only_spmd_packed_exact,
    build_staged_depth_executables,
    prepare_artgor_exact_beam_runtime,
    run_staged_depth,
)


TARGET_DEVICE_COUNT = 8
INFERENCE_LOCAL_BATCH = 32_768
PARITY_B_LOCAL = 131_072
PARITY_STEPS = 3
REAL_B_LOCAL = 2_097_152
REAL_PID = 1034
MIN_INFERENCE_SPEEDUP = 1.5
RESULT_NAME = "artgor_exact_notebook_validation.json"
OUTPUT_NAMES = (
    "states",
    "packed_backpointer",
    "min_v_log",
    "found_step",
    "found_pos_local",
    "found_pos_rank",
    "verify_state",
    "history",
    "in_move",
    "trace_hit",
    "trace_pos",
    "trace_v",
    "trace_cutoff",
)


def checkpoint(path: Path, report: dict) -> None:
    serialized = json.dumps(report, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def _get(document: dict, *path, default=None):
    node = document
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def decide_publication(report: dict) -> dict:
    """Apply immutable publication gates; do not infer missing evidence."""

    speedup = _get(report, "inference", "speedup")
    speedup_ok = (
        isinstance(speedup, (int, float))
        and not isinstance(speedup, bool)
        and math.isfinite(speedup)
        and speedup >= MIN_INFERENCE_SPEEDUP
    )
    gates = {
        "eight_tpu_devices": _get(
            report, "context", "runtime", "active_device_count"
        )
        == TARGET_DEVICE_COUNT,
        "legal_inference_bitwise_exact": _get(
            report, "inference", "legal_scrambles", "exact"
        )
        is True,
        "stress_inference_bitwise_exact": _get(
            report, "inference", "categorical_stress", "exact"
        )
        is True,
        "inference_speedup_at_least_1_5x": speedup_ok,
        "one_depth_all_tensors_exact": _get(
            report, "one_depth", "all_tensor_hashes_equal"
        )
        is True,
        "short_frontiers_exact": _get(
            report, "short_solve", "frontiers_equal"
        )
        is True,
        "short_backpointers_exact": _get(
            report, "short_solve", "backpointers_equal"
        )
        is True,
        "real_pid_1034_found": (
            _get(report, "real_solve", "pid") == REAL_PID
            and _get(report, "real_solve", "sym") == 0
            and _get(report, "real_solve", "inverted") is False
            and _get(report, "real_solve", "found") is True
        ),
        "real_path_replay_valid": _get(
            report, "real_solve", "verify_ok"
        )
        is True,
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "publishable": not failed,
        "gates": gates,
        "failed_gates": failed,
        "minimum_inference_speedup": MIN_INFERENCE_SPEEDUP,
        "inference_speedup": speedup,
    }


def _array_sha256(value) -> str:
    array = np.asarray(value)
    return hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest()


def _tensor_comparison(reference, candidate) -> dict:
    expected = np.asarray(reference)
    actual = np.asarray(candidate)
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return {
            "exact": False,
            "reference_shape": list(expected.shape),
            "candidate_shape": list(actual.shape),
            "reference_dtype": str(expected.dtype),
            "candidate_dtype": str(actual.dtype),
            "reference_sha256": _array_sha256(expected),
            "candidate_sha256": _array_sha256(actual),
            "mismatch_count": None,
            "first_mismatch": None,
        }
    unequal = expected != actual
    mismatch_count = int(np.count_nonzero(unequal))
    first = None
    if mismatch_count:
        index = tuple(int(v) for v in np.argwhere(unequal)[0])
        first = {
            "index": list(index),
            "reference": expected[index].item(),
            "candidate": actual[index].item(),
        }
    return {
        "exact": mismatch_count == 0,
        "shape": list(expected.shape),
        "dtype": str(expected.dtype),
        "reference_sha256": _array_sha256(expected),
        "candidate_sha256": _array_sha256(actual),
        "mismatch_count": mismatch_count,
        "first_mismatch": first,
    }


def _dataset_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in (
        Path("/kaggle/input/cube555-tpu-artifacts"),
        Path("/kaggle/input/datasets/artgor/cube555-tpu-artifacts"),
    ):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("attach artgor/cube555-tpu-artifacts")


def _competition_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in (
        Path("/kaggle/input/cayley-py-555-cube"),
        Path("/kaggle/input/competitions/cayley-py-555-cube"),
    ):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("attach cayley-py-555-cube competition")


def _replicate(tree, mesh):
    sharding = NamedSharding(mesh, P())
    return jax.tree.map(lambda value: jax.device_put(value, sharding), tree)


def _make_original_inference(original_apply, params, mesh):
    payload, metadata = runtime_params(params)
    weight_specs = jax.tree.map(lambda _: P(), payload)

    def local(states, runtime_payload):
        return original_apply(
            {**metadata, **runtime_payload}, states, dtype=jnp.bfloat16
        )

    mapped = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P("core", None), weight_specs),
            out_specs=P("core", None),
            check_vma=False,
        )
    )
    return mapped, _replicate(payload, mesh)


def _measure_pair(original_call, exact_call, *, warmups=2, repeats=7):
    first = {}
    for name, call in (("original_jax", original_call), ("exact_split", exact_call)):
        started = time.perf_counter()
        output = jax.block_until_ready(call())
        first[name] = {
            "s": time.perf_counter() - started,
            "output": output,
        }
    for _ in range(warmups):
        jax.block_until_ready(original_call())
        jax.block_until_ready(exact_call())
    samples = {"original_jax": [], "exact_split": []}
    order_log = []
    for repeat in range(repeats):
        order = (
            ("original_jax", original_call),
            ("exact_split", exact_call),
        )
        if repeat % 2:
            order = tuple(reversed(order))
        for name, call in order:
            started = time.perf_counter()
            jax.block_until_ready(call())
            samples[name].append(time.perf_counter() - started)
            order_log.append(name)
    timing = {
        name: {
            "first_s": first[name]["s"],
            "samples_s": values,
            "median_s": statistics.median(values),
            "min_s": min(values),
            "max_s": max(values),
        }
        for name, values in samples.items()
    }
    timing["warmups"] = warmups
    timing["repeats"] = repeats
    timing["execution_order"] = order_log
    return first["original_jax"]["output"], first["exact_split"]["output"], timing


def run_inference_gate(
    *,
    original_apply,
    params,
    runtime,
    puzzle,
    mesh,
):
    global_batch = TARGET_DEVICE_COUNT * INFERENCE_LOCAL_BATCH
    legal = make_legal_scrambles(
        puzzle, batch=global_batch, seed=42
    ).states
    stress = np.random.default_rng(43).integers(
        0,
        150,
        (global_batch, 150),
        dtype=np.uint8,
    )
    corpora = {
        "legal_scrambles": legal,
        "categorical_stress": stress,
    }
    state_sharding = NamedSharding(mesh, P("core", None))
    states_d = {
        name: jax.device_put(values, state_sharding)
        for name, values in corpora.items()
    }
    original_inference, payload_d = _make_original_inference(
        original_apply, params, mesh
    )
    weights_d = _replicate(runtime.weights, mesh)

    results = {}
    ratios = []
    for name in corpora:
        original_output, exact_output, timing = _measure_pair(
            lambda name=name: original_inference(states_d[name], payload_d),
            lambda name=name: runtime.inference(states_d[name], weights_d),
        )
        comparison = _tensor_comparison(original_output, exact_output)
        ratio = (
            timing["original_jax"]["median_s"]
            / timing["exact_split"]["median_s"]
        )
        ratios.append(ratio)
        results[name] = {
            **comparison,
            "input_sha256": _array_sha256(corpora[name]),
            "global_batch": global_batch,
            "local_batch": INFERENCE_LOCAL_BATCH,
            "timing": timing,
            "speedup": ratio,
        }
        print(
            f"INFERENCE {name}: exact={comparison['exact']} "
            f"speedup={ratio:.4f}x",
            flush=True,
        )
    results["speedup"] = min(ratios)
    results["speedup_scope"] = (
        "minimum paired median exact full-Q forward speedup across legal and "
        "categorical corpora; 8 TPU devices, 32768 states/device"
    )
    return results, corpora, weights_d


def _initial_carry(
    *,
    mesh,
    axis_name: str,
    b_local: int,
    state_size: int,
    num_steps: int,
    history_depth: int,
):
    sharding = NamedSharding(mesh, P(axis_name))
    world = int(mesh.size)
    host = (
        np.full((world, num_steps), 1e6, np.float32),
        np.full((world,), -1, np.int32),
        np.full((world,), -1, np.int32),
        np.full((world,), -1, np.int32),
        np.zeros((world, state_size), np.uint8),
        np.full((world, max(1, history_depth), b_local), -1, np.int64),
        np.full((world, b_local), -1, np.int8),
    )
    return tuple(jax.device_put(value, sharding) for value in host)


def _named_output_comparisons(original, exact):
    return {
        name: _tensor_comparison(reference, candidate)
        for name, reference, candidate in zip(OUTPUT_NAMES, original, exact)
    }


def run_depth_and_short_gate(
    *,
    original_beam,
    params,
    runtime,
    weights_d,
    puzzle,
    inference_corpora,
    exact_mesh,
):
    b_local = PARITY_B_LOCAL
    world = TARGET_DEVICE_COUNT
    n_gen = len(puzzle.move_names)
    state_size = puzzle.solved.size
    parent_chunk = 131_072
    history_depth = 4
    k_per_peer = (2 * b_local) // world
    global_rows = world * b_local
    legal = inference_corpora["legal_scrambles"]
    repeats = global_rows // len(legal)
    if repeats * len(legal) != global_rows:
        raise ValueError("parity geometry must tile the inference corpus")
    states_host = np.tile(legal, (repeats, 1)).reshape(
        world, b_local, state_size
    )

    all_moves = jnp.asarray(puzzle.moves, dtype=jnp.int32)
    v0 = jnp.asarray(puzzle.solved, dtype=jnp.uint8)
    hash_vec = jnp.asarray(
        np.random.default_rng(0).integers(
            0, int(1e15), size=state_size, dtype=np.int64
        )
    )
    owner_hash_vec = jnp.asarray(
        np.random.default_rng(12345).integers(
            0,
            np.iinfo(np.uint32).max,
            size=state_size,
            dtype=np.uint32,
        )
    )
    v0_hash = jnp.int64(
        int(np.sum(puzzle.solved.astype(np.int64) * np.asarray(hash_vec)))
    )

    original_mesh = Mesh(np.asarray(jax.devices()[:world]), ("cores",))
    original_body = original_beam._build_step_body_v_only_packed_streaming(
        params,
        all_moves,
        v0,
        hash_vec,
        v0_hash,
        b_local,
        world,
        k_per_peer,
        n_gen,
        state_size,
        jnp.bfloat16,
        INFERENCE_LOCAL_BATCH,
        parent_chunk=parent_chunk,
        pack_v_score=True,
        owner_hash_vec=owner_hash_vec,
        history_depth=history_depth,
        history_exact=True,
        q_mode=True,
    )
    original_step = jax.jit(
        jax.shard_map(
            original_body,
            mesh=original_mesh,
            in_specs=(P("cores"),) * 8 + (P(),),
            out_specs=(P("cores"),) * 13,
            check_vma=False,
        )
    )

    exact_config = ArtgorExactConfig(
        prefix_bm=4096,
        head_bm=256,
        head_bk=1024,
        head_bn=128,
        dense_rounding="late",
        inference_chunk=INFERENCE_LOCAL_BATCH,
        parent_chunk=parent_chunk,
    )
    exact_executables = build_staged_depth_executables(
        runtime.inference,
        runtime.weights,
        mesh=exact_mesh,
        config=StagedDepthConfig(
            world_size=world,
            b_local=b_local,
            inference_chunk=INFERENCE_LOCAL_BATCH,
            parent_chunk=parent_chunk,
            n_gen=n_gen,
            state_size=state_size,
        ),
        all_moves=all_moves,
        V0=v0,
        hash_vec=hash_vec,
        V0_hash=v0_hash,
        K_per_peer=k_per_peer,
        owner_hash_vec=owner_hash_vec,
        history_depth=history_depth,
        history_exact=True,
        prefix_local=runtime.prefix_local,
        head_local=runtime.head_local,
        donate_search_carry=False,
    )
    # Keep the exact config in the report even though the low-level builder
    # receives its equivalent geometry fields separately.
    exact_config.validate()

    original_states = jax.device_put(
        states_host, NamedSharding(original_mesh, P("cores"))
    )
    exact_states = jax.device_put(
        states_host, NamedSharding(exact_mesh, P("core"))
    )
    original_carry = _initial_carry(
        mesh=original_mesh,
        axis_name="cores",
        b_local=b_local,
        state_size=state_size,
        num_steps=PARITY_STEPS + 1,
        history_depth=history_depth,
    )
    exact_carry = _initial_carry(
        mesh=exact_mesh,
        axis_name="core",
        b_local=b_local,
        state_size=state_size,
        num_steps=PARITY_STEPS + 1,
        history_depth=history_depth,
    )

    depth_records = []
    frontiers_equal = True
    backpointers_equal = True
    all_outputs_equal = True
    original_times = []
    exact_times = []
    for step in range(1, PARITY_STEPS + 1):
        started = time.perf_counter()
        original_output = jax.block_until_ready(
            original_step(
                original_states,
                *original_carry,
                jnp.int32(step),
            )
        )
        original_s = time.perf_counter() - started
        started = time.perf_counter()
        exact_output = jax.block_until_ready(
            run_staged_depth(
                exact_executables,
                exact_states,
                weights_d,
                *exact_carry,
                jnp.int32(step),
            )
        )
        exact_s = time.perf_counter() - started
        comparisons = _named_output_comparisons(
            original_output, exact_output
        )
        step_exact = all(item["exact"] for item in comparisons.values())
        frontiers_equal &= comparisons["states"]["exact"]
        backpointers_equal &= comparisons["packed_backpointer"]["exact"]
        all_outputs_equal &= step_exact
        depth_records.append(
            {
                "step": step,
                "all_tensors_exact": step_exact,
                "tensors": comparisons,
                "timing": {
                    "original_jax_s": original_s,
                    "exact_staged_s": exact_s,
                },
            }
        )
        original_times.append(original_s)
        exact_times.append(exact_s)
        print(
            f"DEPTH {step}: exact={step_exact} "
            f"original={original_s:.3f}s staged={exact_s:.3f}s",
            flush=True,
        )
        original_states = original_output[0]
        exact_states = exact_output[0]
        original_carry = tuple(original_output[index] for index in range(2, 9))
        exact_carry = tuple(exact_output[index] for index in range(2, 9))

    steady_original = original_times[1:] or original_times
    steady_exact = exact_times[1:] or exact_times
    depth_speedup = statistics.median(steady_original) / statistics.median(
        steady_exact
    )
    one_depth = {
        "b_local": b_local,
        "global_beam": world * b_local,
        "all_tensor_hashes_equal": all(
            item["exact"] for item in depth_records[0]["tensors"].values()
        ),
        "tensors": depth_records[0]["tensors"],
    }
    short = {
        "steps": PARITY_STEPS,
        "frontiers_equal": frontiers_equal,
        "backpointers_equal": backpointers_equal,
        "all_outputs_equal": all_outputs_equal,
        "depths": depth_records,
        "steady_depth_speedup": depth_speedup,
        "speedup_scope": (
            "paired 8-TPU 131072-parent/core depth after first compile-bearing "
            "depth; includes inference, routing, communication, dedup and top-K"
        ),
    }
    return one_depth, short


def _read_competition_state(competition: Path, pid: int) -> np.ndarray:
    with (competition / "test.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            if int(row["initial_state_id"]) == pid:
                return np.asarray(
                    [int(value) for value in row["initial_state"].split(",")],
                    dtype=np.uint8,
                )
    raise KeyError(f"pid {pid} is absent from test.csv")


def _replay(state, path, moves):
    current = np.asarray(state)
    for move in path:
        current = current[moves[int(move)]]
    return current


def _load_endgame(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as archive:
        return {
            "hashes": np.asarray(archive["hashes"]),
            "depths": np.asarray(archive["depths"]),
            "ztab": np.asarray(archive["ztab"]),
            "max_depth": int(archive["max_depth"]),
        }


def _endgame_lookup(states, endgame):
    values = np.atleast_2d(np.asarray(states, dtype=np.int64))
    hashes = np.zeros(values.shape[0], dtype=np.int64)
    for position in range(values.shape[1]):
        hashes ^= endgame["ztab"][position][values[:, position]]
    indices = np.clip(
        np.searchsorted(endgame["hashes"], hashes),
        0,
        endgame["hashes"].size - 1,
    )
    result = np.full(values.shape[0], -1, dtype=np.int64)
    hit = endgame["hashes"][indices] == hashes
    result[hit] = endgame["depths"][indices[hit]]
    return result


def _endgame_descend(state, endgame, moves, solved):
    depth = int(_endgame_lookup(state[None, :], endgame)[0])
    if depth < 0:
        return None
    path = []
    current = np.asarray(state)
    while depth > 0:
        children = current[moves]
        hit = np.nonzero(
            _endgame_lookup(children, endgame) == depth - 1
        )[0]
        if not hit.size:
            return None
        move = int(hit[0])
        path.append(move)
        current = children[move]
        depth -= 1
    return path if np.array_equal(current, solved) else None


def run_real_solve_gate(
    *,
    params,
    runtime,
    puzzle,
    dataset: Path,
    competition: Path,
    mesh,
):
    state = _read_competition_state(competition, REAL_PID)
    if state.shape != puzzle.solved.shape:
        raise ValueError("competition state shape does not match puzzle")
    endgame_path = dataset / "bfs_endgame.npz"
    if not endgame_path.is_file():
        raise FileNotFoundError("bfs_endgame.npz is required by the notebook gate")
    endgame = _load_endgame(endgame_path)
    eg_hashes = jnp.asarray(endgame["hashes"])
    eg_ztab = jnp.asarray(endgame["ztab"])
    all_moves = jnp.asarray(puzzle.moves, dtype=jnp.int32)
    v0 = jnp.asarray(puzzle.solved, dtype=jnp.uint8)
    hash_vec = jnp.asarray(
        np.random.default_rng(0).integers(
            0, int(1e15), size=puzzle.solved.size, dtype=np.int64
        )
    )
    owner_hash_vec = jnp.asarray(
        np.random.default_rng(12345).integers(
            0,
            np.iinfo(np.uint32).max,
            size=puzzle.solved.size,
            dtype=np.uint32,
        )
    )
    exact_config = ArtgorExactConfig(
        prefix_bm=4096,
        head_bm=256,
        head_bk=1024,
        head_bn=128,
        dense_rounding="late",
        inference_chunk=INFERENCE_LOCAL_BATCH,
        parent_chunk=131_072,
    )
    tree_path = Path("/tmp/artgor_exact_validation_pid1034.u32")
    started = time.perf_counter()
    solver = beam_solve_v_only_spmd_packed_exact(
        state.tolist(),
        params,
        all_moves,
        v0,
        hash_vec,
        mesh,
        B_local=REAL_B_LOCAL,
        K_per_peer=(2 * REAL_B_LOCAL) // TARGET_DEVICE_COUNT,
        n_gen=len(puzzle.move_names),
        state_size=puzzle.solved.size,
        num_steps=300,
        dtype=jnp.bfloat16,
        internal_bs=INFERENCE_LOCAL_BATCH,
        tree_path=str(tree_path),
        parent_chunk=131_072,
        pack_v_score=True,
        progress_every=10,
        owner_hash_vec=owner_hash_vec,
        history_depth=4,
        history_exact=True,
        eg_hashes=eg_hashes,
        eg_ztab=eg_ztab,
        inv_move_tbl=None,
        q_mode=True,
        qv_consistency=0.0,
        exact_config=exact_config,
        exact_runtime=runtime,
    )
    gate_wall_s = time.perf_counter() - started
    record = {
        "pid": REAL_PID,
        "sym": 0,
        "inverted": False,
        "b_global": TARGET_DEVICE_COUNT * REAL_B_LOCAL,
        "found": bool(solver["found"]),
        "verify_ok": False,
        "solver_wall_s": float(solver["wall_s"]),
        "gate_wall_s": gate_wall_s,
        "found_step": int(solver["found_step"]),
        "compile_s": float(solver["compile_s"]),
        "lower_s": float(solver["lower_s"]),
        "timing_breakdown": solver.get("timing_breakdown", {}),
        "endgame_max_depth": endgame["max_depth"],
    }
    if not solver["found"]:
        return record

    beam_path = [int(move) for move in solver["path_idx"]]
    reached = _replay(state, beam_path, puzzle.moves)
    tail = []
    false_positive = False
    if not np.array_equal(reached, puzzle.solved):
        tail = _endgame_descend(
            reached, endgame, puzzle.moves, puzzle.solved
        )
        if tail is None:
            false_positive = True
            tail = []
    full_path = beam_path + tail
    verify_ok = bool(
        not false_positive
        and np.array_equal(
            _replay(state, full_path, puzzle.moves), puzzle.solved
        )
    )
    record.update(
        {
            "verify_ok": verify_ok,
            "path_len": len(full_path),
            "beam_path_len": len(beam_path),
            "endgame_tail_len": len(tail),
            "endgame_false_positive": false_positive,
            "path_idx": full_path,
            "path": ".".join(puzzle.move_names[index] for index in full_path),
            "path_sha256": hashlib.sha256(
                np.asarray(full_path, dtype=np.uint8).tobytes()
            ).hexdigest(),
        }
    )
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--competition", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/artgor_exact_notebook_validation"),
    )
    return parser.parse_args(argv)


def run_validation(
    *, dataset: Path, competition: Path, output: Path
) -> dict:
    result_path = output / RESULT_NAME
    report = {
        "status": "running",
        "context": {},
        "inference": {},
        "one_depth": {},
        "short_solve": {},
        "real_solve": {},
        "decision": {},
    }
    try:
        devices = jax.devices()
        if len(devices) < TARGET_DEVICE_COUNT or any(
            device.platform != "tpu"
            for device in devices[:TARGET_DEVICE_COUNT]
        ):
            raise RuntimeError(
                f"requires eight TPU devices, found: {devices}"
            )
        if not jax.config.jax_enable_x64:
            raise RuntimeError("JAX_ENABLE_X64 must be true")
        inventory = runtime_inventory()
        inventory["active_device_count"] = TARGET_DEVICE_COUNT
        source_commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True
        ).strip()
        checkpoint_path = dataset / "q555_2k_BEST.pt"
        model_source = dataset / "jax_model.py"
        beam_source = dataset / "jax_beam_spmd_v_only.py"
        puzzle_path = dataset / "puzzle_info.json"
        source_hashes = {
            "jax_model.py": sha256_file(model_source),
            "jax_beam_spmd_v_only.py": sha256_file(beam_source),
        }
        expected_hashes = {
            "jax_model.py": (
                "6d00da89ce45cf84167db20780e30f676cde3ae756d376c8e05a7e0dcf98e46e"
            ),
            "jax_beam_spmd_v_only.py": (
                "aaa0dbe16fd82a0f2bc08f1216f4e87c8a2a63c855f5d7012b6c18d8b57d42cb"
            ),
        }
        if source_hashes != expected_hashes:
            raise RuntimeError(
                f"Artgor runtime source hashes changed: {source_hashes}"
            )
        report["context"] = {
            "source_commit": source_commit,
            "artgor_script_version": 344319112,
            "runtime": inventory,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "model_source_sha256": source_hashes["jax_model.py"],
            "beam_source_sha256": source_hashes[
                "jax_beam_spmd_v_only.py"
            ],
            "puzzle_sha256": sha256_file(puzzle_path),
            "competition_test_sha256": sha256_file(
                competition / "test.csv"
            ),
            "seeds": {
                "legal": 42,
                "stress": 43,
                "hash": 0,
                "owner_hash": 12345,
            },
        }
        checkpoint(result_path, report)
        print(json.dumps(report["context"], indent=2), flush=True)

        sys.path.insert(0, str(dataset))
        from jax_model import apply as original_apply, load_params_from_pt
        import jax_beam_spmd_v_only as original_beam

        with jax.default_device(jax.local_devices()[0]):
            params = load_params_from_pt(checkpoint_path)
        puzzle = load_puzzle(puzzle_path, state_len=150, move_count=30)
        exact_mesh = Mesh(
            np.asarray(devices[:TARGET_DEVICE_COUNT]), ("core",)
        )
        exact_config = ArtgorExactConfig(
            prefix_bm=4096,
            head_bm=256,
            head_bk=1024,
            head_bn=128,
            dense_rounding="late",
            inference_chunk=INFERENCE_LOCAL_BATCH,
            parent_chunk=131_072,
        )
        runtime = prepare_artgor_exact_beam_runtime(
            params,
            mesh=exact_mesh,
            exact_config=exact_config,
            state_storage_len=150,
        )

        inference, corpora, weights_d = run_inference_gate(
            original_apply=original_apply,
            params=params,
            runtime=runtime,
            puzzle=puzzle,
            mesh=exact_mesh,
        )
        report["inference"] = inference
        report["context"]["input_sha256"] = {
            name: _array_sha256(value) for name, value in corpora.items()
        }
        checkpoint(result_path, report)
        inference_ok = (
            inference["legal_scrambles"]["exact"]
            and inference["categorical_stress"]["exact"]
            and inference["speedup"] >= MIN_INFERENCE_SPEEDUP
        )
        if not inference_ok:
            report["status"] = "rejected"
            report["decision"] = decide_publication(report)
            checkpoint(result_path, report)
            return report

        one_depth, short_solve = run_depth_and_short_gate(
            original_beam=original_beam,
            params=params,
            runtime=runtime,
            weights_d=weights_d,
            puzzle=puzzle,
            inference_corpora=corpora,
            exact_mesh=exact_mesh,
        )
        report["one_depth"] = one_depth
        report["short_solve"] = short_solve
        checkpoint(result_path, report)
        parity_ok = (
            one_depth["all_tensor_hashes_equal"]
            and short_solve["frontiers_equal"]
            and short_solve["backpointers_equal"]
            and short_solve["all_outputs_equal"]
        )
        if not parity_ok:
            report["status"] = "rejected"
            report["decision"] = decide_publication(report)
            checkpoint(result_path, report)
            return report

        report["real_solve"] = run_real_solve_gate(
            params=params,
            runtime=runtime,
            puzzle=puzzle,
            dataset=dataset,
            competition=competition,
            mesh=exact_mesh,
        )
        report["decision"] = decide_publication(report)
        report["status"] = (
            "complete" if report["decision"]["publishable"] else "rejected"
        )
        checkpoint(result_path, report)
        return report
    except Exception as error:
        report.update(
            status="error",
            fatal_error_type=type(error).__name__,
            fatal_error=str(error),
            fatal_traceback=traceback.format_exc(),
        )
        report["decision"] = decide_publication(report)
        checkpoint(result_path, report)
        raise


def main(argv=None):
    args = parse_args(argv)
    dataset = _dataset_path(args.dataset)
    competition = _competition_path(args.competition)
    report = run_validation(
        dataset=dataset,
        competition=competition,
        output=args.output,
    )
    print("DECISION", json.dumps(report["decision"]), flush=True)
    print("RESULT_PATH", args.output / RESULT_NAME, flush=True)


if __name__ == "__main__":
    main()
