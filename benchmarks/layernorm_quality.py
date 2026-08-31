"""NumPy-only input and quality diagnostics for the minimizing cube555 Q head.

The stable (score, flat_id) ranking is a diagnostic. It does not reproduce the
distributed consumer's owner quotas, score packing, hash dedup or history. No
quality-acceptance threshold is implied by finite/eligible status.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Puzzle:
    move_names: tuple[str, ...]
    solved: np.ndarray
    moves: np.ndarray
    inverse: np.ndarray
    sha256: str


@dataclass(frozen=True)
class LegalScrambles:
    states: np.ndarray
    lengths: np.ndarray
    last_moves: np.ndarray
    sequences: np.ndarray


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _integers(values, name):
    result = np.asarray(values)
    if not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    return result


def load_puzzle(path, *, state_len=150, move_count=30) -> Puzzle:
    """Read real puzzle_info.json; retain generator order as Q-column order.

    This loader intentionally requires an identity-labelled picture puzzle.
    Reachability is established by applying its generators, not by merely
    checking that a state is a permutation. Nothing is downloaded or imported.
    """
    state_len = _positive_int(state_len, "state_len")
    move_count = _positive_int(move_count, "move_count")
    if state_len > 256:
        raise ValueError("state_len must fit uint8 labels (at most 256)")
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict) or not isinstance(document.get("generators"), dict):
        raise ValueError("puzzle must have a generators object")
    generators = document["generators"]
    names = tuple(generators)
    if len(names) != move_count or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("generator names/count do not match the requested puzzle")
    solved = _integers(document.get("central_state"), "central_state")
    identity = np.arange(state_len)
    if solved.shape != (state_len,) or not np.array_equal(solved, identity):
        raise ValueError("central_state must be the requested identity permutation")
    moves = _integers([generators[name] for name in names], "generators")
    if moves.shape != (move_count, state_len):
        raise ValueError("generator shape does not match the requested puzzle")
    if not np.all(np.sort(moves, axis=1) == identity[None, :]):
        raise ValueError("each generator must permute every state index exactly once")
    name_to_index = {name: index for index, name in enumerate(names)}
    inverse_names = [name[1:] if name.startswith("-") else "-" + name for name in names]
    if any(name not in name_to_index for name in inverse_names):
        raise ValueError("every generator must have its named inverse")
    inverse = np.array([name_to_index[name] for name in inverse_names], dtype=np.int32)
    moves = moves.astype(np.int32)
    if not np.array_equal(inverse[inverse], np.arange(move_count)):
        raise ValueError("inverse map must be an involution")
    if not np.all(np.take_along_axis(moves, moves[inverse], axis=1) == identity):
        raise ValueError("named inverse permutations do not compose to identity")
    return Puzzle(names, solved.astype(np.uint8), moves, inverse, hashlib.sha256(raw).hexdigest())


def make_legal_scrambles(
    puzzle: Puzzle, *, batch, seed=42, depth_choices=(0, 1, 2, 4, 8, 16, 32, 64, 128)
) -> LegalScrambles:
    """Generate reachable random walks stratified cyclically by requested depth.

    Moves are independent uniform draws; immediate inverses are allowed. Depth
    is walk length, not optimal distance. Reusing seed and depth_choices keeps
    corpus prefixes identical across batch sizes on the same NumPy runtime.
    Sequences use -1 padding; solved/zero-depth rows have last_move=-1.
    """
    batch = _positive_int(batch, "batch")
    depths = _integers(depth_choices, "depth_choices")
    if depths.ndim != 1 or not depths.size or np.any(depths < 0):
        raise ValueError("depth_choices must be a nonempty sequence of nonnegative integers")
    if np.any(depths > np.iinfo(np.int32).max):
        raise ValueError("depth_choices exceed int32 sequence lengths")
    lengths = np.resize(depths.astype(np.int32), batch)
    max_depth = int(depths.max())
    states = np.broadcast_to(puzzle.solved, (batch, puzzle.solved.size)).copy()
    sequences = np.random.default_rng(seed).integers(
        0, len(puzzle.move_names), size=(batch, max_depth), dtype=np.int32
    )
    sequences[np.arange(max_depth)[None, :] >= lengths[:, None]] = -1
    last_moves = np.full(batch, -1, dtype=np.int32)
    for step in range(max_depth):
        active = lengths > step
        chosen = sequences[active, step]
        states[active] = np.take_along_axis(states[active], puzzle.moves[chosen], axis=1)
        last_moves[active] = chosen
    return LegalScrambles(states, lengths, last_moves, sequences)


def inverse_valid_mask(last_moves, inverse) -> np.ndarray:
    """True means an allowed action; -1 incoming moves ban nothing."""
    last = _integers(last_moves, "last_moves")
    inv = _integers(inverse, "inverse")
    if inv.ndim != 1 or not inv.size:
        raise ValueError("inverse must be a nonempty vector")
    count = inv.size
    if not np.array_equal(np.sort(inv), np.arange(count)):
        raise ValueError("inverse must be a permutation of move indices")
    if not np.array_equal(inv[inv], np.arange(count)):
        raise ValueError("inverse must be an involution")
    if last.ndim != 1 or np.any(last < -1) or np.any(last >= count):
        raise ValueError("last_moves must be a vector with indices in [-1, move_count)")
    valid = np.ones((last.size, count), dtype=bool)
    active = np.flatnonzero(last >= 0)
    valid[active, inv[last[active]]] = False
    return valid


def _tensor_pair(ref, candidate):
    if np.iscomplexobj(ref) or np.iscomplexobj(candidate):
        raise ValueError("metrics require real tensors")
    # Casting here also accepts a NumPy array with the optional bfloat16 dtype;
    # this module itself has no JAX, Torch or ml_dtypes dependency.
    expected = np.asarray(ref, dtype=np.float64)
    actual = np.asarray(candidate, dtype=np.float64)
    if expected.shape != actual.shape or not expected.size:
        raise ValueError("tensors must have equal, nonempty shapes")
    return expected, actual


def _json_float(value):
    return float(value) if np.isfinite(value) else None


def tensor_metrics(ref, candidate) -> dict:
    """Pairwise tensor errors; undefined/nonfinite statistics become JSON null.

    Nonfinite inputs invalidate the whole comparison, including masked scores
    when called by minimizing_q_metrics. Both zero vectors have cosine=1;
    exactly one zero vector has cosine=0. These are diagnostics, not gates.
    """
    expected, actual = _tensor_pair(ref, candidate)
    ref_bad = int(np.count_nonzero(~np.isfinite(expected)))
    candidate_bad = int(np.count_nonzero(~np.isfinite(actual)))
    result = {
        "finite": ref_bad == 0 and candidate_bad == 0,
        "reference_finite": ref_bad == 0,
        "candidate_finite": candidate_bad == 0,
        "reference_nonfinite_count": ref_bad,
        "candidate_nonfinite_count": candidate_bad,
        "max_abs": None, "mean_abs": None, "rmse": None,
        "cosine": None, "exact_fraction": None,
    }
    if not result["finite"]:
        return result
    with np.errstate(over="ignore", invalid="ignore"):
        absolute = np.abs(actual - expected)
        max_abs = np.max(absolute)
        if max_abs == 0:
            mean_abs = rmse = 0.0
        else:
            scaled_error = absolute / max_abs
            mean_abs = max_abs * np.mean(scaled_error)
            rmse = max_abs * np.sqrt(np.mean(scaled_error ** 2))
        ref_scale, actual_scale = np.max(np.abs(expected)), np.max(np.abs(actual))
        if ref_scale == 0 or actual_scale == 0:
            cosine = float(ref_scale == actual_scale)
        else:
            left = (expected / ref_scale).reshape(-1)
            right = (actual / actual_scale).reshape(-1)
            cosine = np.clip(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)), -1, 1)
    result.update(
        max_abs=_json_float(max_abs), mean_abs=_json_float(mean_abs),
        rmse=_json_float(rmse), cosine=_json_float(cosine),
        exact_fraction=float(np.mean(actual == expected)),
    )
    return result


def _ranking(values, valid, counts, k):
    ids = np.flatnonzero(valid.reshape(-1))
    flat = values.reshape(-1)
    # lexsort's last key is primary. Flat IDs break exact score ties explicitly.
    order = ids[np.lexsort((ids, flat[ids]))]
    selected = order[:k]
    row_sorted = np.sort(np.where(valid, values, np.inf), axis=1)
    gaps = []
    with np.errstate(over="ignore", invalid="ignore"):
        for row, count in enumerate(counts):
            gaps.append(_json_float(row_sorted[row, 1] - row_sorted[row, 0]) if count >= 2 else None)
        boundary_gap = _json_float(flat[order[k]] - flat[order[k - 1]]) if k < order.size else None
    kth = flat[order[k - 1]]
    summary = {
        "best_second_gaps": gaps,
        "best_tie_counts": np.sum(valid & (values == row_sorted[:, :1]), axis=1).tolist(),
        "kth_score": float(kth),
        "kplus1_score": float(flat[order[k]]) if k < order.size else None,
        "k_boundary_gap": boundary_gap,
        "k_boundary_tie_count": int(np.count_nonzero(flat[ids] == kth)),
    }
    return selected, summary


def minimizing_q_metrics(ref, candidate, *, valid_mask=None, k) -> dict:
    """Compare minimizing row decisions and stable flattened global top-K.

    IDs are parent-major/move-minor (parent * move_count + move). Both row
    argmin and auxiliary argmax honor the same mask; all-masked rows are omitted
    from agreement denominators and counted explicitly. k must fit valid slots.
    eligible means only that both raw tensors are finite: it is NOT a numerical
    tolerance, solve-quality acceptance decision or distributed-beam result.
    """
    expected, actual = _tensor_pair(ref, candidate)
    if expected.ndim != 2:
        raise ValueError("Q outputs must have shape [parents, moves]")
    if valid_mask is None:
        valid = np.ones(expected.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask)
        if valid.shape != expected.shape or valid.dtype != np.bool_:
            raise ValueError("valid_mask must be a boolean array matching Q output shape")
    k = _positive_int(k, "k")
    counts = np.sum(valid, axis=1)
    total = int(np.sum(counts))
    if k > total:
        raise ValueError("k must not exceed the number of valid candidates")
    result = tensor_metrics(expected, actual)
    result.update({
        "eligible": result["finite"],
        "score_direction": "minimize",
        "selection_scope": "global_flat_diagnostic_not_distributed_beam",
        "tie_policy": "score_then_flat_id",
        "k": k,
        "valid_candidates": total,
        "valid_counts": counts.tolist(),
        "all_masked_rows": int(np.count_nonzero(counts == 0)),
        "argmin_agreement": None, "argmax_agreement": None,
        "topk_overlap": None, "topk_order_agreement": None,
        "topk_order_exact": None, "selected_invalid_count": None,
        "reference_topk_flat_ids": None, "candidate_topk_flat_ids": None,
        "reference_ranking": None, "candidate_ranking": None,
    })
    if not result["eligible"]:
        return result
    real_rows = counts > 0
    ref_min = np.argmin(np.where(valid, expected, np.inf), axis=1)
    cand_min = np.argmin(np.where(valid, actual, np.inf), axis=1)
    ref_max = np.argmax(np.where(valid, expected, -np.inf), axis=1)
    cand_max = np.argmax(np.where(valid, actual, -np.inf), axis=1)
    ref_ids, ref_ranking = _ranking(expected, valid, counts, k)
    cand_ids, cand_ranking = _ranking(actual, valid, counts, k)
    result.update(
        argmin_agreement=float(np.mean(ref_min[real_rows] == cand_min[real_rows])),
        argmax_agreement=float(np.mean(ref_max[real_rows] == cand_max[real_rows])),
        topk_overlap=float(np.intersect1d(ref_ids, cand_ids).size / k),
        topk_order_agreement=float(np.mean(ref_ids == cand_ids)),
        topk_order_exact=bool(np.array_equal(ref_ids, cand_ids)),
        selected_invalid_count=int(np.count_nonzero(~valid.reshape(-1)[cand_ids])),
        reference_topk_flat_ids=ref_ids.tolist(),
        candidate_topk_flat_ids=cand_ids.tolist(),
        reference_ranking=ref_ranking, candidate_ranking=cand_ranking,
    )
    return result
