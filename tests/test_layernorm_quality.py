from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json

import numpy as np
import pytest


def _quality():
    # Lazy import makes the initial RED a missing-feature assertion, not a
    # collection error. Every test below calls the actual production helper.
    assert importlib.util.find_spec("benchmarks.layernorm_quality") is not None
    return importlib.import_module("benchmarks.layernorm_quality")


def _write_puzzle(tmp_path, change=None):
    # Two independently specified cycles on four labels. Deliberately unsorted
    # names catch reordering of Q-head columns by the dataset loader.
    document = {
        "central_state": [0, 1, 2, 3],
        "generators": {
            "-a": [2, 0, 1, 3],
            "b": [0, 3, 1, 2],
            "a": [1, 2, 0, 3],
            "-b": [0, 2, 3, 1],
        },
    }
    if change is not None:
        change(document)
    path = tmp_path / "puzzle_info.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_load_puzzle_preserves_q_move_order_and_validates_inverse(tmp_path):
    quality = _quality()
    path = _write_puzzle(tmp_path)
    puzzle = quality.load_puzzle(path, state_len=4, move_count=4)
    assert puzzle.move_names == ("-a", "b", "a", "-b")
    np.testing.assert_array_equal(puzzle.solved, [0, 1, 2, 3])
    np.testing.assert_array_equal(puzzle.moves[1], [0, 3, 1, 2])
    np.testing.assert_array_equal(puzzle.inverse, [2, 3, 0, 1])
    assert puzzle.solved.dtype == np.uint8
    assert puzzle.moves.dtype == np.int32
    assert puzzle.inverse.dtype == np.int32
    assert puzzle.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(central_state=[0, 1, 1, 3]),
        lambda d: d.update(central_state=[1, 0, 2, 3]),
        lambda d: d.update(central_state=[0, 1, 2]),
        lambda d: d["generators"].update(a=[0, 1, 1, 3]),
        lambda d: d["generators"].update(a=[0, 1, 2, 4]),
        lambda d: d["generators"].update(a=[0.0, 1.0, 2.0, 3.0]),
        lambda d: d["generators"].update(a=[0, 1, 2]),
        lambda d: d["generators"].update({"-a": [1, 2, 0, 3]}),
        lambda d: d["generators"].update(c=d["generators"].pop("a")),
    ],
)
def test_load_puzzle_rejects_invalid_group_data(tmp_path, change):
    quality = _quality()
    with pytest.raises(ValueError):
        quality.load_puzzle(_write_puzzle(tmp_path, change), state_len=4, move_count=4)


def test_load_puzzle_requires_real_file_and_requested_shape(tmp_path):
    quality = _quality()
    with pytest.raises(FileNotFoundError):
        quality.load_puzzle(tmp_path / "missing.json")
    with pytest.raises(ValueError):
        quality.load_puzzle(_write_puzzle(tmp_path))


def test_legal_scrambles_replay_and_reverse_with_independent_moves(tmp_path):
    quality = _quality()
    puzzle = quality.load_puzzle(_write_puzzle(tmp_path), state_len=4, move_count=4)
    corpus = quality.make_legal_scrambles(
        puzzle, batch=32, seed=91, depth_choices=(0, 1, 2, 4)
    )
    assert corpus.states.shape == (32, 4)
    assert corpus.states.dtype == np.uint8
    np.testing.assert_array_equal(corpus.lengths, [0, 1, 2, 4] * 8)
    moves = ([2, 0, 1, 3], [0, 3, 1, 2], [1, 2, 0, 3], [0, 2, 3, 1])
    inverse = (2, 3, 0, 1)
    for row, length in enumerate(corpus.lengths):
        state = np.array([0, 1, 2, 3], dtype=np.uint8)
        path = corpus.sequences[row, :length]
        assert np.all(corpus.sequences[row, length:] == -1)
        for move in path:
            state = state[moves[move]]
        np.testing.assert_array_equal(corpus.states[row], state)
        assert corpus.last_moves[row] == (path[-1] if length else -1)
        for move in reversed(path):
            state = state[moves[inverse[move]]]
        np.testing.assert_array_equal(state, [0, 1, 2, 3])


def test_legal_scrambles_are_deterministic_and_batch_prefix_stable(tmp_path):
    quality = _quality()
    puzzle = quality.load_puzzle(_write_puzzle(tmp_path), state_len=4, move_count=4)
    small = quality.make_legal_scrambles(puzzle, batch=12, seed=12, depth_choices=(1, 7))
    large = quality.make_legal_scrambles(puzzle, batch=24, seed=12, depth_choices=(1, 7))
    other = quality.make_legal_scrambles(puzzle, batch=12, seed=13, depth_choices=(1, 7))
    np.testing.assert_array_equal(small.states, large.states[:12])
    np.testing.assert_array_equal(small.sequences, large.sequences[:12])
    assert not np.array_equal(small.sequences, other.sequences)


def test_legal_scrambles_keep_high_uint8_labels_and_zero_depth(tmp_path):
    quality = _quality()
    path = tmp_path / "large.json"
    path.write_text(json.dumps({
        "central_state": list(range(150)),
        "generators": {
            "a": list(range(1, 150)) + [0],
            "-a": [149] + list(range(149)),
        },
    }), encoding="utf-8")
    puzzle = quality.load_puzzle(path, move_count=2)
    corpus = quality.make_legal_scrambles(puzzle, batch=4, depth_choices=(0,))
    assert corpus.sequences.shape == (4, 0)
    assert np.all(corpus.last_moves == -1)
    assert int(corpus.states[0, 149]) == 149
    moved = quality.make_legal_scrambles(puzzle, batch=4, depth_choices=(1,))
    np.testing.assert_array_equal(np.sort(moved.states, axis=1), corpus.states)


@pytest.mark.parametrize("kwargs", [{"batch": 0}, {"batch": -1}, {"batch": 1.5},
                                      {"batch": 2, "depth_choices": ()},
                                      {"batch": 2, "depth_choices": (-1,)},
                                      {"batch": 2, "depth_choices": (1.5,)}])
def test_legal_scrambles_reject_invalid_sizes(tmp_path, kwargs):
    quality = _quality()
    puzzle = quality.load_puzzle(_write_puzzle(tmp_path), state_len=4, move_count=4)
    with pytest.raises(ValueError):
        quality.make_legal_scrambles(puzzle, **kwargs)


def test_inverse_valid_mask_uses_incoming_moves_and_seed_sentinel():
    quality = _quality()
    actual = quality.inverse_valid_mask([-1, 0, 1, 2, 3], [2, 3, 0, 1])
    np.testing.assert_array_equal(actual, [
        [True, True, True, True], [True, True, False, True],
        [True, True, True, False], [False, True, True, True],
        [True, False, True, True],
    ])
    assert actual.dtype == np.bool_


@pytest.mark.parametrize("last,inverse", [([-2], [1, 0]), ([2], [1, 0]),
                                           ([0.5], [1, 0]), ([0], [0, 0]),
                                           ([0], [1, 2, 0])])
def test_inverse_valid_mask_rejects_invalid_indices(last, inverse):
    quality = _quality()
    with pytest.raises(ValueError):
        quality.inverse_valid_mask(last, inverse)


def test_tensor_metrics_measure_pairwise_values_without_threshold():
    result = _quality().tensor_metrics([[1, 2]], [[2, 2]])
    assert result["finite"] is True
    assert result["max_abs"] == 1.0
    assert result["mean_abs"] == 0.5
    assert result["rmse"] == pytest.approx(0.5 ** 0.5)
    assert result["cosine"] == pytest.approx(6 / (40 ** 0.5))
    assert result["exact_fraction"] == 0.5
    assert "valid" not in result


@pytest.mark.parametrize("ref,candidate,want", [([0, 0], [0, 0], 1.0),
                                              ([0, 0], [1, 0], 0.0)])
def test_tensor_metrics_define_zero_norm_cosine(ref, candidate, want):
    assert _quality().tensor_metrics(ref, candidate)["cosine"] == want


@pytest.mark.parametrize("ref,candidate", [([1], [1, 2]), ([], []), ([1j], [1j])])
def test_tensor_metrics_reject_incompatible_empty_or_complex_tensors(ref, candidate):
    with pytest.raises(ValueError):
        _quality().tensor_metrics(ref, candidate)


def test_minimizing_global_topk_catches_rowwise_agreement_blindspot():
    result = _quality().minimizing_q_metrics(
        [[1, 2], [3, 4]], [[1, 100], [1.5, 4]], k=2
    )
    assert result["eligible"] is True
    assert result["argmin_agreement"] == result["argmax_agreement"] == 1.0
    assert result["reference_topk_flat_ids"] == [0, 1]
    assert result["candidate_topk_flat_ids"] == [0, 2]
    assert result["topk_overlap"] == 0.5
    assert result["topk_order_agreement"] == 0.5
    assert result["topk_order_exact"] is False
    assert result["reference_ranking"]["k_boundary_gap"] == 1.0
    assert result["candidate_ranking"]["k_boundary_gap"] == 2.5


def test_minimizing_metrics_respect_mask_and_report_all_masked_rows():
    result = _quality().minimizing_q_metrics(
        [[0, -100, 2], [7, 8, 9]], [[0, -1000, 2], [7, 8, 9]],
        valid_mask=[[True, False, True], [False, False, False]], k=2,
    )
    assert result["reference_topk_flat_ids"] == [0, 2]
    assert result["selected_invalid_count"] == 0
    assert result["valid_counts"] == [2, 0]
    assert result["valid_candidates"] == 2
    assert result["all_masked_rows"] == 1
    assert result["argmin_agreement"] == result["argmax_agreement"] == 1.0
    assert result["reference_ranking"]["best_second_gaps"] == [2.0, None]
    assert result["reference_ranking"]["k_boundary_gap"] is None


def test_minimizing_metrics_handle_single_legal_move_and_ties():
    quality = _quality()
    sole = quality.minimizing_q_metrics([[9, 1, 2]], [[-9, 7, -2]],
                                        valid_mask=[[False, True, False]], k=1)
    assert sole["candidate_topk_flat_ids"] == [1]
    assert sole["reference_ranking"]["best_second_gaps"] == [None]
    tied = quality.minimizing_q_metrics([[0, 0, 1, 2]], [[0, 0, 1, 2]], k=1)
    assert tied["reference_topk_flat_ids"] == [0]
    assert tied["topk_order_exact"] is True
    assert tied["reference_ranking"]["best_tie_counts"] == [2]
    assert tied["reference_ranking"]["k_boundary_gap"] == 0.0
    assert tied["reference_ranking"]["k_boundary_tie_count"] == 2


def test_minimizing_metrics_distinguish_same_set_from_different_order():
    result = _quality().minimizing_q_metrics([[0, 1, 10]], [[1, 0, 10]], k=2)
    assert result["topk_overlap"] == 1.0
    assert result["topk_order_agreement"] == 0.0
    assert result["topk_order_exact"] is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("bad_reference", [False, True])
def test_nonfinite_scores_are_ineligible_and_json_safe_even_when_masked(bad, bad_reference):
    quality = _quality()
    ref, candidate = [[1, 2]], [[1, bad]]
    if bad_reference:
        ref, candidate = candidate, ref
    result = quality.minimizing_q_metrics(ref, candidate,
                                          valid_mask=[[True, False]], k=1)
    assert result["eligible"] is False
    assert result["finite"] is False
    assert result["max_abs"] is None
    assert result["argmin_agreement"] is None
    assert result["topk_overlap"] is None
    assert result["candidate_topk_flat_ids"] is None
    assert result["reference_nonfinite_count"] == int(bad_reference)
    assert result["candidate_nonfinite_count"] == int(not bad_reference)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"k": 3}, {"k": 1.5},
                                      {"k": 1, "valid_mask": [[False, False]]},
                                      {"k": 1, "valid_mask": [[True]]},
                                      {"k": 1, "valid_mask": [[1, 0]]}])
def test_minimizing_metrics_reject_invalid_k_and_masks(kwargs):
    with pytest.raises(ValueError):
        _quality().minimizing_q_metrics([[1, 2]], [[1, 2]], **kwargs)


def test_minimizing_metrics_require_a_matrix():
    with pytest.raises(ValueError):
        _quality().minimizing_q_metrics([1, 2], [1, 2], k=1)
