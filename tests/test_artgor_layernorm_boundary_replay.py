import numpy as np

from benchmarks.artgor_layernorm_boundary_replay import (
    RESULT_NAME,
    compare_replays,
    replay_checkpoint_names,
)


def test_boundary_replay_contract_tracks_the_five_real_pallas_calls():
    assert RESULT_NAME == "artgor_layernorm_boundary_replay.json"
    assert replay_checkpoint_names(relu=True) == (
        "mean", "centered", "variance", "invstd", "affine_relu",
    )


def test_boundary_replay_compares_pallas_to_both_jax_controls():
    pallas = {
        "mean": np.asarray([[1]], np.float32),
        "affine_relu": np.asarray([[2, 3]], np.float32),
    }
    materialized = {
        "mean": np.asarray([[1]], np.float32),
        "affine_relu": np.asarray([[2, 3]], np.float32),
    }
    monolithic = np.asarray([[2, 4]], np.float32)
    result = compare_replays(
        pallas=pallas, materialized=materialized, monolithic=monolithic,
    )
    assert result["pallas_vs_materialized"]["first_mismatch"] is None
    assert result["pallas_vs_monolithic_final"]["mismatch_count"] == 1
    assert result["pallas_vs_monolithic_final"]["rmse"] == np.sqrt(0.5)
    assert len(result["pallas_vs_monolithic_final"]["candidate_sha256"]) == 64
    assert result["pallas_vs_monolithic_final"]["witnesses"][0]["flat_index"] == 1
