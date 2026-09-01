"""Contracts for the exact eight-TPU inference frontier bundle."""

import json

import numpy as np

from benchmarks import stream1_exact_inference_frontier as frontier
from test_layernorm_followup import model_fixture
from tpu_beam_search.artgor_reference import artgor_reference_apply
from tpu_beam_search.stream1_layernorm_exact import (
    prepare_exact_layernorm_inference_weights,
    stream1_layernorm_exact_head,
    stream1_layernorm_exact_prefix,
)


def _row(identifier, corpus, median_ms, *, exact=True, status="ok"):
    return {
        "id": identifier,
        "corpus": corpus,
        "status": status,
        "exact_oracle_on_sample": exact,
        "timing_comparable": True,
        "timing": {"median_ms": median_ms},
    }


def test_frontier_matrix_keeps_the_accepted_winner_and_bounds_the_sweep():
    assert frontier.PREFIX_BMS == (2048, 4096, 8192, 16384)
    assert frontier.IDENTITY_BMS == (128, 512, 2048)

    heads = frontier.head_configs()
    assert heads[0] == {
        "id": frontier.JAX_HEAD_ID,
        "backend": "jax",
        "control": True,
    }
    candidates = heads[1:]
    assert len(candidates) == 40
    assert len({config["id"] for config in heads}) == len(heads)
    assert {config["bm"] for config in candidates} == {128, 256, 512, 1024, 2048}
    assert {config["bk"] for config in candidates} == {128, 256, 512, 1024}
    assert {config["bn"] for config in candidates} == {128}
    assert {config["dense_rounding"] for config in candidates} == {
        "late", "bf16_before_bias",
    }

    full = frontier.full_candidate_configs(
        exact_prefix_bms=(2048, 8192),
        promoted_head_ids=(candidates[0]["id"], candidates[1]["id"]),
    )
    assert full[0]["id"] == frontier.ORIGINAL_ID
    assert full[1]["id"] == frontier.TYPED_ID
    assert full[2]["id"] == frontier.ACCEPTED_ID
    assert full[2]["role"] == "accepted_control"
    assert len({config["id"] for config in full}) == len(full)
    assert any(config["backend"] == "materialized_identity" for config in full)
    assert any(
        config["backend"] == "split" and config["head_backend"] == "pallas"
        for config in full
    )


def test_head_promotion_requires_both_corpora_exact_and_ranks_by_speed():
    rows = []
    for corpus in ("legal", "stress"):
        rows.extend([
            _row(frontier.JAX_HEAD_ID, corpus, 2.0),
            _row("fast", corpus, 1.0),
            _row("second", corpus, 1.25),
            _row("inexact", corpus, 0.1, exact=corpus != "stress"),
        ])
    rows.append(_row("missing", "legal", 0.05))

    decision = frontier.select_head_promotions(
        rows, corpus_names=("legal", "stress"), limit=2,
    )

    assert decision["selected_ids"] == ["fast", "second"]
    assert decision["rejected_ids"] == ["inexact", "missing"]
    assert decision["per_corpus_speedup"]["fast"] == {
        "legal": 2.0, "stress": 2.0,
    }


def test_frontier_improvement_must_beat_the_accepted_exact_path_everywhere():
    rows = []
    for corpus in ("legal", "stress"):
        rows.extend([
            _row(frontier.ACCEPTED_ID, corpus, 8.0),
            _row("wins_both", corpus, 7.0),
            _row("mixed", corpus, 6.0 if corpus == "legal" else 9.0),
        ])

    decision = frontier.select_frontier_improvement(
        rows, corpus_names=("legal", "stress"),
    )

    assert decision["selected_id"] == "wins_both"
    assert decision["improvement_achieved"] is True
    assert decision["accepted_control_id"] == frontier.ACCEPTED_ID


def test_materialization_is_exact_while_pallas_head_remains_an_experimental_arm():
    _, states, architecture, weights = model_fixture()
    prepared = prepare_exact_layernorm_inference_weights(weights, architecture)
    hidden = stream1_layernorm_exact_prefix(
        states, prepared, architecture, bm=2, interpret=True,
    )
    expected_hidden = np.asarray(hidden)
    expected_q = np.asarray(
        stream1_layernorm_exact_head(hidden, prepared, architecture)
    )

    materialized = frontier.materialize_hidden(
        hidden, bm=2, interpret=True,
    )
    pallas_q = frontier.head_call(
        {
            "id": "tiny",
            "backend": "pallas",
            "control": False,
            "bm": 2,
            "bk": 8,
            "bn": 8,
            "dense_rounding": "late",
        },
        architecture,
        interpret=True,
    )(materialized, prepared)

    np.testing.assert_array_equal(np.asarray(materialized), expected_hidden)
    assert np.asarray(pallas_q).shape == expected_q.shape
    assert np.isfinite(np.asarray(pallas_q, dtype=np.float32)).all()


def test_tiny_frontier_bundle_preserves_the_accepted_exact_fallback(tmp_path):
    params, states, architecture, weights = model_fixture()
    heads = [
        {"id": frontier.JAX_HEAD_ID, "backend": "jax", "control": True},
        {
            "id": "tiny_pallas_head",
            "backend": "pallas",
            "control": False,
            "bm": 2,
            "bk": 8,
            "bn": 8,
            "dense_rounding": "late",
        },
    ]
    report = frontier.run_suite(
        params,
        artgor_reference_apply,
        architecture,
        weights,
        {"legal": np.asarray(states), "stress": np.asarray(states[::-1])},
        tmp_path,
        screen_local_batch=2,
        confirmation_local_batch=2,
        device_count=1,
        prefix_bms=(2,),
        accepted_prefix_bm=2,
        identity_bms=(),
        heads=heads,
        head_promotion_limit=1,
        warmups=1,
        repeats=1,
        interpret=True,
        collect_profiles=False,
    )

    assert report["status"] == "complete"
    assert report["screen_decision"]["selected_id"] == frontier.ACCEPTED_ID
    assert report["screen_decision"]["improvement_achieved"] is False
    assert report["confirmation_decision"]["selected_id"] == frontier.ACCEPTED_ID
    assert report["confirmation_decision"]["confirmed"] is True
    assert all(
        row["exact_oracle_on_sample"]
        for row in report["full_measurements"]
        if row["id"] == frontier.ACCEPTED_ID
    )
    saved = json.loads(
        (tmp_path / "stream1_exact_inference_frontier.json").read_text()
    )
    assert saved["status"] == "complete"
    assert list((tmp_path / "hlo").glob("*.compiled.txt"))


def test_cli_paths_are_explicit_and_do_not_depend_on_the_working_directory(tmp_path):
    output = tmp_path / "output"
    args = frontier.parse_args([
        "--dataset", str(tmp_path), "--output", str(output),
    ])

    assert args.dataset == tmp_path
    assert args.output == output
    assert frontier._dataset_path(tmp_path) == tmp_path


def test_profile_plan_includes_both_accepted_and_new_selected_split_stages():
    selected = {
        "id": "new_split",
        "backend": "split",
        "prefix_bm": 8192,
        "head_id": "pallas_head",
    }

    plan = frontier.profile_stage_ids(
        {"selected_id": "new_split"},
        [
            {
                "id": frontier.ACCEPTED_ID,
                "backend": "split",
                "prefix_bm": 2048,
                "head_id": frontier.JAX_HEAD_ID,
            },
            selected,
        ],
    )

    assert plan == {
        "compiled": [
            frontier.ORIGINAL_ID,
            frontier.TYPED_ID,
            "prefix_bm2048",
            frontier.JAX_HEAD_ID,
            "prefix_bm8192",
            "pallas_head",
        ],
        "composed": [frontier.ACCEPTED_ID, "new_split"],
    }
