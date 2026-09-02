import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path

import benchmarks.artgor_pallas_same_suffix as benchmark

from benchmarks.artgor_pallas_same_suffix import (
    reference_embedding,
    reference_hidden_after_depth,
    reference_input_dense,
    reference_suffix,
    reference_suffix_from_embedding,
    reference_suffix_from_input_dense,
)
from test_layernorm_followup import model_fixture
from tpu_beam_search.stream1_layernorm_reference import (
    stream1_layernorm_reference_inference,
)


def test_boundary_plus_same_suffix_reconstructs_typed_reference():
    _, states, architecture, weights = model_fixture()
    expected = stream1_layernorm_reference_inference(states, weights, architecture)
    for depth in range(architecture.RESIDUAL_COUNT + 1):
        hidden = reference_hidden_after_depth(states, weights, architecture, depth)
        actual = reference_suffix(hidden, weights, architecture, depth)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    embedded = reference_embedding(states, weights, architecture)
    dense = reference_input_dense(embedded, weights)
    np.testing.assert_array_equal(
        np.asarray(reference_suffix_from_embedding(embedded, weights, architecture)),
        np.asarray(expected),
    )
    np.testing.assert_array_equal(
        np.asarray(reference_suffix_from_input_dense(dense, weights, architecture)),
        np.asarray(expected),
    )


def test_depth_contract_rejects_invalid_boundaries():
    _, states, architecture, weights = model_fixture()
    with pytest.raises(ValueError, match="outside"):
        reference_hidden_after_depth(states, weights, architecture, -1)
    with pytest.raises(ValueError, match="outside"):
        reference_suffix(jnp.zeros((2, architecture.HIDDEN2)), weights, architecture, 99)


def test_main_passes_optional_dataset_through_shared_resolver(monkeypatch, tmp_path):
    resolved = Path("resolved-dataset")
    seen = {}
    monkeypatch.setattr(benchmark, "_dataset_path", lambda explicit: resolved if explicit is None else explicit)
    monkeypatch.setattr(benchmark, "run", lambda **kwargs: seen.update(kwargs) or {"status": "complete"})

    benchmark.main(["--output", str(tmp_path)])

    assert seen == {"dataset": resolved, "output": tmp_path}


def test_isolated_control_does_not_attribute_prefix_drift_to_exact_operator():
    assert hasattr(benchmark, "compare_isolated_operator")
    x = jnp.array([[2., 3.]])
    result = benchmark.compare_isolated_operator(
        x, reference_op=lambda h: h * 2, candidate_op=lambda h: h * 2,
        suffix=lambda h: h + 1, monolithic=jnp.array([[9., 9.]]),
        prefix_output=jnp.array([[7., 7.]]),
    )
    assert result["boundary"]["exact"]
    assert result["candidate_vs_same_suffix"]["exact"]
    assert not result["same_suffix_control_vs_monolithic"]["exact"]
    assert not result["isolated_reference_vs_prefix"]["exact"]
