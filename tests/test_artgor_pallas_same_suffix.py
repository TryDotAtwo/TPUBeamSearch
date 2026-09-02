import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.artgor_pallas_same_suffix import (
    reference_hidden_after_depth,
    reference_suffix,
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


def test_depth_contract_rejects_invalid_boundaries():
    _, states, architecture, weights = model_fixture()
    with pytest.raises(ValueError, match="outside"):
        reference_hidden_after_depth(states, weights, architecture, -1)
    with pytest.raises(ValueError, match="outside"):
        reference_suffix(jnp.zeros((2, architecture.HIDDEN2)), weights, architecture, 99)
