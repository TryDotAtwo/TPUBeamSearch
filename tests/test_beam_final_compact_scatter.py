"""A tile-width final response must publish compact persistent rows safely."""
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu


@pytest.mark.parametrize("prior_error", [0, 7])
def test_compact_scatter_clears_response_index_and_preserves_other_rows(prior_error):
    from tpu_beam_search.beam_final_scatter import pallas_scatter_compact_final_responses

    frontier = np.zeros((128, 160), np.uint8)
    frontier[:, :150] = 0xa5
    wire = np.full((128, 256), 0xee, np.uint8)
    wire[0, :150] = np.arange(150, dtype=np.uint8)
    wire[0, 150:154] = [9, 0, 0, 0]
    prior = np.zeros((1, 128), np.uint32)
    prior[0, 0] = prior_error
    mode = pltpu.InterpretParams(detect_races=True)

    actual, errors = pallas_scatter_compact_final_responses(
        jnp.asarray(frontier), jnp.asarray(wire), jnp.array([1], jnp.uint32),
        state_len=150, prior_error=jnp.asarray(prior), interpret=mode,
    )
    want = frontier.copy()
    if prior_error == 0:
        want[9, :150] = np.arange(150, dtype=np.uint8)
        want[9, 150:] = 0
    np.testing.assert_array_equal(actual, want)
    assert (int(errors[0, 0]) != 0) == bool(prior_error)


def test_compact_scatter_rejects_target_overflow_without_partial_write():
    from tpu_beam_search.beam_final_scatter import pallas_scatter_compact_final_responses

    frontier = np.zeros((128, 160), np.uint8)
    frontier[:, :150] = 73
    wire = np.zeros((128, 256), np.uint8)
    wire[0, :150] = 11
    wire[0, 150:154] = [128, 0, 0, 0]
    actual, errors = pallas_scatter_compact_final_responses(
        jnp.asarray(frontier), jnp.asarray(wire), jnp.array([1], jnp.uint32),
        state_len=150, interpret=pltpu.InterpretParams(detect_races=True),
    )
    np.testing.assert_array_equal(actual, frontier)
    assert int(errors[0, 0]) != 0
