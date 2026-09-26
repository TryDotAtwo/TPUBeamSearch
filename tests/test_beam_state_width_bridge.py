"""Compact persistent states must round-trip through TPU final-kernel tiles."""
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu


@pytest.mark.parametrize("state_len,move_count,compact,tile", [
    (120, 24, 128, 128),
    (150, 30, 160, 256),
])
def test_pallas_state_width_bridge_preserves_payload_and_zeros_tile_tail(
    state_len, move_count, compact, tile,
):
    from tpu_beam_search.beam_types import BeamStorage
    from tpu_beam_search.beam_state_width_bridge import (
        pallas_compact_state_rows, pallas_expand_state_rows,
    )

    layout = BeamStorage(state_len, move_count, 128)
    assert layout.STATE_STORAGE_LEN == compact
    assert layout.STATE_KERNEL_LEN == tile
    source = np.zeros((128, compact), np.uint8)
    source[:, :state_len] = np.arange(128 * state_len, dtype=np.uint32).reshape(128, state_len) % 256
    source[:, state_len:state_len + 4] = np.array([1, 2, 3, 4], np.uint8)
    source[:, state_len + 4:] = 255  # Dirty persistent padding is never published.
    mode = pltpu.InterpretParams(detect_races=True)

    expanded = pallas_expand_state_rows(jnp.asarray(source), state_len=state_len,
                                        kernel_width=tile, interpret=mode)
    want_expanded = np.zeros((128, tile), np.uint8)
    want_expanded[:, :state_len + 4] = source[:, :state_len + 4]
    np.testing.assert_array_equal(expanded, want_expanded)

    compacted = pallas_compact_state_rows(expanded, state_len=state_len,
                                          storage_width=compact, interpret=mode)
    np.testing.assert_array_equal(compacted, want_expanded[:, :compact])


def test_pallas_state_width_bridge_rejects_unaligned_and_truncating_shapes():
    from tpu_beam_search.beam_state_width_bridge import pallas_expand_state_rows
    with pytest.raises(ValueError):
        pallas_expand_state_rows(jnp.zeros((1, 160), jnp.uint8),
                                 state_len=150, kernel_width=256, interpret=True)
    with pytest.raises(ValueError):
        pallas_expand_state_rows(jnp.zeros((128, 160), jnp.uint8),
                                 state_len=150, kernel_width=128, interpret=True)


def test_compact_150_parent_and_generator_feed_final_materialization():
    from tpu_beam_search.beam_state_width_bridge import (
        pallas_compact_state_rows, pallas_expand_generator_rows,
        pallas_expand_state_rows,
    )
    from tpu_beam_search.beam_final_materialize import pallas_materialize_final

    compact = np.zeros((128, 160), np.uint8)
    compact[0, :150] = np.arange(150, dtype=np.uint8)
    compact[0, 150:154] = [5, 6, 7, 8]
    generator = np.zeros((30, 160), np.int32)
    generator[0, :150] = np.arange(149, -1, -1, dtype=np.int32)
    requests = np.zeros((4, 128), np.uint32)
    requests[2, 0] = 9
    mode = pltpu.InterpretParams(detect_races=True)

    parents_tile = pallas_expand_state_rows(jnp.asarray(compact), state_len=150,
                                            kernel_width=256, interpret=mode)
    generators_tile = pallas_expand_generator_rows(jnp.asarray(generator), state_len=150,
                                                   kernel_width=256, interpret=mode)
    wire, errors = pallas_materialize_final(
        parents_tile, generators_tile, jnp.asarray(requests),
        jnp.array([1], jnp.uint32), jnp.array([128], jnp.uint32),
        state_len=150, interpret=mode,
    )
    compact_wire = pallas_compact_state_rows(wire, state_len=150,
                                             storage_width=160, interpret=mode)
    want = np.zeros((128, 160), np.uint8)
    want[0, :150] = np.arange(149, -1, -1, dtype=np.uint8)
    want[0, 150] = 9
    np.testing.assert_array_equal(compact_wire, want)
    assert int(errors[0, 0]) == 0
