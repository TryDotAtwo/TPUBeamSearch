import jax.numpy as jnp
import numpy as np
import pytest

from tpu_beam_search.beam_types import pack_candidates, unpack_candidates
from tpu_beam_search.beam_dedup import pallas_threshold_dedup


@pytest.mark.parametrize('mode', ['stream3', 'stream4'])
def test_dedup_preserves_128bit_identity_and_stage_tiebreak(mode):
    # Same low hash but distinct high hash. Parent tie order opposes payload order.
    words = pack_candidates([5, 5, 2**100 + 5, 0, 9],
                            [2**40, 1, 7, 0, 2], [8, 8, 3, 0, 11],
                            [3, 4, 6, 0, 1], capacity=128)
    payload = np.zeros((1, 128), np.uint32)
    payload[0, :5] = [0, 1, 2, 3, 4]
    result, count = pallas_threshold_dedup(jnp.array(words), jnp.array(payload),
        jnp.array([5], jnp.uint32), jnp.array([10], jnp.uint32), mode=mode, interpret=True)
    assert int(count[0]) == 3
    hashes, parents, scores, routes = unpack_candidates(np.array(result), count=3)
    assert hashes == [0, 5, 2**100 + 5]
    assert parents == [0, 2**40 if mode == 'stream3' else 1, 7]
    assert scores == [0, 8, 3]
    assert routes == [0, 3 if mode == 'stream3' else 4, 6]


@pytest.mark.parametrize('count', [0, 1, 128])
def test_uintmax_threshold_does_not_admit_padding(count):
    words = pack_candidates(list(range(128)), list(range(128)), [0] * 128, [0] * 128, capacity=128)
    out, n = pallas_threshold_dedup(jnp.array(words), jnp.zeros((1, 128), jnp.uint32),
        jnp.array([count], jnp.uint32), jnp.array([0xffffffff], jnp.uint32),
        mode='stream4', interpret=True)
    assert int(n[0]) == count
    assert np.all(np.asarray(out)[6, count:] == 0xffffffff)


def test_stream4_tie_break_uses_route_after_full_parent():
    words = pack_candidates([1, 1, 1], [2**32, 2**32, 2**32 + 1], [1, 1, 1], [9, 2, 0], capacity=128)
    out, n = pallas_threshold_dedup(jnp.array(words), jnp.zeros((1, 128), jnp.uint32),
        jnp.array([3], jnp.uint32), jnp.array([1], jnp.uint32), mode='stream4', interpret=True)
    assert int(n[0]) == 1
    assert unpack_candidates(np.asarray(out), count=1)[3] == [2]
