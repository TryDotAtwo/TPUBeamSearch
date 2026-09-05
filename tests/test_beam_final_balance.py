import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize('count', [0, 3, 131, (1 << 32) + 37])
def test_balance_matches_cpp_global_index_formula(count):
    from tpu_beam_search.beam_final_balance import pallas_final_balance
    world = 8
    starts = [(rank * count + world - 1) // world for rank in range(world + 1)]
    probes = [max(0, boundary + offset) for boundary in starts for offset in (-1, 0, 1)]
    probes = (probes * 10)[:256]
    indices = np.array([[x & 0xffffffff for x in probes], [x >> 32 for x in probes]], np.uint32)
    boundaries = np.zeros((2, 128), np.uint32)
    boundaries[0, :world+1] = [x & 0xffffffff for x in starts]
    boundaries[1, :world+1] = [x >> 32 for x in starts]
    rank, local, valid = pallas_final_balance(jnp.asarray(indices), jnp.asarray(boundaries), world_size=world, interpret=True)
    expected = np.zeros((3,256), np.uint32)
    for i, index in enumerate(probes):
        if index < count:
            target = index * world // count
            expected[:,i] = target, index - starts[target], 1
    np.testing.assert_array_equal(np.concatenate((rank,local,valid)), expected)
