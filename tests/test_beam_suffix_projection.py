import importlib.util
import itertools
import numpy as np
import jax.numpy as jnp
from tpu_beam_search.beam_suffix_table import prepare_k2_suffix_table


def test_pallas_suffix_projection_preserves_noncommuting_move_order_and_padding():
    assert importlib.util.find_spec('tpu_beam_search.beam_suffix_projection') is not None
    from tpu_beam_search.beam_suffix_projection import pallas_suffix_projection
    g = np.tile(np.arange(128,dtype=np.int32),(2,1))
    g[0,:3] = [1,0,2]
    g[1,:3] = [1,2,0]
    table = prepare_k2_suffix_table(move_count=2,radius=2)
    got = pallas_suffix_projection(jnp.asarray(g),jnp.asarray(table.words),
                                   count=table.count,interpret=True)
    want = np.zeros((128,128),np.int32)
    chains = [(),(0,),(1,),(0,0),(0,1),(1,0),(1,1)]
    for suffix,chain in enumerate(chains):
        state = np.arange(128,dtype=np.int32)
        for move in chain:
            state = state[g[move]]
        want[:,suffix] = state
    np.testing.assert_array_equal(got,want)
    assert not np.array_equal(want[:,4],want[:,5])


def test_suffix_projection_crosses_suffix_tiles_at_radius_three():
    from tpu_beam_search.beam_suffix_projection import pallas_suffix_projection
    rng = np.random.default_rng(604)
    g = np.stack([rng.permutation(128).astype(np.int32) for _ in range(5)])
    table = prepare_k2_suffix_table(move_count=5,radius=3)
    assert table.count == 156
    got = pallas_suffix_projection(jnp.asarray(g),jnp.asarray(table.words),
                                   count=table.count,interpret=True)
    want = np.zeros((128,256),np.int32)
    suffix = 0
    for depth in range(4):
        for chain in itertools.product(range(5),repeat=depth):
            state = np.arange(128,dtype=np.int32)
            for move in chain:
                state = state[g[move]]
            want[:,suffix] = state
            suffix += 1
    np.testing.assert_array_equal(got,want)
