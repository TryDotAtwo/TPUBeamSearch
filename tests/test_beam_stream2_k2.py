import numpy as np
import pytest
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from test_beam_k1_neighborhood import inputs
from tpu_beam_search.beam_k1_neighborhood import prepare_k1_neighborhood
from tpu_beam_search.beam_suffix_table import prepare_k2_suffix_table


@pytest.mark.parametrize('parent_count,radius', [(0,2),(3,0),(3,2)])
def test_complete_diagnostic_k2_preserves_hashes_and_first_suffix(parent_count,radius):
    from tpu_beam_search.beam_stream2_k2 import pallas_hash_k2_goal
    central,g,z = inputs()
    neighborhood = prepare_k1_neighborhood(central,g,z,state_len=3,radius=1,max_entries=3,bucket_count=4)
    suffixes = prepare_k2_suffix_table(move_count=2,radius=radius)
    parents = np.tile(central,(4,1))
    parents[1,:3],parents[2,:3] = [0,2,1],[2,2,2]
    actual = pallas_hash_k2_goal(*map(jnp.asarray,(parents,g,central,z)),
        jnp.array([parent_count],jnp.uint32),jnp.asarray(neighborhood.table.words),
        jnp.asarray(suffixes.words),bucket_count=4,suffix_count=suffixes.count,
        interpret=pltpu.InterpretParams(detect_races=True))
    hashes = np.zeros((4,128),np.uint32)
    found_hash = hashes.copy()
    hit,valid,ids = (np.zeros((1,128),np.uint32) for _ in range(3))
    def hash_state(state):
        return np.bitwise_xor.reduce(z[:,np.arange(128)*3+state],axis=1)
    for parent in range(parent_count):
        for move in range(2):
            i = parent*2+move
            child = parents[parent,g[move]]
            hashes[:,i] = found_hash[:,i] = hash_state(child)
            valid[0,i] = 1
            for suffix in range(suffixes.count):
                projected = child.copy()
                packed = int(suffixes.words[0,suffix])
                for step in range(int(suffixes.words[2,suffix])):
                    projected = projected[g[(packed>>(5*step))&31]]
                h = hash_state(projected)
                if tuple(map(int,h)) in neighborhood.suffix_by_hash:
                    found_hash[:,i],hit[0,i],ids[0,i] = h,1,suffix
                    break
    if parent_count and radius:
        assert np.any(ids != 0)
    for got,want in zip(actual,(hashes,hit,valid,found_hash,ids),strict=True):
        np.testing.assert_array_equal(got,want)
