import importlib.util
import numpy as np
import pytest
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from tpu_beam_search.beam_k1_neighborhood import prepare_k1_neighborhood
from test_beam_k1_neighborhood import inputs


@pytest.mark.parametrize('parent_count', [0, 1, 3, 4])
def test_stream2_k1_retains_immediate_hashes_and_recognizes_neighborhood_hits(parent_count):
    assert importlib.util.find_spec('tpu_beam_search.beam_stream2_k1') is not None
    from tpu_beam_search.beam_stream2_k1 import pallas_hash_k1_goal
    central,g,z = inputs()
    neighborhood = prepare_k1_neighborhood(central,g,z,state_len=3,radius=1,max_entries=3,bucket_count=4)
    parents = np.tile(central,(4,1))
    parents[1,:3],parents[2,:3],parents[3,:3] = [1,0,2],[2,0,1],[2,2,2]
    actual = pallas_hash_k1_goal(*map(jnp.asarray,(parents,g,central,z)),
        jnp.array([parent_count],jnp.uint32),jnp.asarray(neighborhood.table.words),bucket_count=4,
        interpret=pltpu.InterpretParams(detect_races=True))
    hashes = np.zeros((4,128),np.uint32)
    hits,valid = np.zeros((1,128),np.uint32),np.zeros((1,128),np.uint32)
    for parent in range(parent_count):
        for move in range(2):
            index = parent*2+move
            child = parents[parent,g[move]]
            key = np.bitwise_xor.reduce(z[:,np.arange(128)*3+child],axis=1)
            hashes[:,index] = key
            hits[0,index] = int(tuple(map(int,key)) in neighborhood.suffix_by_hash)
            valid[0,index] = 1
    if parent_count >= 3:
        np.testing.assert_array_equal(hits[0,:6],[1,0,1,0,0,1])
    for got,expected in zip(actual,(hashes,hits,valid),strict=True):
        np.testing.assert_array_equal(got,expected)
