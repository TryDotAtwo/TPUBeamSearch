import importlib.util
import numpy as np
import pytest


def inputs():
    central = np.zeros(128,np.uint8)
    central[:3] = [0,1,2]
    generators = np.tile(np.arange(128,dtype=np.int32),(2,1))
    generators[0,:3],generators[1,:3] = [1,0,2],[1,2,0]
    zobrist = np.random.default_rng(610).integers(0,2**32,(4,128*3),dtype=np.uint32)
    zobrist[:,9:] = 0
    return central,generators,zobrist


def test_inverse_neighborhood_retains_first_bfs_suffix_and_replays_to_goal():
    assert importlib.util.find_spec('tpu_beam_search.beam_k1_neighborhood') is not None
    from tpu_beam_search.beam_k1_neighborhood import prepare_k1_neighborhood
    central,g,z = inputs()
    result = prepare_k1_neighborhood(central,g,z,state_len=3,radius=2,max_entries=6,bucket_count=8)
    assert result.table.count == 6
    states_and_suffixes = [((0,1,2),()),((1,0,2),(0,)),((2,0,1),(1,)),
                          ((2,1,0),(1,0)),((0,2,1),(0,1)),((1,2,0),(1,1))]
    expected_keys = []
    for values,suffix in states_and_suffixes:
        state = central.copy()
        state[:3] = values
        key = tuple(map(int,np.bitwise_xor.reduce(z[:,np.arange(128)*3+state],axis=1)))
        expected_keys.append(key)
        assert result.suffix_by_hash[key] == suffix
        for move in suffix:
            state = state[g[move]]
        np.testing.assert_array_equal(state,central)
    assert list(result.suffix_by_hash) == expected_keys
    with pytest.raises(ValueError,match='entries'):
        prepare_k1_neighborhood(central,g,z,state_len=3,radius=2,max_entries=5,bucket_count=8)


def test_neighborhood_disabled_and_hash_collision_follow_source_behavior():
    from tpu_beam_search.beam_k1_neighborhood import prepare_k1_neighborhood
    central,g,z = inputs()
    disabled = prepare_k1_neighborhood(central,g,z,state_len=3,radius=0,max_entries=6,bucket_count=8)
    assert disabled.table.count == 0 and not disabled.suffix_by_hash
    collision = prepare_k1_neighborhood(central,g,np.zeros_like(z),state_len=3,radius=2,max_entries=1,bucket_count=8)
    assert collision.table.count == 1 and collision.suffix_by_hash[(0,0,0,0)] == ()


def test_generated_neighborhood_is_queryable_by_pallas_lookup():
    from tpu_beam_search.beam_k1_neighborhood import prepare_k1_neighborhood
    from tpu_beam_search.beam_k1_lookup import pallas_k1_contains
    from jax.experimental.pallas import tpu as pltpu
    import jax.numpy as jnp
    central,g,z = inputs()
    neighborhood = prepare_k1_neighborhood(central,g,z,state_len=3,radius=2,max_entries=6,bucket_count=8)
    queries = np.zeros((4,128),np.uint32)
    queries[:,:6] = np.asarray(list(neighborhood.suffix_by_hash),np.uint32).T
    outsider = queries[:,0].copy()
    while tuple(map(int,outsider)) in neighborhood.suffix_by_hash:
        outsider[0] += np.uint32(1)
    queries[:,6] = outsider
    actual = pallas_k1_contains(jnp.asarray(queries),jnp.asarray(neighborhood.table.words),
        jnp.array([7],jnp.uint32),bucket_count=8,
        interpret=pltpu.InterpretParams(detect_races=True))
    expected = np.zeros((1,128),np.uint32)
    expected[0,:6] = 1
    np.testing.assert_array_equal(actual,expected)
