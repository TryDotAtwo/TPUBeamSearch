import importlib.util
import numpy as np
import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu
from tpu_beam_search.beam_k1_keys import pallas_k1_keys


def test_k1_lookup_checks_full_hash_not_only_fingerprint_and_masks_padding():
    assert importlib.util.find_spec('tpu_beam_search.beam_k1_lookup') is not None
    from tpu_beam_search.beam_k1_lookup import pallas_k1_contains
    hashes = np.random.default_rng(606).integers(0,2**32,(4,128),dtype=np.uint32)
    keys = np.asarray(pallas_k1_keys(jnp.asarray(hashes),bucket_count=64,interpret=True))
    table = np.zeros((5,256),np.uint32)
    # Place one true hit in each bucket path; a fingerprint collision is a miss.
    occupied = set()
    for query,bucket_index,collision in ((0,1,False),(1,2,False),(2,1,True)):
        base = int(keys[bucket_index,query])*4
        slot = next(s for s in range(base,base+4) if s not in occupied)
        occupied.add(slot)
        table[0,slot] = keys[0,query]
        table[1:,slot] = hashes[:,query]
        if collision:
            table[4,slot] ^= np.uint32(1)
    # A real table hit outside the valid query count must not escape.
    hashes[:,3] = hashes[:,0]
    got = pallas_k1_contains(jnp.asarray(hashes),jnp.asarray(table),jnp.array([3],jnp.uint32),
        bucket_count=64,interpret=pltpu.InterpretParams(detect_races=True))
    expected = np.zeros((1,128),np.uint32)
    expected[0,:2] = 1
    np.testing.assert_array_equal(got,expected)


@pytest.mark.parametrize('slot',[0,1,2,3,4,127])
def test_k1_lookup_uses_exactly_four_slots_even_when_dma_contains_padding(slot):
    from tpu_beam_search.beam_k1_lookup import pallas_k1_contains
    hashes = np.zeros((4,128),np.uint32)
    keys = np.asarray(pallas_k1_keys(jnp.asarray(hashes),bucket_count=1,interpret=True))
    table = np.zeros((5,128),np.uint32)
    table[0,slot] = keys[0,0]
    got = pallas_k1_contains(jnp.asarray(hashes),jnp.asarray(table),jnp.array([1],jnp.uint32),
        bucket_count=1,interpret=pltpu.InterpretParams(detect_races=True))
    expected = np.zeros((1,128),np.uint32)
    expected[0,0] = int(slot < 4)
    np.testing.assert_array_equal(got,expected)
