import importlib.util
import numpy as np
import pytest


def test_fixed_k1_table_packs_four_slots_in_input_order_and_rejects_overflow():
    assert importlib.util.find_spec('tpu_beam_search.beam_k1_table') is not None
    from tpu_beam_search.beam_k1_table import prepare_k1_table
    hashes = np.arange(20,dtype=np.uint32).reshape(4,5)
    table = prepare_k1_table(hashes[:,:4],bucket_count=1)
    assert table.count == 4 and table.bucket_count == 1
    assert table.words.shape == (5,128)
    np.testing.assert_array_equal(table.words[1:,:4],hashes[:,:4])
    assert np.all(table.words[0,:4] != 0)
    assert not table.words[:,4:].any() and not table.words.flags.writeable
    with pytest.raises(ValueError,match='fit'):
        prepare_k1_table(hashes,bucket_count=1)


def test_fixed_k1_table_placement_matches_first_free_order_with_pallas_keys():
    from tpu_beam_search.beam_k1_table import prepare_k1_table
    from tpu_beam_search.beam_k1_keys import pallas_k1_keys
    import jax.numpy as jnp
    hashes = np.random.default_rng(607).integers(0,2**32,(4,128),dtype=np.uint32)
    table = prepare_k1_table(hashes,bucket_count=128)
    keys = np.asarray(pallas_k1_keys(jnp.asarray(hashes),bucket_count=128,interpret=True))
    used = np.zeros(512,bool)
    for query in range(128):
        # Independently walk the original first-free placement with device keys.
        choices = [int(keys[c,query])*4+i for c in (1,2) for i in range(4)]
        slot = next(s for s in choices if not used[s])
        used[slot] = True
        assert table.words[0,slot] == keys[0,query]
        np.testing.assert_array_equal(table.words[1:,slot],hashes[:,query])


def test_fixed_arena_rejects_bucket_collision_even_with_unused_global_slots():
    from tpu_beam_search.beam_k1_table import prepare_k1_table
    from tpu_beam_search.beam_k1_keys import pallas_k1_keys
    import jax.numpy as jnp
    hashes = np.random.default_rng(608).integers(0,2**32,(4,128),dtype=np.uint32)
    keys = np.asarray(pallas_k1_keys(jnp.asarray(hashes),bucket_count=2,interpret=True))
    selected = np.flatnonzero((keys[1] == 0)&(keys[2] == 0))[:5]
    assert len(selected) == 5
    with pytest.raises(ValueError,match='fit'):
        prepare_k1_table(hashes[:,selected],bucket_count=2)


def test_prepared_table_round_trips_through_hbm_lookup_and_empty_table_misses():
    from tpu_beam_search.beam_k1_table import prepare_k1_table
    from tpu_beam_search.beam_k1_lookup import pallas_k1_contains
    from jax.experimental.pallas import tpu as pltpu
    import jax.numpy as jnp
    hashes = np.random.default_rng(609).integers(0,2**32,(4,128),dtype=np.uint32)
    hashes[:,0] = 0  # Zero is a valid Hash128, not the empty-slot marker.
    table = prepare_k1_table(hashes[:,:3],bucket_count=4)
    count = jnp.array([4],jnp.uint32)
    got = pallas_k1_contains(jnp.asarray(hashes),jnp.asarray(table.words),count,
        bucket_count=4,interpret=pltpu.InterpretParams(detect_races=True))
    expected = np.zeros((1,128),np.uint32)
    expected[0,:3] = 1
    np.testing.assert_array_equal(got,expected)
    empty = prepare_k1_table(hashes[:,:0],bucket_count=4)
    got = pallas_k1_contains(jnp.asarray(hashes),jnp.asarray(empty.words),count,
        bucket_count=4,interpret=pltpu.InterpretParams(detect_races=True))
    np.testing.assert_array_equal(got,np.zeros_like(expected))
