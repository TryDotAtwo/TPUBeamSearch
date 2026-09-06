import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('n',[128,256,512])
def test_history_projection_in_one_arena_preserves_unrelated_regions(n):
    from tpu_beam_search.beam_history_arena import pallas_history_in_arena
    tiles = n//128
    meta = np.arange(8*n,dtype=np.uint32).reshape(8,n)
    meta[5] += np.uint32(0x80000000)
    arena = np.full((13*tiles+4,128),0xdeadbeef,np.uint32)
    arena[1:1+8*tiles] = meta.reshape(8*tiles,128)
    target = np.arange(n,dtype=np.uint32)[None,:][...,::-1].copy()
    valid = (np.arange(n)%3 != 0).astype(np.uint32)[None,:]*7
    actual = pallas_history_in_arena(jnp.asarray(arena),jnp.asarray(target),jnp.asarray(valid),
                                    meta_offset=512,history_offset=(1+8*tiles)*512,interpret=True)
    expected = arena.copy()
    history = np.stack((meta[4],meta[5],meta[7],target[0],(valid[0]!=0).astype(np.uint32)))
    history[:,valid[0]==0] = 0
    expected[1+8*tiles:1+13*tiles] = history.reshape(5*tiles,128)
    np.testing.assert_array_equal(actual,expected)


def test_overlapping_history_output_is_rejected():
    from tpu_beam_search.beam_history_arena import pallas_history_in_arena
    with pytest.raises(ValueError):
        pallas_history_in_arena(jnp.zeros((20,128),jnp.uint32),jnp.zeros((1,128),jnp.uint32),
                               jnp.ones((1,128),jnp.uint32),meta_offset=0,history_offset=512,interpret=True)
