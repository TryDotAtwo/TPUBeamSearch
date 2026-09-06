import jax.numpy as jnp
import numpy as np
import pytest


def test_region_write_preserves_common_prefix_and_persistent_tail():
    from tpu_beam_search.beam_scratch import plan_scratch, pallas_write_scratch_region
    p = plan_scratch(common_bytes=512,select_temp_bytes=512,materialize_temp_bytes=1024,
                     stream_temp_bytes=512,stream_persistent_bytes=512)
    arena = jnp.arange(512,dtype=jnp.uint32).reshape(p.pool_shape)
    values = jnp.full((2,128),0xfeedbeef,jnp.uint32)
    result = pallas_write_scratch_region(arena,values,region=p.materialize_temp,interpret=True)
    expected = np.arange(512,dtype=np.uint32).reshape(4,128)
    expected[1:3] = 0xfeedbeef
    np.testing.assert_array_equal(result,expected)
    # Same pool, next exclusive layout. The suffix outside selection stays live.
    result = pallas_write_scratch_region(result,jnp.zeros((1,128),jnp.uint32),
                                         region=p.select_temp,interpret=True)
    expected[1] = 0
    np.testing.assert_array_equal(result,expected)


@pytest.mark.parametrize('region',[(1,512),(0,513),(1024,512),(-512,512)])
def test_bad_region_cannot_address_outside_pool(region):
    from tpu_beam_search.beam_scratch import pallas_write_scratch_region
    with pytest.raises(ValueError):
        pallas_write_scratch_region(jnp.zeros((2,128),jnp.uint32),
            jnp.zeros((1,128),jnp.uint32),region=region,interpret=True)


def test_region_read_extracts_only_requested_rows():
    from tpu_beam_search.beam_scratch import pallas_read_scratch_region
    arena = jnp.arange(512,dtype=jnp.uint32).reshape(4,128)
    result = pallas_read_scratch_region(arena,region=(512,1024),interpret=True)
    np.testing.assert_array_equal(result,np.arange(128,384,dtype=np.uint32).reshape(2,128))


@pytest.mark.parametrize('region',[(1,512),(512,1024),(-512,512),(0,513)])
def test_read_rejects_invalid_region(region):
    from tpu_beam_search.beam_scratch import pallas_read_scratch_region
    with pytest.raises(ValueError):
        pallas_read_scratch_region(jnp.zeros((2,128),jnp.uint32),region=region,interpret=True)


def test_empty_region_is_noop_and_does_not_compile_empty_dma_grid():
    from tpu_beam_search.beam_scratch import pallas_read_scratch_region, pallas_write_scratch_region
    arena = jnp.arange(256,dtype=jnp.uint32).reshape(2,128)
    empty = pallas_read_scratch_region(arena,region=(1024,0),interpret=True)
    assert empty.shape == (0,128)
    result = pallas_write_scratch_region(arena,empty,region=(1024,0),interpret=True)
    np.testing.assert_array_equal(result,arena)


def test_write_rejects_payload_larger_than_region():
    from tpu_beam_search.beam_scratch import pallas_write_scratch_region
    with pytest.raises(ValueError):
        pallas_write_scratch_region(jnp.zeros((4,128),jnp.uint32),
            jnp.zeros((2,128),jnp.uint32),region=(512,512),interpret=True)
