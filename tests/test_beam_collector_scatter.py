import jax.numpy as jnp
import numpy as np
import pytest

from tpu_beam_search import beam_collector as module


@pytest.mark.parametrize('mode',['success','overflow','empty'])
def test_partition_scatter_is_whole_batch_and_handles_unaligned_ranges(mode):
    assert hasattr(module,'pallas_collector_scatter')
    a = np.arange(2*8*512,dtype=np.uint32).reshape(2,8,512)
    b = a + np.uint32(100000)
    incoming = np.arange(8*512,dtype=np.uint32).reshape(8,512)+np.uint32(200000)
    c = np.zeros((2,8,128),np.uint32)
    c[0,0,0],c[0,1,0] = 400,7
    c[1,0,0],c[1,1,0] = 383,512
    counts = np.zeros((1,128),np.uint32)
    counts[0,:2] = [129,130 if mode == 'overflow' else 129]
    if mode == 'empty':
        counts[:] = 0
        c[:,4:6,0] = 1
    offsets = np.zeros_like(counts)
    offsets[0,1:3] = np.cumsum(counts[0,:2])
    aa,bb,cc,fatal = module.pallas_collector_scatter(
        *(jnp.asarray(x) for x in (a,b,incoming,c,counts,offsets)),interpret=True)
    ea,eb,ec = a.copy(),b.copy(),c.copy()
    if mode == 'success':
        eb[0,:,7:136] = incoming[:,:129]
        ea[1,:,383:512] = incoming[:,129:258]
        ec[0,3,0],ec[0,6,0],ec[1,2,0] = 129,1,129
    elif mode == 'overflow':
        ec[:,7,0] = 1
    np.testing.assert_array_equal(aa,ea)
    np.testing.assert_array_equal(bb,eb)
    np.testing.assert_array_equal(cc,ec)
    ef = np.zeros((1,128),np.uint32)
    ef[0,0] = int(mode == 'overflow')
    np.testing.assert_array_equal(fatal,ef)
