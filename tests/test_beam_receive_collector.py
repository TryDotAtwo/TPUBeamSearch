import jax.numpy as jnp
import numpy as np
import pytest
from tpu_beam_search import beam_receive_batch as module


@pytest.mark.parametrize('per_peer',[64,70])
def test_remote_epoch_snapshots_are_one_admission_not_separate_appends(per_peer):
    assert hasattr(module,'pallas_collect_received')
    wire = np.arange(9*8*128,dtype=np.uint32).reshape(9,8,128)
    wire[:2] = np.uint32(999999)  # reusable slots are not epoch snapshots
    counts = np.zeros((7,128),np.uint32)
    counts[:2,0] = per_peer
    rank = np.zeros((1,128),np.uint32)
    rank[0,0] = 3
    a = np.zeros((1,8,128),np.uint32)
    b = a.copy()
    c = np.zeros((1,8,128),np.uint32)
    aa,bb,cc,f = module.pallas_collect_received(
        *(jnp.asarray(x) for x in (a,b,wire,c,counts,rank)),interpret=True)
    ea,ec,ef = a.copy(),c.copy(),np.zeros((1,128),np.uint32)
    if per_peer == 64:
        # Epoch1 sender1 precedes epoch0 sender2. Both share one shard.
        ea[0,:,:64],ea[0,:,64:] = wire[3,:,:64],wire[2,:,:64]
        ec[0,2,0] = 128
    else:
        # 140 exceeds either128 sibling, even though two independent70 writes fit.
        ec[0,7,0] = 1
        ef[0,0] = 1
    np.testing.assert_array_equal(aa,ea)
    np.testing.assert_array_equal(bb,b)
    np.testing.assert_array_equal(cc,ec)
    np.testing.assert_array_equal(f,ef)
