import numpy as np
import jax.numpy as jnp


def test_reserved_s4_job_dedups_clean_dirty_and_publishes_matching_histogram():
    from tpu_beam_search import beam_s4_commit as module
    assert hasattr(module,'pallas_run_reserved_s4')
    words = np.zeros((8,128),np.uint32)
    words[0,:4] = [1,2,1,3]
    words[6,:4] = [7,5,3,8]
    words[4,:4] = [10,20,30,40]
    hist_a,hist_b = np.full((1,128),11,np.uint32),np.full((1,128),22,np.uint32)
    c = np.zeros((4,128),np.uint32)
    c[:,0] = (2,2,1,1)
    actual = module.pallas_run_reserved_s4(*map(jnp.asarray,(words,hist_a,hist_b,c)),
        jnp.array([7],jnp.uint32),bins=128,interpret=True)
    want = np.zeros_like(words)
    want[6] = np.uint32(0xffffffff)
    want[:,:2] = words[:,[2,1]]
    hist = np.zeros_like(hist_a)
    hist[0,3],hist[0,5] = 1,1
    c[:,0] = (2,0,0,0)
    for got,expected in zip(actual,(want,hist,hist_b,c)):
        np.testing.assert_array_equal(got,expected)

    # Simulate the next reservation; the data and histogram arrays are the
    # returned versions, not stale pre-alias inputs. No device scheduler claim.
    again_control = np.asarray(actual[3]).copy()
    again_control[2,0] = 1
    again = module.pallas_run_reserved_s4(*actual[:3],jnp.asarray(again_control),
        jnp.array([0],jnp.uint32),bins=128,interpret=True)
    empty = np.zeros_like(want)
    empty[6] = np.uint32(0xffffffff)
    again_control[:,0] = (0,0,0,1)
    for got,expected in zip(again,(empty,hist,np.zeros_like(hist),again_control)):
        np.testing.assert_array_equal(got,expected)
