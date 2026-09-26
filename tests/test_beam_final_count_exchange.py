from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('rank', range(8))
def test_offset_wire_is_reordered_by_source_rank_before_final_prefixes(rank):
    from tpu_beam_search.beam_final_count_exchange import pallas_rank_order_final_counts
    counts = np.array([[0,1,0x80000000,7,0,9,2,0xffffffff],
                       [3,0,4,0,5,0,6,1]],np.uint32)
    wire = np.full((16,128),123,np.uint32)
    for offset in range(8):
        wire[2*offset:2*offset+2,0] = counts[:,(rank-offset)%8]
    got = pallas_rank_order_final_counts(jnp.asarray(wire),rank=rank,interpret=True)
    expected = np.zeros((2,128),np.uint32)
    expected[:,:8] = counts
    np.testing.assert_array_equal(got,expected)


def test_single_rank_actual_exchange_retains_phase_counts_and_clears_padding():
    from tpu_beam_search.beam_final_count_exchange import make_final_count_exchange
    counts = jnp.full((2,128),99,jnp.uint32).at[:,0].set(jnp.array([7,11],jnp.uint32))
    got = make_final_count_exchange(SimpleNamespace(size=1),interpret=True)(counts)
    expected = np.zeros((2,128),np.uint32)
    expected[:,0] = [7,11]
    np.testing.assert_array_equal(got,expected)


def test_eight_rank_final_count_exchange_traces_global_rank_order_control():
    from tpu_beam_search.beam_final_count_exchange import make_final_count_exchange
    fn = make_final_count_exchange(SimpleNamespace(size=8))
    trace = jax.make_jaxpr(fn,axis_env=[('core',8)])(
        jax.ShapeDtypeStruct((2,128),jnp.uint32))
    assert tuple(a.shape for a in trace.out_avals) == ((2,128),)
