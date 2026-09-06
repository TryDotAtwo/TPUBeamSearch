import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_periodic_threshold_scan_contains_no_empty_vector_outputs():
    from tpu_beam_search.beam_threshold import pallas_periodic_threshold
    shape=jax.ShapeDtypeStruct((2,128),jnp.uint32)
    traced=jax.make_jaxpr(lambda h,b,p:pallas_periodic_threshold(h,b,p,bins=128))(shape,shape,shape)
    def inspect(value):
        if isinstance(value,(tuple,list)):
            for child in value:
                inspect(child)
        elif isinstance(value,dict):
            for child in value.values():
                inspect(child)
        elif hasattr(value,'jaxpr'):
            inspect(value.jaxpr)
        elif hasattr(value,'eqns'):
            for eqn in value.eqns:
                for out in eqn.outvars:
                    assert 0 not in getattr(out.aval,'shape',()), (eqn.primitive.name,out.aval)
                inspect(eqn.params)
    inspect(traced)


@pytest.mark.parametrize('bins',[1,128,129,256])
def test_fixed_width_scan_matches_uint64_prefix_at_bin_boundaries(bins):
    from tpu_beam_search.beam_threshold import pallas_periodic_threshold
    values=np.random.default_rng(608).integers(1,1<<40,256,dtype=np.uint64)
    values[0]=np.uint64(0xffffffff)
    histogram=np.stack((values.astype(np.uint32),(values>>np.uint64(32)).astype(np.uint32)))
    total=int(values[:bins].sum(dtype=np.uint64))
    beam=np.zeros((2,128),np.uint32)
    beam[:,0]=[total&0xffffffff,total>>32]
    prior=np.zeros((2,128),np.uint32)
    result=np.asarray(pallas_periodic_threshold(*map(jnp.asarray,(histogram,beam,prior)),bins=bins,interpret=True))
    np.testing.assert_array_equal(result[:,0],[bins-1,1])
    assert not result[:,1:].any()
