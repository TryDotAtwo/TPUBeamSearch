from types import SimpleNamespace
import importlib.util
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('requested',[0,1])
def test_single_rank_request_retains_only_request_lane(requested):
    assert importlib.util.find_spec('tpu_beam_search.beam_s5_request') is not None
    from tpu_beam_search.beam_s5_request import make_s5_request_call
    request = np.full((1,128),19,np.uint32)
    request[0,0] = requested
    actual = make_s5_request_call(SimpleNamespace(size=1),interpret=True)(jnp.asarray(request))
    expected = np.zeros_like(request)
    expected[0,0] = requested
    np.testing.assert_array_equal(actual,expected)


def test_eight_rank_request_traces_with_preserved_control_abi():
    from tpu_beam_search.beam_s5_request import make_s5_request_call
    fn = make_s5_request_call(SimpleNamespace(size=8))
    traced = jax.make_jaxpr(fn,axis_env=[('core',8)])(
        jax.ShapeDtypeStruct((1,128),jnp.uint32))
    assert tuple(x.shape for x in traced.out_avals) == ((1,128),)
