from types import SimpleNamespace
import jax.numpy as jnp
import pytest
import jax


@pytest.mark.parametrize('bad', [False,True])
def test_local_coverage_flows_into_common_error(bad):
    from tpu_beam_search.beam_final_agreement import make_final_coverage_agreement
    call = make_final_coverage_agreement(SimpleNamespace(size=1),interpret=True)
    targets = jnp.arange(128,dtype=jnp.uint32)[None,:]
    if bad: targets = targets.at[0,1].set(0)
    common, summary = call(targets,jnp.ones((1,128),jnp.uint32),jnp.array([128],jnp.uint32))
    assert int(common[0,0]) == int(bad)
    assert (int(summary[0,0]) != 0) == bad


def test_eight_rank_coverage_agreement_traces():
    from tpu_beam_search.beam_final_agreement import make_final_coverage_agreement
    call = make_final_coverage_agreement(SimpleNamespace(size=8))
    shapes = (jax.ShapeDtypeStruct((1,256),jnp.uint32),
              jax.ShapeDtypeStruct((1,256),jnp.uint32),
              jax.ShapeDtypeStruct((1,),jnp.uint32))
    traced = jax.make_jaxpr(call,axis_env=[('core',8)])(*shapes)
    assert [x.aval.shape for x in traced.jaxpr.outvars] == [(1,128),(2,128)]
