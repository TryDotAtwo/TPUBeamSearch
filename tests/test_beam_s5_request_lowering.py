from types import SimpleNamespace
import jax
import jax.numpy as jnp
from tpu_beam_search.beam_s5_request import make_s5_request_call


def test_request_max_has_no_unsupported_unsigned_reduction():
    traced = jax.make_jaxpr(make_s5_request_call(SimpleNamespace(size=8)),axis_env=[('core',8)])(
        jax.ShapeDtypeStruct((1,128),jnp.uint32))
    reductions = []
    def visit(value):
        if hasattr(value,'jaxpr'):
            visit(value.jaxpr)
        elif hasattr(value,'eqns'):
            for eqn in value.eqns:
                if eqn.primitive.name == 'reduce_max':
                    reductions.append(eqn)
                visit(eqn.params)
        elif isinstance(value,dict):
            for item in value.values(): visit(item)
        elif isinstance(value,(tuple,list)):
            for item in value: visit(item)
    visit(traced)
    assert reductions
    assert all(e.invars[0].aval.dtype != jnp.uint32 for e in reductions)
