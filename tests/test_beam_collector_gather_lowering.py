import jax
import jax.numpy as jnp
import pytest

from tpu_beam_search.beam_collector import pallas_collector_append, pallas_collector_scatter


def gather_equations(node):
    if hasattr(node,'jaxpr'):
        yield from gather_equations(node.jaxpr)
    elif hasattr(node,'eqns'):
        for eqn in node.eqns:
            if eqn.primitive.name == 'gather':
                yield eqn
            yield from gather_equations(eqn.params)
    elif isinstance(node,dict):
        for value in node.values():
            yield from gather_equations(value)
    elif isinstance(node,(tuple,list)):
        for value in node:
            yield from gather_equations(value)


@pytest.mark.parametrize('kind',['append','scatter'])
def test_collector_gathers_meet_mosaic_take_along_axis_shape_contract(kind):
    if kind == 'append':
        fn = pallas_collector_append
        shapes = ((8,256),(8,256),(8,128),(8,128),(1,128))
    else:
        fn = pallas_collector_scatter
        shapes = ((2,8,512),(2,8,512),(8,512),(2,8,128),(1,128),(1,128))
    traced = jax.make_jaxpr(fn)(*(jax.ShapeDtypeStruct(s,jnp.uint32) for s in shapes))
    gathers = list(gather_equations(traced))
    assert gathers
    for eqn in gathers:
        operand,indices = (v.aval.shape for v in eqn.invars[:2])
        result = eqn.outvars[0].aval.shape
        assert len(operand) == len(result)
        assert indices == (*result,1), (operand,indices,result)


def test_scatter_has_no_dynamic_scalar_vmem_offset_load():
    shapes = ((2,8,512),(2,8,512),(8,512),(2,8,128),(1,128),(1,128))
    traced = jax.make_jaxpr(pallas_collector_scatter)(
        *(jax.ShapeDtypeStruct(s,jnp.uint32) for s in shapes))
    def walk(node):
        if hasattr(node,'jaxpr'):
            yield from walk(node.jaxpr)
        elif hasattr(node,'eqns'):
            for eqn in node.eqns:
                yield eqn
                yield from walk(eqn.params)
        elif isinstance(node,dict):
            for value in node.values():
                yield from walk(value)
        elif isinstance(node,(tuple,list)):
            for value in node:
                yield from walk(value)
    bad = [eqn for eqn in walk(traced) if eqn.primitive.name == 'get'
           and getattr(eqn.invars[0].aval,'shape',None) == (1,128)
           and eqn.outvars[0].aval.shape == ()
           and any(not hasattr(v,'val') for v in eqn.invars[1:])]
    assert not bad, bad


def test_scatter_gathers_do_not_span_multiple_source_lane_registers():
    shapes = ((2,8,512),(2,8,512),(8,512),(2,8,128),(1,128),(1,128))
    traced = jax.make_jaxpr(pallas_collector_scatter)(
        *(jax.ShapeDtypeStruct(s,jnp.uint32) for s in shapes))
    gathers = list(gather_equations(traced))
    assert gathers
    assert all(eqn.invars[0].aval.shape[1] <= 128 for eqn in gathers)
