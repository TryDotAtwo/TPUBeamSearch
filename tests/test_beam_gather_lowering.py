"""Structural regression for the V1 TPU gather rejection, not TPU execution."""
import jax
from jax import lax
import pytest

from benchmarks.beam_primitive_bundle import build_cases


def equations(value):
    if hasattr(value, 'jaxpr'):
        yield from equations(value.jaxpr)
    elif hasattr(value, 'eqns'):
        for eqn in value.eqns:
            yield eqn
            yield from equations(eqn.params)
    elif isinstance(value, dict):
        for child in value.values():
            yield from equations(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from equations(child)


@pytest.mark.parametrize('name', ['hash_goal_120_24', 'hash_goal_150_30', 'dedup_stream3_128'])
def test_gathers_meet_mosaic_supported_shape_and_mode(name):
    case = next(c for c in build_cases(interpret=True) if c['name'] == name)
    graph = jax.make_jaxpr(case['fn'])(*case['args'])
    gathers = [eq for eq in equations(graph) if eq.primitive.name == 'gather']
    assert gathers
    for eq in gathers:
        operand, indices = (var.aval for var in eq.invars)
        result = eq.outvars[0].aval
        p = eq.params
        d = p['dimension_numbers']
        assert len(operand.shape) == len(result.shape)
        assert indices.shape == (*result.shape, 1)
        assert p['mode'] in (lax.GatherScatterMode.FILL_OR_DROP, lax.GatherScatterMode.PROMISE_IN_BOUNDS)
        assert p['slice_sizes'] == (1,) * len(result.shape)
        assert not d.offset_dims
        assert d.collapsed_slice_dims == d.start_index_map
        assert d.operand_batching_dims == d.start_indices_batching_dims
        assert len(d.collapsed_slice_dims) == 1
        assert len(d.operand_batching_dims) == len(result.shape) - 1


@pytest.mark.parametrize('name', ['hash_goal_120_24', 'dedup_stream4_128'])
def test_no_scalar_uint8_load_or_scatter_after_v2(name):
    case = next(c for c in build_cases(interpret=True) if c['name'] == name)
    graph = jax.make_jaxpr(case['fn'])(*case['args'])
    for eq in equations(graph):
        assert eq.primitive.name != 'scatter'
        if eq.primitive.name == 'get':
            for var in eq.outvars:
                assert not (var.aval.shape == () and str(var.aval.dtype) == 'uint8')


def test_dedup_count_reduction_uses_supported_signed_arithmetic():
    case = next(c for c in build_cases(interpret=True) if c['name'] == 'dedup_stream3_128')
    graph = jax.make_jaxpr(case['fn'])(*case['args'])
    reductions = [eq for eq in equations(graph) if eq.primitive.name == 'reduce_sum']
    assert reductions
    assert all(str(eq.invars[0].aval.dtype) != 'uint32' for eq in reductions)
    scalar_stores = [eq for eq in equations(graph) if eq.primitive.name == 'swap'
                     and any(getattr(v.aval, 'shape', None) == () for v in eq.invars)]
    assert not scalar_stores


@pytest.mark.parametrize('name', ['dedup_stream3_128', 'split_0', 'split_127'])
def test_control_outputs_never_store_scalars_to_vmem(name):
    case = next(c for c in build_cases(interpret=True) if c['name'] == name)
    graph = jax.make_jaxpr(case['fn'])(*case['args'])
    scalar_stores = [eq for eq in equations(graph) if eq.primitive.name == 'swap'
                     and any(getattr(v.aval, 'shape', None) == () for v in eq.invars)]
    assert not scalar_stores


@pytest.mark.parametrize('name', ['dedup_stream3_128', 'split_0', 'split_127'])
def test_control_plane_construction_has_no_boolean_select(name):
    case = next(c for c in build_cases(interpret=True) if c['name'] == name)
    graph = jax.make_jaxpr(case['fn'])(*case['args'])
    control_selects = [eq for eq in equations(graph) if eq.primitive.name == 'select_n'
                       and any(getattr(v.aval, 'shape', None) == (1, 128) for v in eq.outvars)]
    assert not control_selects
