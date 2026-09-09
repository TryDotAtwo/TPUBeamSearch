"""Regression for V1 Mosaic unsigned-sum rejection; not physical TPU proof."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('stage',['intervals','receive'])
def test_sum_operands_are_signed_int32_for_mosaic(stage):
    from tpu_beam_search.beam_final_intervals import pallas_final_rank_intervals
    with jax.enable_x64(False):
        if stage == 'intervals':
            traced = jax.make_jaxpr(lambda r,v: pallas_final_rank_intervals(
                r,v,world_size=8,interpret=True))(
                    jnp.zeros((1,256),jnp.uint32),jnp.ones((1,256),jnp.uint32))
        else:
            from tpu_beam_search.beam_final_receive import pallas_compact_final_received
            traced = jax.make_jaxpr(lambda x,c,e: pallas_compact_final_received(
                x,c,e,interpret=True))(jnp.zeros((2,4,128),jnp.uint32),
                    jnp.zeros((2,1,128),jnp.uint32),jnp.zeros((1,128),jnp.uint32))
    sums = []
    def visit(obj):
        if hasattr(obj,'jaxpr'):
            visit(obj.jaxpr)
        elif hasattr(obj,'eqns'):
            for eq in obj.eqns:
                if eq.primitive.name == 'reduce_sum':
                    sums.append(eq.invars[0].aval.dtype)
                for value in eq.params.values():
                    visit(value)
        elif isinstance(obj,(tuple,list)):
            for value in obj:
                visit(value)
    visit(traced)
    assert sums, 'expected actual Pallas count reductions'
    assert all(dtype == np.dtype('int32') for dtype in sums), sums


def test_interval_prefix_has_no_zero_sized_vector_intermediates():
    """V2 rejects a size0 slice from associative_scan during Mosaic lowering."""
    from tpu_beam_search.beam_final_intervals import pallas_final_rank_intervals
    with jax.enable_x64(False):
        traced = jax.make_jaxpr(lambda r,v: pallas_final_rank_intervals(
            r,v,world_size=8,interpret=True))(
                jnp.zeros((1,256),jnp.uint32),jnp.ones((1,256),jnp.uint32))
    empty = []
    def visit(obj):
        if hasattr(obj,'jaxpr'):
            visit(obj.jaxpr)
        elif hasattr(obj,'eqns'):
            for eq in obj.eqns:
                for var in eq.outvars:
                    if 0 in getattr(var.aval,'shape',()):
                        empty.append((eq.primitive.name,var.aval.shape))
                for value in eq.params.values():
                    visit(value)
        elif isinstance(obj,(tuple,list)):
            for value in obj:
                visit(value)
    visit(traced)
    assert not empty, empty
