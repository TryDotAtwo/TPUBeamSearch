import importlib.util
from types import SimpleNamespace
import jax
import jax.numpy as jnp


def test_integrated_probe_adapter_preserves_rank_axis_on_all_results():
    assert importlib.util.find_spec('benchmarks.beam_stream3_collector_probe') is not None
    from benchmarks.beam_stream3_collector_probe import make_local_program
    fn = make_local_program(SimpleNamespace(size=8))
    shapes = ((1,3,8,256),(1,3,8,256),(1,3,8,128),(1,8,128),
              (1,1,128),(1,1),(1,1),(1,8,128))
    traced = jax.make_jaxpr(fn,axis_env=[('core',8)])(
        *(jax.ShapeDtypeStruct(s,jnp.uint32) for s in shapes))
    assert tuple(x.shape for x in traced.out_avals) == (
        (1,3,8,256),(1,3,8,256),(1,3,8,128),(1,1,128))
