import jax
import jax.numpy as jnp
from types import SimpleNamespace
from tpu_beam_search import beam_remote_exchange as module


def test_stream3_exchange_collector_preserves_all_resident_outputs():
    assert hasattr(module,'make_stream3_collect_call')
    fn = module.make_stream3_collect_call(SimpleNamespace(size=8))
    shapes = ((3,8,512),(3,8,512),(3,8,128),(8,128),(1,128),(1,),(1,),(8,128))
    traced = jax.make_jaxpr(fn,axis_env=[('core',8)])(
        *(jax.ShapeDtypeStruct(s,jnp.uint32) for s in shapes))
    assert tuple(x.shape for x in traced.out_avals) == (
        (3,8,512),(3,8,512),(3,8,128),(1,128))
