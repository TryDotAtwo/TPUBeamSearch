import importlib.util
from types import SimpleNamespace
import jax
import jax.numpy as jnp


def test_production_exchange_preserves_snapshot_and_count_abi():
    assert importlib.util.find_spec('tpu_beam_search.beam_remote_exchange') is not None
    from tpu_beam_search.beam_remote_exchange import make_variable_exchange_call
    call = make_variable_exchange_call(SimpleNamespace(size=8),capacity=128)
    shapes = ((7,128),(7,8,128),(7,128),(8,128))
    traced = jax.make_jaxpr(call,axis_env=[('core',8)])(
        *(jax.ShapeDtypeStruct(s,jnp.uint32) for s in shapes))
    assert tuple(x.shape for x in traced.out_avals) == ((9,8,128),(7,128))
