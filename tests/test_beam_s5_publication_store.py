from types import SimpleNamespace
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def test_publication_dependency_has_vector_store(monkeypatch):
    from tpu_beam_search.beam_s5_epoch import make_s5_epoch_call
    original = pl.pallas_call
    captured = []
    def capture(*args, **kwargs):
        call = original(*args, **kwargs)
        if kwargs.get('name') == 'beam_s5_publication_dependency':
            captured.append(call)
        return call
    monkeypatch.setattr(pl, 'pallas_call', capture)
    make_s5_epoch_call(SimpleNamespace(size=1), bins=128, period=3)
    assert len(captured) == 1
    traced = jax.make_jaxpr(captured[0])(jax.ShapeDtypeStruct((1,128),jnp.uint32))
    stores = []
    def walk(value):
        if isinstance(value, dict):
            for child in value.values(): walk(child)
        elif isinstance(value, (tuple,list)):
            for child in value: walk(child)
        elif hasattr(value, 'jaxpr'): walk(value.jaxpr)
        elif hasattr(value, 'eqns'):
            for eqn in value.eqns:
                if eqn.primitive.name == 'swap':
                    stores.append(eqn.invars[1].aval.shape)
                walk(eqn.params)
    walk(traced)
    assert stores
    assert all(shape != () for shape in stores), stores
