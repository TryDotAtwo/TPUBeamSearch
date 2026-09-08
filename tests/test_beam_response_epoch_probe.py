from types import SimpleNamespace
import jax
import jax.numpy as jnp


def test_probe_preparation_and_epoch_have_eight_rank_local_abi():
    from benchmarks.beam_response_epoch_probe import local_calls
    prepare,epoch=local_calls(SimpleNamespace(size=8))
    shaped=lambda shape:jax.ShapeDtypeStruct(shape,jnp.uint32)
    prep=jax.make_jaxpr(prepare)(jax.ShapeDtypeStruct((1,2048,128),jnp.uint8),
        shaped((1,4,2048)),shaped((1,2,128)),shaped((1,2,128)))
    assert [a.shape for a in prep.out_avals]==[(1,35,2048),(1,3,128),(1,1,128)]
    traced=jax.make_jaxpr(epoch,axis_env=[('core',8)])(
        *prep.out_avals,shaped((1,)))
    assert [a.shape for a in traced.out_avals]==[(1,1024,128),(1,2,128)]
