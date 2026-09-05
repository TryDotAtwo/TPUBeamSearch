import importlib.util
import jax
import jax.numpy as jnp


def test_full_collector_adapter_retains_rank_and_all_four_results():
    assert importlib.util.find_spec('benchmarks.beam_collector_full_probe') is not None
    from benchmarks.beam_collector_full_probe import local_collect
    shapes = ((1,3,8,128),(1,3,8,128),(1,8,256),(1,3,8,128),(1,1,128))
    actual = jax.eval_shape(local_collect,*(jax.ShapeDtypeStruct(s,jnp.uint32) for s in shapes))
    assert tuple(x.shape for x in actual) == (
        (1,3,8,128),(1,3,8,128),(1,3,8,128),(1,1,128))
