import importlib.util
import jax
import jax.numpy as jnp


def test_collector_shard_adapter_returns_buffers_and_aligned_control():
    assert importlib.util.find_spec('benchmarks.beam_collector_probe') is not None
    from benchmarks.beam_collector_probe import local_append
    shapes = ((1,8,256),(1,8,256),(1,8,128),(1,8,128),(1,1,128))
    result = jax.eval_shape(local_append,*(jax.ShapeDtypeStruct(s,jnp.uint32) for s in shapes))
    assert tuple(x.shape for x in result) == ((1,8,256),(1,8,256),(1,8,128))


def test_group_adapter_preserves_multitile_capacity():
    from benchmarks import beam_collector_probe as module
    assert hasattr(module,'local_append_group')
    shapes = ((1,8,512),(1,8,512),(1,8,512),(1,8,128),(1,1,128))
    result = jax.eval_shape(module.local_append_group,
        *(jax.ShapeDtypeStruct(s,jnp.uint32) for s in shapes))
    assert tuple(x.shape for x in result) == ((1,8,512),(1,8,512),(1,8,128))
