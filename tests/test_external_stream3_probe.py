import importlib.util
import jax
import jax.numpy as jnp


def test_composed_adapter_removes_both_control_axes():
    assert importlib.util.find_spec('benchmarks.beam_external_stream3_probe') is not None
    from benchmarks.beam_external_stream3_probe import local_stream3
    shapes = ((1,8,256),(1,1,256),(1,1,1),(1,1,1))
    result = jax.eval_shape(lambda *a: local_stream3(*a,local_rank=3),
        *(jax.ShapeDtypeStruct(s,jnp.uint32) for s in shapes))
    assert tuple(x.shape for x in result) == ((1,8,256),(1,8,256),
        (1,1,128),(1,1,128),(1,1,128))
