import importlib
import jax
import jax.numpy as jnp


def test_local_adapter_preserves_aligned_control_shape():
    import importlib.util
    assert importlib.util.find_spec('benchmarks.beam_external_split_probe') is not None
    module = importlib.import_module('benchmarks.beam_external_split_probe')
    shapes = ((1,8,256), (1,1,256), (1,1,128))
    result = jax.eval_shape(lambda *xs: module.local_split(*xs, local_rank=3),
                            *(jax.ShapeDtypeStruct(s,jnp.uint32) for s in shapes))
    assert tuple(x.shape for x in result) == ((1,8,256), (1,8,256),
                                             (1,1,128), (1,1,128), (1,1,128))
