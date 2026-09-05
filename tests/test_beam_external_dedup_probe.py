import jax
import jax.numpy as jnp
from benchmarks.beam_external_dedup_probe import local_dedup


def test_output_blocks_obey_tpu_trailing_dimension_contract(monkeypatch):
    from jax.experimental import pallas as pl
    original = pl.pallas_call

    def checked_call(*args, **kwargs):
        shapes = kwargs['out_shape']
        specs = kwargs['out_specs']
        if not isinstance(shapes, tuple):
            shapes, specs = (shapes,), (specs,)
        for shape, spec in zip(shapes, specs):
            if len(shape.shape) >= 2:
                for block, extent, alignment in zip(
                        spec.block_shape[-2:], shape.shape[-2:], (8, 128)):
                    assert block == extent or block % alignment == 0, (
                        kwargs.get('name'), spec.block_shape, shape.shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(pl, 'pallas_call', checked_call)
    inputs = [jax.ShapeDtypeStruct(s, jnp.uint32)
              for s in ((1, 8, 256), (1, 1, 256), (1, 1, 1), (1, 1, 1))]
    jax.eval_shape(local_dedup, *inputs)


def test_shard_adapter_accepts_local_rank_three_controls():
    inputs = [jax.ShapeDtypeStruct(s, jnp.uint32)
              for s in ((1, 8, 256), (1, 1, 256), (1, 1, 1), (1, 1, 1))]
    out, count = jax.eval_shape(local_dedup, *inputs)
    assert out.shape == (1, 8, 256)
    assert count.shape == (1, 1, 128)
