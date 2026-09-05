import jax
import jax.numpy as jnp
from benchmarks.beam_external_dedup_probe import local_dedup


def test_shard_adapter_accepts_local_rank_three_controls():
    inputs = [jax.ShapeDtypeStruct(s, jnp.uint32)
              for s in ((1, 8, 256), (1, 1, 256), (1, 1, 1), (1, 1, 1))]
    out, count = jax.eval_shape(local_dedup, *inputs)
    assert out.shape == (1, 8, 256)
    assert count.shape == (1, 1, 128)
