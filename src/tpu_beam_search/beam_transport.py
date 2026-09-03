"""Local DMA-pipelined metadata packing; not a remote exchange scheduler."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def pallas_pack_candidates(hash_words, parent_words, scores, routes, *,
                           tile_candidates=128, buffer_count=2,
                           pipelined=True, interpret=False):
    fields = (hash_words, parent_words, scores, routes)
    if any(x.ndim != 2 or x.dtype != jnp.uint32 for x in fields):
        raise ValueError('all fields must be rank-two uint32 arrays')
    capacity = hash_words.shape[1]
    if any(x.shape != (rows, capacity) for x, rows in zip(fields, (4, 2, 1, 1))):
        raise ValueError('field shapes must be [4,N], [2,N], [1,N], [1,N]')
    if (not isinstance(tile_candidates, int) or tile_candidates <= 0
            or tile_candidates % 128 or capacity <= 0 or capacity % tile_candidates):
        raise ValueError('tile must be a positive multiple of 128 dividing capacity')
    if buffer_count not in (2, 3):
        raise ValueError('buffer_count must be 2 or 3')

    def spec(rows, buffers):
        return pl.BlockSpec((rows, tile_candidates), lambda i: (0, i),
                            pipeline_mode=pl.Buffered(buffer_count=buffers))

    def pack(h, p, s, r, out):
        out[...] = jnp.concatenate((h[...], p[...], s[...], r[...]), axis=0)

    def kernel(h, p, s, r, out):
        pltpu.emit_pipeline(pack, grid=(capacity // tile_candidates,),
                            in_specs=tuple(spec(rows, buffer_count) for rows in (4, 2, 1, 1)),
                            out_specs=spec(8, 2),
                            no_pipelining=not pipelined)(h, p, s, r, out)

    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((8, capacity), jnp.uint32),
        in_specs=tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in fields),
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        interpret=interpret, name='beam_pack_candidate_soa_pipeline',
    )(*fields)
