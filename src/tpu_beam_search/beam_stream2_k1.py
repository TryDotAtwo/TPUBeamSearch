"""Immediate-child hashing followed by bounded K1 membership lookup."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_stream2 import pallas_hash_goal
from .beam_k1_lookup import pallas_k1_contains


def pallas_hash_k1_goal(parents, generators, central, zobrist_words, count,
                        table, *, bucket_count, tile_candidates=128,
                        interpret=False):
    """Return immediate hashes, K1 hits and validity, without K2 projection.

    Caller supplies an enabled K1 table including the central state and a
    parent count within capacity. This diagnostic composition does not collect
    solved records or change the default Stream2 implementation.
    """
    move_count = generators.shape[0]
    if parents.shape[0] * move_count > 0xffffffff:
        raise ValueError('child count exceeds uint32 capacity')
    hashes, _, valid = pallas_hash_goal(
        parents, generators, central, zobrist_words, count,
        tile_candidates=tile_candidates, interpret=interpret)

    def multiply_count(n, out):
        out[...] = n[...] * jnp.uint32(move_count)

    child_count = pl.pallas_call(
        multiply_count, out_shape=jax.ShapeDtypeStruct((1,), jnp.uint32),
        interpret=interpret, name='beam_k1_child_count')(count)
    hits = pallas_k1_contains(hashes, table, child_count,
                             bucket_count=bucket_count, interpret=interpret)
    return hashes, hits, valid
