"""Immediate-child Stream2 primitive (K1=K2=0), not the solved collector.

The caller prepares valid permutations/state values and a count <= parent
capacity. Zobrist padding rows must be zero for the production contract.
Only hashes and flags leave the kernel; no child-state array is materialized.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .tpu_layout import pad_to_multiple


def _take_clipped(vector, indices):
    # Mosaic rejects GatherScatterMode.CLIP; make the bound explicit first.
    indices = jnp.clip(indices, 0, vector.shape[0] - 1)
    return jnp.take_along_axis(vector, indices, axis=0, mode='promise_in_bounds')


def pallas_hash_goal(parents, generators, central, zobrist_words, count, *,
                     tile_candidates=128, interpret=False):
    if parents.ndim != 2 or parents.dtype != jnp.uint8 or min(parents.shape) <= 0:
        raise ValueError('parents must be nonempty uint8 [B,S]')
    batch, width = parents.shape
    if (generators.ndim != 2 or generators.shape[1] != width
            or generators.shape[0] <= 0 or generators.dtype != jnp.int32):
        raise ValueError('generators must be int32 [MOVE_COUNT,S]')
    if central.shape != (width,) or central.dtype != jnp.uint8:
        raise ValueError('central must be uint8 [S]')
    if (zobrist_words.ndim != 2 or zobrist_words.shape[0] != 4
            or zobrist_words.shape[1] % width or zobrist_words.shape[1] < width
            or zobrist_words.dtype != jnp.uint32):
        raise ValueError('zobrist_words must be uint32 [4,S*NUM_CLASSES]')
    if count.shape != (1,) or count.dtype != jnp.uint32:
        raise ValueError('count must be uint32 [1]')
    if not isinstance(tile_candidates, int) or tile_candidates <= 0 or tile_candidates % 128:
        raise ValueError('tile_candidates must be a positive multiple of 128')
    moves = generators.shape[0]
    classes = zobrist_words.shape[1] // width
    capacity = pad_to_multiple(batch * moves, tile_candidates)

    def kernel(pr, gr, cr, zr, nr, hashes, goals, validity):
        lane = pl.program_id(0) * tile_candidates + jnp.arange(tile_candidates)
        parent = lane // moves
        move = lane % moves
        valid = (parent < batch) & (parent.astype(jnp.uint32) < nr[0])
        safe_parent = jnp.minimum(parent, batch - 1)
        parent_values = pr[...].reshape(-1)
        permutation = gr[...].reshape(-1)
        table = zr[...]

        def position(p, acc):
            h0, h1, h2, h3, goal = acc
            source = _take_clipped(permutation, move * width + p)
            value = _take_clipped(parent_values, safe_parent * width + source)
            address = p * classes + value.astype(jnp.int32)
            return (h0 ^ _take_clipped(table[0], address),
                    h1 ^ _take_clipped(table[1], address),
                    h2 ^ _take_clipped(table[2], address),
                    h3 ^ _take_clipped(table[3], address),
                    goal & (value == cr[p]))

        zero = jnp.zeros((tile_candidates,), jnp.uint32)
        h0, h1, h2, h3, goal = jax.lax.fori_loop(
            0, width, position, (zero, zero, zero, zero, valid))
        hashes[...] = jnp.where(valid[None, :], jnp.stack((h0, h1, h2, h3)), 0)
        goals[...] = goal[None, :].astype(jnp.uint32)
        validity[...] = valid[None, :].astype(jnp.uint32)

    return pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((4, capacity), jnp.uint32),
                   jax.ShapeDtypeStruct((1, capacity), jnp.uint32),
                   jax.ShapeDtypeStruct((1, capacity), jnp.uint32)),
        in_specs=tuple(pl.BlockSpec(x.shape)
                       for x in (parents, generators, central, zobrist_words, count)),
        out_specs=tuple(pl.BlockSpec((rows, tile_candidates), lambda i: (0, i))
                        for rows in (4, 1, 1)),
        grid=(capacity // tile_candidates,), interpret=interpret,
        name='beam_stream2_immediate_hash_goal',
    )(parents, generators, central, zobrist_words, count)
