"""Opt-in exact flat lookup candidates, including a banked TensorCore kernel.

Caller guarantees state values are in [0, category_count). Static shape/dtype
contracts are checked here; runtime range validation belongs to input preflight.
The legacy ``pallas_banked`` path prepares LUTs inside every compiled call;
``flat_embedding_prepacked`` exposes model-initialization prepacking explicitly.
"""
from __future__ import annotations

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .tpu_layout import pad_to_multiple


class BankedEmbedding(NamedTuple):
    """Precomputed phase/category banks passed as ordinary runtime arrays."""

    low: jax.Array
    high: jax.Array


def _validate_embedding_table(embedding):
    if (embedding.ndim != 2 or min(embedding.shape) <= 0
            or not jnp.issubdtype(embedding.dtype, jnp.floating)):
        raise ValueError("embedding must be a nonempty floating matrix")
    classes, embed_dim = embedding.shape
    if classes > 256 or embed_dim > 128:
        raise ValueError("banked TPU gather requires classes<=256 and embed_dim<=128")
    return classes, embed_dim


def prepare_banked_embedding(embedding, *, storage_dtype=jnp.float32):
    """Convert one model embedding into reusable Mosaic gather banks.

    The checkpoint table is first rounded to BF16, exactly like Artgor's
    inference.  ``storage_dtype`` changes only the physical bank storage and
    is therefore safe to A/B without changing the logical values.
    """
    classes, embed_dim = _validate_embedding_table(embedding)
    dtype = jnp.dtype(storage_dtype)
    if dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)):
        raise ValueError("bank storage_dtype must be bfloat16 or float32")
    table = jnp.pad(
        embedding.astype(jnp.bfloat16), ((0, 256 - classes), (0, 0))
    ).astype(dtype)
    phases = embed_dim // math.gcd(embed_dim, 128)
    offsets = (
        jnp.arange(phases)[:, None] * 128 + jnp.arange(128)[None, :]
    ) % embed_dim
    # [phase, output lane, category within the selected 128-entry bank].
    return BankedEmbedding(
        low=table[:128, offsets].transpose(1, 2, 0),
        high=table[128:, offsets].transpose(1, 2, 0),
    )


def _flat_kernel(states_ref, low_ref, high_ref, out_ref, *, embed_dim, width):
    tile = pl.program_id(1)
    columns = jnp.arange(128, dtype=jnp.int32)
    positions = ((tile * 128 + columns) // embed_dim) % 128
    # Each state bank is only 128 columns, avoiding a 256-source lane gather.
    position_matrix = jnp.broadcast_to(positions[None, :], states_ref.shape)
    categories = jnp.take_along_axis(states_ref[...], position_matrix,
                                     axis=1, mode="promise_in_bounds")
    indices = (categories % 128).T
    low = jnp.take_along_axis(low_ref[...], indices, axis=1, mode="promise_in_bounds")
    high = jnp.take_along_axis(high_ref[...], indices, axis=1, mode="promise_in_bounds")
    values = jnp.where(categories.T >= 128, high, low).T
    valid = jax.lax.broadcasted_iota(jnp.int32, values.shape, 1) + tile * 128 < width
    out_ref[...] = jnp.where(valid, values, 0.).astype(jnp.bfloat16)


def _bank_lut_index(row_block, tile, *, phases):
    """Return Mosaic LUT indices without x64-promoted Python zero literals."""
    del row_block
    zero = jnp.int32(0)
    return tile % jnp.int32(phases), zero, zero


def flat_embedding_prepacked(states, banks, *, embed_dim, bm=128, interpret=False):
    """Exact position-major lookup using banks prepared once per checkpoint."""
    if states.ndim != 2 or min(states.shape) <= 0 or states.dtype != jnp.uint8:
        raise ValueError("states must be a nonempty uint8 matrix")
    if (not isinstance(embed_dim, int) or isinstance(embed_dim, bool)
            or embed_dim <= 0 or embed_dim > 128):
        raise ValueError("embed_dim must be an integer in [1,128]")
    if not isinstance(bm, int) or isinstance(bm, bool) or bm <= 0:
        raise ValueError("bm must be a positive integer")
    if not isinstance(banks, BankedEmbedding):
        raise TypeError("banks must be a BankedEmbedding")
    phases = embed_dim // math.gcd(embed_dim, 128)
    expected = (phases, 128, 128)
    if banks.low.shape != expected or banks.high.shape != expected:
        raise ValueError(f"bank shapes must both be {expected}")
    if (banks.low.dtype != banks.high.dtype
            or banks.low.dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32))):
        raise ValueError("banks must have one shared bfloat16 or float32 dtype")
    if not interpret and bm % 8:
        raise ValueError("banked TPU gather requires BM divisible by8")

    rows, length = states.shape
    width = length * embed_dim
    padded_rows = pad_to_multiple(rows, bm)
    padded_width = pad_to_multiple(width, 128)
    padded_states = jnp.pad(
        states.astype(jnp.int32),
        ((0, padded_rows - rows), (0, pad_to_multiple(length, 128) - length)),
    )
    lut_spec = pl.BlockSpec(
        (None, 128, 128), functools.partial(_bank_lut_index, phases=phases)
    )
    call = pl.pallas_call(
        functools.partial(_flat_kernel, embed_dim=embed_dim, width=width),
        grid=(padded_rows // bm, padded_width // 128),
        in_specs=[
            pl.BlockSpec((bm, 128), lambda i, j: (i, j // embed_dim)),
            lut_spec,
            lut_spec,
        ],
        out_specs=pl.BlockSpec((bm, 128), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct(
            (padded_rows, padded_width), jnp.bfloat16
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        interpret=interpret,
        name="stream1_flat_embedding_banked",
    )
    return call(padded_states, banks.low, banks.high)[:rows, :width]


def flat_embedding(states, embedding, *, implementation, bm=128, interpret=False):
    """Position-major BF16 lookup; no embedding/Dense contraction or new weights."""
    if states.ndim != 2 or min(states.shape) <= 0 or states.dtype != jnp.uint8:
        raise ValueError("states must be a nonempty uint8 matrix")
    if (embedding.ndim != 2 or min(embedding.shape) <= 0
            or not jnp.issubdtype(embedding.dtype, jnp.floating)):
        raise ValueError("embedding must be a nonempty floating matrix")
    if not isinstance(bm, int) or isinstance(bm, bool) or bm <= 0:
        raise ValueError("bm must be a positive integer")
    if implementation not in ("jax_flat", "jax_tiled", "pallas_banked"):
        raise ValueError("unknown embedding implementation")
    rows, length = states.shape
    classes, embed_dim = embedding.shape
    table = embedding.astype(jnp.bfloat16)
    width = length * embed_dim
    if implementation == "jax_flat":
        features = jnp.arange(width, dtype=jnp.int32)
        categories = states[:, features // embed_dim].astype(jnp.int32)
        return table.reshape(-1)[categories * embed_dim + features % embed_dim]
    padded_rows = pad_to_multiple(rows, bm)
    if implementation == "jax_tiled":
        blocks = jnp.pad(states, ((0, padded_rows - rows), (0, 0))).reshape(-1, bm, length)
        result = jax.lax.map(lambda block: table[block.astype(jnp.int32)].reshape(bm, width), blocks)
        return result.reshape(padded_rows, width)[:rows]
    banks = prepare_banked_embedding(embedding, storage_dtype=jnp.float32)
    return flat_embedding_prepacked(
        states, banks, embed_dim=embed_dim, bm=bm, interpret=interpret
    )
