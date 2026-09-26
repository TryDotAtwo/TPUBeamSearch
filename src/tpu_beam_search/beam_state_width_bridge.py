"""Pallas-only conversion between compact HBM rows and final TPU tiles.

This is a layout bridge, not a final-stage caller or a physical TPU gate.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _check(rows, state_len, target_width, *, expand):
    if (rows.ndim != 2 or rows.dtype != jnp.uint8 or rows.shape[0] == 0
            or rows.shape[0] % 128 or not isinstance(state_len, int)
            or not 0 < state_len <= rows.shape[1] - 4
            or not isinstance(target_width, int) or target_width < state_len + 4
            or (target_width % 128 if expand else target_width % 16)
            or (target_width < rows.shape[1] if expand else target_width > rows.shape[1])):
        raise ValueError("invalid compact/kernel state width bridge ABI")


def pallas_expand_state_rows(rows, *, state_len, kernel_width, interpret=False):
    """Copy logical bytes and response index into a zero-tailed TPU tile."""
    _check(rows, state_len, kernel_width, expand=True)
    source_width = rows.shape[1]

    def kernel(source, output):
        valid = jnp.where(jnp.arange(source_width)[None, :] < state_len + 4,
                          source[...], jnp.uint8(0))
        output[...] = jnp.concatenate(
            (valid, jnp.zeros((128, kernel_width - source_width), jnp.uint8)), axis=1,
        )

    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((rows.shape[0], kernel_width), jnp.uint8),
        in_specs=(pl.BlockSpec((128, source_width), lambda i: (i, 0)),),
        out_specs=pl.BlockSpec((128, kernel_width), lambda i: (i, 0)),
        grid=(rows.shape[0] // 128,), interpret=interpret,
        name="beam_state_compact_to_tpu_tile",
    )(rows)


def pallas_compact_state_rows(rows, *, state_len, storage_width, interpret=False):
    """Discard only tile padding; clear persistent padding after the index."""
    _check(rows, state_len, storage_width, expand=False)
    kernel_width = rows.shape[1]

    def kernel(source, output):
        positions = jnp.arange(storage_width)[None, :]
        output[...] = jnp.where(positions < state_len + 4,
                                source[:, :storage_width], jnp.uint8(0))

    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((rows.shape[0], storage_width), jnp.uint8),
        in_specs=(pl.BlockSpec((128, kernel_width), lambda i: (i, 0)),),
        out_specs=pl.BlockSpec((128, storage_width), lambda i: (i, 0)),
        grid=(rows.shape[0] // 128,), interpret=interpret,
        name="beam_state_tpu_tile_to_compact",
    )(rows)


def pallas_expand_generator_rows(generators, *, state_len, kernel_width,
                                 interpret=False):
    """Pad a compact move table for the materializer without changing moves."""
    if (generators.ndim != 2 or generators.dtype != jnp.int32
            or not 0 < generators.shape[0] <= 256
            or not isinstance(state_len, int)
            or not 0 < state_len <= generators.shape[1]
            or not isinstance(kernel_width, int) or kernel_width % 128
            or kernel_width < generators.shape[1]):
        raise ValueError("invalid compact generator width bridge ABI")
    moves, source_width = generators.shape

    def kernel(source, output):
        valid = jnp.where(jnp.arange(source_width)[None, :] < state_len,
                          source[...], jnp.int32(0))
        output[...] = jnp.concatenate(
            (valid, jnp.zeros((moves, kernel_width - source_width), jnp.int32)),
            axis=1,
        )

    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((moves, kernel_width), jnp.int32),
        in_specs=(pl.BlockSpec((moves, source_width), lambda i: (0, 0)),),
        out_specs=pl.BlockSpec((moves, kernel_width), lambda i: (0, 0)),
        grid=(1,), interpret=interpret, name="beam_generator_compact_to_tpu_tile",
    )(generators)
