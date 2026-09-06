"""Lossless little-endian response bytes <-> uint32 SoA transport adapter.

Routing, validity and record counts stay separate. This diagnostic adapter
preserves padding too; it does not authorize invalid rows for transmission.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_wire_to_planes(wire, *, interpret=False):
    if (wire.ndim != 2 or wire.dtype != jnp.uint8 or not wire.shape[0]
            or wire.shape[0] % 128 or not wire.shape[1] or wire.shape[1] % 128):
        raise ValueError('expected aligned uint8 response rows')
    n, width = wire.shape

    def kernel(src, dst):
        for word in range(width // 4):
            value = jnp.zeros((128,), jnp.uint32)
            for byte in range(4):
                value |= src[:, word * 4 + byte].astype(jnp.uint32) << jnp.uint32(byte * 8)
            dst[word, :] = value

    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((width // 4, n), jnp.uint32),
        in_specs=(pl.BlockSpec((128, width), lambda i: (i, 0)),),
        out_specs=pl.BlockSpec((width // 4, 128), lambda i: (0, i)),
        grid=(n // 128,), interpret=interpret, name='beam_final_wire_to_planes')(wire)


def pallas_planes_to_wire(planes, *, interpret=False):
    if (planes.ndim != 2 or planes.dtype != jnp.uint32 or not planes.shape[0]
            or planes.shape[0] % 32 or not planes.shape[1] or planes.shape[1] % 128):
        raise ValueError('expected aligned uint32 response planes')
    words, n = planes.shape

    def kernel(src, dst):
        for word in range(words):
            for byte in range(4):
                dst[:, word * 4 + byte] = ((src[word, :] >> jnp.uint32(byte * 8))
                                          & jnp.uint32(255)).astype(jnp.uint8)

    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((n, words * 4), jnp.uint8),
        in_specs=(pl.BlockSpec((words, 128), lambda i: (0, i)),),
        out_specs=pl.BlockSpec((128, words * 4), lambda i: (i, 0)),
        grid=(n // 128,), interpret=interpret, name='beam_final_planes_to_wire')(planes)
