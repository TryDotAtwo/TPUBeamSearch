"""Original Hash128 routing with explicit modulo-2**64 uint32 pair arithmetic.

Constants and mixing order: D:/100XH100/src/hash.hpp. No lossy conversion,
native uint64 TPU operation, or owner assignment before dedup is required.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _xor(a, b):
    return a[0] ^ b[0], a[1] ^ b[1]


def _shr(a, shift):
    return (a[0] >> shift) | (a[1] << (32 - shift)), a[1] >> shift


def _constant(value):
    return jnp.uint32(value & 0xffffffff), jnp.uint32(value >> 32)


def _mul_constant(a, value):
    # Four 16-bit products recover the high word without mulhi/uint64.
    b0, b1 = _constant(value)
    x0, x1 = a[0] & jnp.uint32(65535), a[0] >> 16
    y0, y1 = b0 & jnp.uint32(65535), b0 >> 16
    p0 = x0 * y0
    t = x1 * y0 + (p0 >> 16)
    t2 = x0 * y1 + (t & jnp.uint32(65535))
    low = (t2 << 16) | (p0 & jnp.uint32(65535))
    high = x1 * y1 + (t >> 16) + (t2 >> 16) + a[0] * b1 + a[1] * b0
    return low, high


def _mix(a):
    a = _mul_constant(_xor(a, _shr(a, 30)), 0xbf58476d1ce4e5b9)
    a = _mul_constant(_xor(a, _shr(a, 27)), 0x94d049bb133111eb)
    return _xor(a, _shr(a, 31))


def _distribution(words, salt):
    lo = (words[0], words[1])
    hi = (words[2], words[3])
    c = _constant(0x9e3779b97f4a7c15)
    added_lo = hi[0] + c[0]
    added = added_lo, hi[1] + c[1] + (added_lo < hi[0]).astype(jnp.uint32)
    return _mix(_xor(_xor(_xor(lo, (hi[1], hi[0])), _constant(salt)), _mix(added)))


def _mod(a, modulus):
    if modulus & (modulus - 1) == 0:
        return a[0] & jnp.uint32(modulus - 1)
    m = jnp.uint32(modulus)
    def bit_step(i, rem):
        bit = jnp.where(i < 32, (a[1] >> (31 - i)) & 1,
                        (a[0] >> (63 - i)) & 1)
        doubled = (rem << 1) | bit
        return jnp.where(((rem >> 31) != 0) | (doubled >= m), doubled - m, doubled)
    return jax.lax.fori_loop(0, 64, bit_step, jnp.zeros_like(a[0]))


def pallas_route_hashes(words, *, world_size, shard_count, tile_candidates=128, interpret=False):
    if words.ndim != 2 or words.shape[0] != 4 or words.dtype != jnp.uint32:
        raise ValueError('hash words must be uint32 [4,N]')
    if not isinstance(world_size, int) or not 1 <= world_size <= 256:
        raise ValueError('world_size must be in [1,256]')
    if not isinstance(shard_count, int) or not 1 <= shard_count <= 0xffffffff:
        raise ValueError('shard_count must fit a positive uint32')
    n = words.shape[1]
    if (not isinstance(tile_candidates, int) or tile_candidates <= 0
            or tile_candidates % 128 or n <= 0 or n % tile_candidates):
        raise ValueError('tile must be a positive multiple of 128 dividing N')
    def kernel(h, out):
        values = h[...]
        owner = _mod(_distribution(values, 0x243f6a8885a308d3), world_size)
        shard = _mod(_distribution(values, 0x13198a2e03707344), shard_count)
        out[...] = jnp.stack((owner, shard))
    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((2, n), jnp.uint32),
        in_specs=(pl.BlockSpec((4, tile_candidates), lambda i: (0, i)),),
        out_specs=pl.BlockSpec((2, tile_candidates), lambda i: (0, i)),
        grid=(n // tile_candidates,), interpret=interpret,
        name='beam_hash128_owner_shard_u32',
    )(words)
