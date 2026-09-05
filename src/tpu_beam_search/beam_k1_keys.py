"""K1 fingerprint and two source bucket keys using uint32-pair arithmetic."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_hash import _distribution


def pallas_k1_keys(words,*,bucket_count,interpret=False):
    if (words.ndim != 2 or words.shape[0] != 4 or words.dtype != jnp.uint32
            or not words.shape[1] or words.shape[1]%128
            or not isinstance(bucket_count,int) or not 1 <= bucket_count <= 2**30
            or bucket_count&(bucket_count-1)):
        raise ValueError('K1 requires aligned Hash128 columns and power-of-two buckets')
    count = words.shape[1]
    def kernel(h,out):
        values = h[...]
        low,high = _distribution(values,0xa4093822299f31d0)
        fingerprint = low^high
        fingerprint = jnp.where(fingerprint == 0,jnp.uint32(1),fingerprint)
        mask = jnp.uint32(bucket_count-1)
        b0 = _distribution(values,0x082efa98ec4e6c89)[0]&mask
        b1 = _distribution(values,0x452821e638d01377)[0]&mask
        out[...] = jnp.stack((fingerprint,b0,b1))
    return pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((3,count),jnp.uint32),
        in_specs=(pl.BlockSpec((4,128),lambda i:(0,i)),),
        out_specs=pl.BlockSpec((3,128),lambda i:(0,i)),grid=(count//128,),
        interpret=interpret,name='beam_k1_fingerprint_buckets')(words)
