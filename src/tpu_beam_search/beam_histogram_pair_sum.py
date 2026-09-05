"""Exact sum of received uint32-pair histograms; transport is separate."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_sum_histogram_pairs(pairs,*,interpret=False):
    """Rows are rank0 low/high, rank1 low/high, etc. Total must fit uint64.

    Zero padding is a caller contract. Each rank's high plane contributes to
    the result in addition to carries from adding the low planes.
    """
    if (pairs.ndim != 2 or pairs.shape[0]%2 or not 2 <= pairs.shape[0] <= 256
            or not pairs.shape[1] or pairs.shape[1]%128 or pairs.dtype != jnp.uint32):
        raise ValueError('histogram pairs require uint32 [2*ranks,128-aligned bins]')
    planes,width = pairs.shape
    def kernel(source,out):
        low = jnp.zeros((128,),jnp.uint32)
        high = jnp.zeros((128,),jnp.uint32)
        for rank in range(planes//2):
            next_low = low+source[2*rank]
            high = high+source[2*rank+1]+(next_low < low).astype(jnp.uint32)
            low = next_low
        out[...] = jnp.stack((low,high))
    return pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((2,width),jnp.uint32),
        in_specs=(pl.BlockSpec((planes,128),lambda i:(0,i)),),
        out_specs=pl.BlockSpec((2,128),lambda i:(0,i)),grid=(width//128,),
        interpret=interpret,name='beam_global_histogram_pair_sum')(pairs)
