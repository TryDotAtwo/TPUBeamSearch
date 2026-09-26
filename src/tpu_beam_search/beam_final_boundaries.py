"""Exact ceil(rank*K/world) using uint32 pair arithmetic only."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_final_boundaries(keep,prior_error,*,world_size,interpret=False):
    """Return pair-word boundaries, uint32 target counts and error.

    Boundary slots 0..world inclusive occupy [2,128]; counts occupy [1,128].
    Reject max target count >UINT32_MAX (the materialization index ABI), rather
    than truncating it. Invalid results are zero and must not be consumed.
    Caller supplies the same keep/error on every rank after final selection.
    """
    if (type(world_size) is not int or not 1 <= world_size <= 127
            or keep.shape != (2,128) or prior_error.shape != (1,128)
            or keep.dtype != jnp.uint32 or prior_error.dtype != jnp.uint32):
        raise ValueError('invalid final boundary ABI')
    def kernel(k,e,bounds,counts,error):
        divisor = jnp.uint32(world_size)
        qlo,qhi,rem = (jnp.uint32(0),)*3
        # Restoring division: remainder always <world<=127, so doubling it
        # fits uint32 even for the largest input pair. No float or x64 path.
        for bit in range(63,-1,-1):
            value = k[1,0] if bit >= 32 else k[0,0]
            rem = (rem<<jnp.uint32(1)) | ((value>>jnp.uint32(bit%32)) & jnp.uint32(1))
            take = rem >= divisor
            rem = jnp.where(take,rem-divisor,rem)
            mask = take.astype(jnp.uint32)<<jnp.uint32(bit%32)
            if bit >= 32:
                qhi |= mask
            else:
                qlo |= mask
        bad = ((e[0,0] != 0) | (qhi != 0)
               | ((qlo == jnp.uint32(0xffffffff)) & (rem != 0)))
        ranks = jnp.arange(128,dtype=jnp.uint32)
        # qlo*ranks via two 16-bit products, each fits uint32 at r<=127.
        low_product = (qlo & jnp.uint32(65535))*ranks
        high_product = (qlo>>jnp.uint32(16))*ranks
        shifted = high_product<<jnp.uint32(16)
        low = low_product+shifted
        high = (high_product>>jnp.uint32(16))+(low<low_product).astype(jnp.uint32)
        extra = (ranks*rem+divisor-jnp.uint32(1))//divisor
        boundary_low = low+extra
        boundary_high = high+(boundary_low<low).astype(jnp.uint32)
        live = (ranks <= divisor) & ~bad
        bounds[0,:] = jnp.where(live,boundary_low,jnp.uint32(0))
        bounds[1,:] = jnp.where(live,boundary_high,jnp.uint32(0))
        next_extra = ((ranks+jnp.uint32(1))*rem+divisor-jnp.uint32(1))//divisor
        counts[0,:] = jnp.where((ranks<divisor)&~bad,qlo+next_extra-extra,jnp.uint32(0))
        error[0,:] = jnp.where(ranks == 0,bad.astype(jnp.uint32),jnp.uint32(0))
    return pl.pallas_call(kernel,
        out_shape=(jax.ShapeDtypeStruct((2,128),jnp.uint32),
                   jax.ShapeDtypeStruct((1,128),jnp.uint32),
                   jax.ShapeDtypeStruct((1,128),jnp.uint32)),
        interpret=interpret,name='beam_final_exact_boundaries')(keep,prior_error)
