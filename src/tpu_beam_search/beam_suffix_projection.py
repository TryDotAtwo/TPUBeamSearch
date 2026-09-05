"""Pallas preparation of K2 composed source-index permutations."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_stream2 import _take_clipped


def pallas_suffix_projection(generators,words,*,count,interpret=False):
    """Return int32 [state_width,suffix_capacity], valid suffixes in columns.

    Input must be a validated K2 table (length<=3, move IDs in range) and valid
    source-index generators. This composes only the suffix; Stream2 applies
    the immediate move afterwards. Padding columns are zero, not valid entries.
    """
    if (generators.ndim != 2 or not 1 <= generators.shape[0] <= 32
            or not generators.shape[1] or generators.shape[1]%128
            or generators.dtype != jnp.int32 or words.ndim != 2
            or words.shape[0] != 3 or not words.shape[1] or words.shape[1]%128
            or words.dtype != jnp.uint32 or not 1 <= count <= words.shape[1]):
        raise ValueError('invalid aligned K2 projection ABI')
    width,capacity = generators.shape[1],words.shape[1]
    def kernel(g,w,out):
        lane = pl.program_id(0)*128+jnp.arange(128,dtype=jnp.int32)
        packed,length = w[0],w[2].astype(jnp.int32)
        flat = g[...].reshape(-1)
        def position(p,_):
            source = jnp.full((128,),p,jnp.int32)
            for reverse_index in range(3):
                step = jnp.maximum(length-1-reverse_index,0)
                move = ((packed >> (5*step).astype(jnp.uint32))&jnp.uint32(31)).astype(jnp.int32)
                projected = _take_clipped(flat,move*width+source)
                source = jnp.where(reverse_index < length,projected,source)
            out[p,:] = jnp.where(lane < count,source,jnp.int32(0))
        jax.lax.fori_loop(0,width,position,None)
    return pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((width,capacity),jnp.int32),
        in_specs=(pl.BlockSpec(generators.shape),pl.BlockSpec((3,128),lambda i:(0,i))),
        out_specs=pl.BlockSpec((width,128),lambda i:(0,i)),grid=(capacity//128,),
        interpret=interpret,name='beam_k2_suffix_projection')(generators,words)
