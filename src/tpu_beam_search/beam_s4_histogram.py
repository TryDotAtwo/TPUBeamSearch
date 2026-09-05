"""Bounded score-sort/search histogram baseline; no atomics or publication."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_external_sort import pallas_external_bitonic_sort


def pallas_score_histogram(words,count,*,bins,interpret=False):
    """Return uint32 [1,ceil128(bins)] with zero padding.

    Reads only count valid clean records; ignores scores >= bins. Capacity is
    bounded by the whole sorted-score search window, not a scalable production
    histogram claim. Each bin performs two logarithmic searches, not an NxBins
    comparison matrix. Caller must publish only after this output completes.
    """
    if words.ndim != 2 or words.shape[0] != 8 or words.dtype != jnp.uint32:
        raise ValueError('metadata must be uint32 [8,N]')
    n = words.shape[1]
    if n < 128 or n > 16384 or n&(n-1):
        raise ValueError('histogram baseline capacity must be power of two128..16384')
    if count.shape != (1,) or count.dtype != jnp.uint32 or not 0 < bins < 0xffffffff:
        raise ValueError('invalid histogram count/bins')
    width = (bins+127)//128*128
    def prepare(w,c,out):
        idx = pl.program_id(0)*128+jnp.arange(128,dtype=jnp.int32)
        score = jnp.where(idx < c[0],w[6],jnp.uint32(0xffffffff))
        out[...] = jnp.broadcast_to(score[None],(8,128))
    tile = pl.BlockSpec((8,128),lambda t:(0,t))
    sorted_scores = pl.pallas_call(prepare,
        out_shape=jax.ShapeDtypeStruct((8,n),jnp.uint32),
        in_specs=(tile,pl.BlockSpec((1,))),out_specs=tile,
        grid=(n//128,),interpret=interpret,name='beam_hist_score_prepare')(words,count)
    sorted_scores = pallas_external_bitonic_sort(sorted_scores,key_planes=(0,),interpret=interpret)
    def search(s,out):
        keys = (pl.program_id(0)*128+jnp.arange(128,dtype=jnp.int32)).astype(jnp.uint32)
        def lower_bound(key):
            lo = jnp.zeros((128,),jnp.int32)
            hi = jnp.full((128,),n,jnp.int32)
            for _ in range(n.bit_length()):
                mid = (lo+hi)//2
                indices = jnp.broadcast_to(jnp.minimum(mid,n-1)[None],(8,128))
                value = jnp.take_along_axis(s[...],indices,axis=1)[0]
                right = (lo < hi)&(value < key)
                next_lo = jnp.where(right,mid+1,lo)
                hi = jnp.where((lo < hi)&~right,mid,hi)
                lo = next_lo
            return lo
        amount = lower_bound(keys+jnp.uint32(1))-lower_bound(keys)
        out[...] = jnp.where(keys < bins,amount,jnp.int32(0))[None].astype(jnp.uint32)
    return pl.pallas_call(search,
        out_shape=jax.ShapeDtypeStruct((1,width),jnp.uint32),
        in_specs=(pl.BlockSpec((8,n)),),out_specs=pl.BlockSpec((1,128),lambda t:(0,t)),
        grid=(width//128,),interpret=interpret,name='beam_hist_sorted_ranges')(sorted_scores)
