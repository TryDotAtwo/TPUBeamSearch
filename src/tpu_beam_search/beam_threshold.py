"""Carry-aware periodic threshold arithmetic; no S5 collective/publication."""
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_periodic_threshold(histogram,beam,prior,*,bins,interpret=False):
    """Histogram/beam low-high pairs; prior/result threshold-initialized pairs.

    Global histogram sum must fit uint64 and beam must be positive. Caller
    performs coordinated reduction before entry and publishes returned state.
    """
    if (histogram.ndim != 2 or histogram.shape[0] != 2 or histogram.shape[1]%128
            or not 0 < bins <= histogram.shape[1] < 0x7fffffff
            or beam.shape != (2,128) or prior.shape != (2,128)
            or any(x.dtype != jnp.uint32 for x in (histogram,beam,prior))):
        raise ValueError('invalid periodic threshold ABI')
    def add(x,y):
        lo = x[0]+y[0]
        return lo,x[1]+y[1]+(lo < x[0]).astype(jnp.uint32)
    def scan(h,target,out):
        rows = jnp.broadcast_to(jnp.arange(3,dtype=jnp.int32)[:,None],(3,128))
        lanes = jnp.arange(128,dtype=jnp.int32)
        @pl.when(pl.program_id(0) == 0)
        def initialize():
            out[...] = jnp.where(rows == 2,jnp.uint32(0x7fffffff),jnp.uint32(0))
        index = pl.program_id(0)*128+lanes
        values = jnp.where((index < bins)[None],h[...],jnp.uint32(0))
        # Tuple associative_scan recursively emits uint32[0] slices. Mosaic
        # rejects those before execution (physical V7). Keep all scan stages
        # width128 and preserve carry-aware integer addition exactly.
        low,high = values[0],values[1]
        for distance in (1,2,4,8,16,32,64):
            shifted_low = jnp.concatenate((jnp.zeros((distance,),jnp.uint32),low[:-distance]))
            shifted_high = jnp.concatenate((jnp.zeros((distance,),jnp.uint32),high[:-distance]))
            low,high = add((low,high),(shifted_low,shifted_high))
        low,high = add((low,high),(out[0,0],out[1,0]))
        reached = (high > target[1,0])|((high == target[1,0])&(low >= target[0,0]))
        first = jnp.min(jnp.where(reached&(index < bins),index,jnp.int32(0x7fffffff)))
        selected = jnp.minimum(out[2,0],first.astype(jnp.uint32))
        out[...] = jnp.where(rows == 0,low[-1],jnp.where(rows == 1,high[-1],selected))
    state = pl.pallas_call(scan,
        out_shape=jax.ShapeDtypeStruct((3,128),jnp.uint32),
        in_specs=(pl.BlockSpec((2,128),lambda i:(0,i)),pl.BlockSpec((2,128))),
        out_specs=pl.BlockSpec((3,128),lambda i:(0,0)),
        grid=(histogram.shape[1]//128,),interpret=interpret,name='beam_threshold_uint64_scan')(histogram,beam)
    def finish(s,old,out):
        selected = s[2,0]
        enough = selected != jnp.uint32(0x7fffffff)
        initialized = old[1,0] != 0
        value = jnp.where(enough,jnp.where(initialized,jnp.minimum(old[0,0],selected),selected),
                          jnp.where(initialized,old[0,0],jnp.uint32(0xffffffff)))
        rows = jnp.arange(2)[:,None]
        out[...] = jnp.where(jnp.arange(128)[None] == 0,
            jnp.where(rows == 0,value,(initialized|enough).astype(jnp.uint32)),jnp.uint32(0))
    return pl.pallas_call(finish,out_shape=jax.ShapeDtypeStruct((2,128),jnp.uint32),
        in_specs=(pl.BlockSpec((3,128)),pl.BlockSpec((2,128))),out_specs=pl.BlockSpec((2,128)),
        interpret=interpret,name='beam_periodic_threshold_choice')(state,prior)
