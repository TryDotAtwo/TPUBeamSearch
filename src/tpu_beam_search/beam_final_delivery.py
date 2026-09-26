"""One balanced destination plan for final requests and history records."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_final_boundaries import pallas_final_boundaries
from .beam_final_plan import pallas_final_history_plan


class FinalDeliveryPlan(NamedTuple):
    requests: jax.Array
    sources: jax.Array
    valid: jax.Array
    history: jax.Array
    destinations: jax.Array
    boundaries: jax.Array
    target_counts: jax.Array
    error: jax.Array


def pallas_final_delivery_plan(packed,keep,prior_error,*,world_size,interpret=False):
    """Consume capped compacted meta/index/valid from one frozen final epoch.

    Requests route to the original parent source; history routes to the new
    destination. Both share one local destination index. Invalid selection or
    boundary overflow cannot revive padding even when its index is zero.
    This prepares records only: caller must group/chunk by the corresponding
    rank key and validity, perform actual exchange, and verify target coverage
    before publication. Packed global indices must already be unique.
    """
    if (packed.ndim != 2 or packed.shape[0] != 11 or not packed.shape[1]
            or packed.shape[1]%128 or packed.dtype != jnp.uint32):
        raise ValueError('invalid packed final delivery ABI')
    bounds,counts,error = pallas_final_boundaries(keep,prior_error,
        world_size=world_size,interpret=interpret)
    n = packed.shape[1]
    def unpack(x,b,e,meta,index):
        live = (x[10,:] != 0) & (e[0,0] == 0)
        meta[...] = jnp.where(live[None],x[:8,:],jnp.uint32(0))
        for word in range(2):
            index[word,:] = jnp.where(live,x[8+word,:],b[word,world_size])
    tile = lambda rows: pl.BlockSpec((rows,128),lambda i:(0,i))
    meta,index = pl.pallas_call(unpack,
        out_shape=(jax.ShapeDtypeStruct((8,n),jnp.uint32),
                   jax.ShapeDtypeStruct((2,n),jnp.uint32)),
        in_specs=(tile(11),pl.BlockSpec((2,128)),pl.BlockSpec((1,128))),
        out_specs=(tile(8),tile(2)),grid=(n//128,),interpret=interpret,
        name='beam_final_delivery_unpack')(packed,bounds,error)
    requests,sources,valid,history,destinations = pallas_final_history_plan(
        meta,index,bounds,world_size=world_size,interpret=interpret)
    return FinalDeliveryPlan(requests,sources,valid,history,destinations,
                             bounds,counts,error)
