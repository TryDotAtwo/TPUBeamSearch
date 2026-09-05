"""Per-request source-compatible reason bits before final materialization."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_validate_final_requests(requests,count,frontier_size,target_count,*,
                                   move_count,require_local_slot=False,interpret=False):
    """Bits parent1, target2, move4, local-slot8; padding reasons are zero.

    frontier_size is low/high uint32. Caller ensures count<=capacity, all
    return ranks valid, then aggregates reasons and gates materialization.
    Slot equality applies only to local indexed requests, not arbitrary grouped
    remote requests. This does not itself report/stop an invalid batch.
    """
    if (requests.ndim != 2 or requests.shape[0] != 4 or not requests.shape[1]
            or requests.shape[1]%128 or count.shape != (1,)
            or frontier_size.shape != (2,) or target_count.shape != (1,)
            or not isinstance(move_count,int) or not 1 <= move_count <= 32
            or any(x.dtype != jnp.uint32 for x in (requests,count,frontier_size,target_count))):
        raise ValueError('invalid final validation ABI')
    def kernel(r,n,f,t,out):
        index = pl.program_id(0).astype(jnp.uint32)*jnp.uint32(128)+jnp.arange(128,dtype=jnp.uint32)
        parent = (r[1] > f[1]) | ((r[1] == f[1]) & (r[0] >= f[0]))
        target = r[2] >= t[0]
        move = ((r[3]>>jnp.uint32(16))&jnp.uint32(255)) >= move_count
        reason = parent.astype(jnp.uint32) | (target.astype(jnp.uint32)*2) | (move.astype(jnp.uint32)*4)
        if require_local_slot:
            reason |= (r[2] != index).astype(jnp.uint32)*8
        out[...] = jnp.where(index < n[0],reason,jnp.uint32(0))[None]
    return pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct((1,requests.shape[1]),jnp.uint32),
        in_specs=(pl.BlockSpec((4,128),lambda i:(0,i)),pl.BlockSpec((1,)),
                  pl.BlockSpec((2,)),pl.BlockSpec((1,))),
        out_specs=pl.BlockSpec((1,128),lambda i:(0,i)),
        grid=(requests.shape[1]//128,),interpret=interpret,
        name='beam_final_request_validation')(requests,count,frontier_size,target_count)
