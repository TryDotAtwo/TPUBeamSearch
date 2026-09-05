"""Source-compatible solved metadata assembly, not beam routing metadata."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_solved_records(hashes, suffix_ids, parent_base, depth, *,
                          move_count, local_rank, interpret=False):
    """Ten planes: hash4, parent2, goal score0, route, depth, suffix.

    parent_base is uint32 [2] low/high; caller guarantees uint64 addition fits.
    Candidate order is parent-major, move-minor. Padding metadata is generated
    but never valid: collector must receive the separately masked found flags.
    """
    if (not isinstance(move_count,int) or not 1 <= move_count <= 32
            or not isinstance(local_rank,int) or not 0 <= local_rank <= 255
            or hashes.ndim != 2 or hashes.shape[0] != 4
            or not hashes.shape[1] or hashes.shape[1]%128
            or suffix_ids.shape != (1,hashes.shape[1])
            or parent_base.shape != (2,) or depth.shape != (1,)
            or any(x.dtype != jnp.uint32 for x in (hashes,suffix_ids,parent_base,depth))):
        raise ValueError('invalid solved metadata ABI')
    def kernel(h,s,b,d,out):
        lane = pl.program_id(0).astype(jnp.uint32)*jnp.uint32(128)+jnp.arange(128,dtype=jnp.uint32)
        low = (b[0]+lane//jnp.uint32(move_count)).astype(jnp.uint32)
        high = b[1]+(low < b[0]).astype(jnp.uint32)
        out[:4,:] = h[...]
        out[4,:],out[5,:] = low,high
        out[6,:] = jnp.zeros((128,),jnp.uint32)
        out[7,:] = jnp.uint32((local_rank<<16)|(local_rank<<8)) | (lane%move_count)
        out[8,:] = jnp.full((128,),d[0],jnp.uint32)
        out[9,:] = s[0,:]
    return pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((10,hashes.shape[1]),jnp.uint32),
        in_specs=(pl.BlockSpec((4,128),lambda i:(0,i)),
                  pl.BlockSpec((1,128),lambda i:(0,i)),
                  pl.BlockSpec((2,)),pl.BlockSpec((1,))),
        out_specs=pl.BlockSpec((10,128),lambda i:(0,i)),
        grid=(hashes.shape[1]//128,),interpret=interpret,
        name='beam_solved_record_metadata')(hashes,suffix_ids,parent_base,depth)
