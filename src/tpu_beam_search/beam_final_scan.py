"""Ordered tiled phase ordinals; local count contract is uint32."""
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_final_phase_scan(masks,*,interpret=False):
    """Return same-shape ordinals and phase counts [2,128] in lane0.

    Input is binary uint32[2,shards,capacity]. Unselected ordinals are
    UINT32_MAX. Grid order preserves shard/slot traversal per phase; counts
    are local, not global. No parallel grid dimensions may be introduced
    without replacing the count dependency with a hierarchical scan.
    """
    if (masks.ndim != 3 or masks.shape[0] != 2 or not masks.shape[1]
            or not masks.shape[2] or masks.shape[2]%128
            or masks.shape[1]*masks.shape[2] >= 0xffffffff or masks.dtype != jnp.uint32):
        raise ValueError('invalid final phase scan ABI')
    def kernel(m,out,count):
        phase,shard,tile = pl.program_id(0),pl.program_id(1),pl.program_id(2)
        @pl.when((phase == 0)&(shard == 0)&(tile == 0))
        def initialize():
            count[...] = jnp.zeros((2,128),jnp.uint32)
        values = m[0,0,:]
        prefix = lax.associative_scan(jnp.add,values)
        base = count[phase,0]
        out[0,0,:] = jnp.where(values != 0,base+prefix-jnp.uint32(1),jnp.uint32(0xffffffff))
        count[phase,:] = jnp.where(jnp.arange(128) == 0,base+prefix[-1],jnp.uint32(0))
    tile_spec = pl.BlockSpec((1,1,128),lambda p,s,t:(p,s,t))
    return pl.pallas_call(kernel,
        out_shape=(jax.ShapeDtypeStruct(masks.shape,jnp.uint32),jax.ShapeDtypeStruct((2,128),jnp.uint32)),
        in_specs=(tile_spec,),out_specs=(tile_spec,pl.BlockSpec((2,128),lambda p,s,t:(0,0))),
        grid=(2,masks.shape[1],masks.shape[2]//128),interpret=interpret,
        name='beam_final_phase_scan')(masks)
