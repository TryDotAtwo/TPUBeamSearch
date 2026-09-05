"""Serialized physical-sibling reservation; not a resident S4 execution proof."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def clean_ready_threshold(*,capacity,logical_shards,stream3_batch):
    """Static reserve geometry; wide host arithmetic avoids uint32 round-up wrap.

    Production dimensions must still be checked against allocation limits.
    No runtime candidate counts are read on the host.
    """
    if not all(isinstance(x,int) and 0 < x <= 0xffffffff
               for x in (capacity,logical_shards,stream3_batch)):
        raise ValueError('ready dimensions must be positive uint32 integers')
    average = (stream3_batch+logical_shards-1)//logical_shards
    reserve = min(0xffffffff,average+(average+3)//4)
    return capacity-min(capacity,reserve)


def pallas_claim_ready(controls,*,capacity,clean_ready_threshold,dirty_trigger,
                      force_dirty=False,force_clean=False,interpret=False):
    """CUDA ready selection for one logical A/B pair, including drain modes.

    Caller serializes this with collector control publication. Geometry and
    clean+dirty <= capacity are caller contracts. Returns controls and lane-zero
    [enabled, physical_buffer] job descriptor, not a globally compacted queue.
    """
    if controls.shape != (8,128) or controls.dtype != jnp.uint32:
        raise ValueError('controls must be uint32 [8,128]')
    if not 0 < capacity <= 0xffffffff or not 0 <= clean_ready_threshold <= capacity or dirty_trigger <= 0:
        raise ValueError('invalid ready geometry')
    def kernel(c,out,job):
        total_a,total_b = c[0,0]+c[2,0],c[1,0]+c[3,0]
        da = (c[2,0] != 0)&(force_dirty|(c[2,0] >= dirty_trigger)|(total_a >= clean_ready_threshold))
        db = (c[3,0] != 0)&(force_dirty|(c[3,0] >= dirty_trigger)|(total_b >= clean_ready_threshold))
        ra = da|((c[2,0] == 0)&(c[0,0] != 0)&(force_clean|(c[0,0] >= clean_ready_threshold)))
        rb = db|((c[3,0] == 0)&(c[1,0] != 0)&(force_clean|(c[1,0] >= clean_ready_threshold)))
        current = c[6,0]&jnp.uint32(1)
        # With two unoccupied siblings, larger sibling space means selecting
        # the fuller buffer. On equal space the non-current candidate wins,
        # except the source's final full-capacity tie rule selects current.
        tie_b = jnp.where(total_a >= capacity,current == 1,current == 0)
        prefer_b = (db&~da)|((db == da)&((total_b > total_a)|((total_b == total_a)&tie_b)))
        choose_b = rb&(~ra|prefer_b)
        enabled = (ra|rb)&(c[4,0] == 0)&(c[5,0] == 0)
        selected = choose_b.astype(jnp.uint32)
        sibling_space = jnp.where(choose_b,total_a,total_b) < capacity
        write = jnp.where(enabled&sibling_space,selected^jnp.uint32(1),c[6,0])
        rows = jnp.arange(8,dtype=jnp.uint32)[:,None]
        lane0 = jnp.arange(128,dtype=jnp.uint32)[None] == 0
        values = jnp.where((rows == 4+selected)&enabled,jnp.uint32(1),c[...])
        values = jnp.where(rows == 6,write,values)
        out[...] = jnp.where(lane0,values,c[...])
        job[...] = jnp.where(jnp.arange(2)[:,None] == 0,enabled.astype(jnp.uint32),
                            jnp.where(enabled,selected,jnp.uint32(0)))*lane0.astype(jnp.uint32)
    return pl.pallas_call(kernel,
        out_shape=(jax.ShapeDtypeStruct((8,128),jnp.uint32),jax.ShapeDtypeStruct((2,128),jnp.uint32)),
        in_specs=(pl.BlockSpec((8,128)),),out_specs=(pl.BlockSpec((8,128)),pl.BlockSpec((2,128))),
        grid=(),interpret=interpret,name='beam_claim_s4_ready')(controls)
