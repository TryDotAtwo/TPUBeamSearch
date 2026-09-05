"""Host reference for serialized collector reservation, not TPU execution.

One logical shard has two resident siblings. This reference does not publish
dirty counts to concurrent consumers; a device implementation must complete
record stores before committing counts. No group splitting or spill fallback.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Reservation:
    buffer: int | None
    offset: int
    dirty: tuple[int, int]
    fatal_overflow: bool


def reserve_group(*, capacity, clean, dirty, processing, current, amount):
    """Choose current if writable and fitting, otherwise its sole sibling.

    Inputs are validated host scalars; fatal overflow leaves counts untouched.
    Empty groups are no-ops even if both buffers are full or processing.
    """
    if (not isinstance(capacity, int) or not 0 < capacity <= 0xffffffff
            or not isinstance(amount, int) or not 0 <= amount <= 0xffffffff
            or current not in (0,1)
            or len(clean) != 2 or len(dirty) != 2 or len(processing) != 2):
        raise ValueError('invalid collector geometry')
    if any(not isinstance(x,int) or x < 0 for x in (*clean,*dirty)):
        raise ValueError('invalid resident counts')
    if any(clean[i]+dirty[i] > capacity for i in (0,1)):
        raise ValueError('resident counts exceed capacity')
    if amount == 0:
        return Reservation(None,0,tuple(dirty),False)
    for index in (current,1-current):
        used = clean[index]+dirty[index]
        if not processing[index] and amount <= capacity-used:
            updated = list(dirty)
            updated[index] += amount
            return Reservation(index,used,tuple(updated),False)
    return Reservation(None,0,tuple(dirty),True)


def pallas_collector_append(a, b, incoming, control, count, *, interpret=False):
    """Serialized functional append of <=128 records to one resident A/B pair.

    Control [8,128] lane zero: clean A/B, dirty A/B, busy A/B, current, fatal.
    Caller must await this entire call before publishing returned state. There
    is no concurrent S4 reader, in-place alias guarantee or DMA overlap here.
    Valid controls/counts are required; prior fatal is sticky and blocks writes.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    if (a.ndim != 2 or a.shape[0] != 8 or b.shape != a.shape
            or a.shape[1] < 128 or a.shape[1] % 128):
        raise ValueError('resident buffers must be [8,128*k]')
    if incoming.shape != (8,128) or control.shape != (8,128) or count.shape != (1,128):
        raise ValueError('invalid collector input geometry')
    if any(x.dtype != jnp.uint32 for x in (a,b,incoming,control,count)):
        raise ValueError('collector requires uint32')
    capacity = a.shape[1]

    def decision(c,n):
        used_a, used_b = c[0,0]+c[2,0], c[1,0]+c[3,0]
        amount = n[0,0]
        fit_a = (c[4,0] == 0) & (amount <= capacity-used_a)
        fit_b = (c[5,0] == 0) & (amount <= capacity-used_b)
        active = (amount != 0) & (c[7,0] == 0)
        choose_a = active & fit_a & ((c[6,0] == 0) | ~fit_b)
        choose_b = active & fit_b & ((c[6,0] != 0) | ~fit_a)
        failed = (c[7,0] != 0) | (active & ~fit_a & ~fit_b)
        return used_a,used_b,amount,choose_a,choose_b,failed

    def append(ar,br,ir,cr,nr,ao,bo):
        ua,ub,amount,ca,cb,_ = decision(cr,nr)
        index = (pl.program_id(0)*128+jnp.arange(128)).astype(jnp.uint32)
        for old,out,offset,chosen in ((ar,ao,ua,ca),(br,bo,ub,cb)):
            relative = index-offset
            # Safe modulo addressing for masked-off lanes; incoming is one tile.
            values = jnp.take(ir[...],(relative & 127).astype(jnp.int32),axis=1)
            mask = chosen & (index >= offset) & (relative < amount)
            out[...] = jnp.where(mask[None],values,old[...])

    spec = pl.BlockSpec((8,128),lambda tile:(0,tile))
    aa,bb = pl.pallas_call(append,
        out_shape=(jax.ShapeDtypeStruct(a.shape,jnp.uint32),)*2,
        in_specs=(spec,spec,pl.BlockSpec(incoming.shape),pl.BlockSpec(control.shape),pl.BlockSpec(count.shape)),
        out_specs=(spec,spec),grid=(capacity//128,),interpret=interpret,
        name='beam_collector_append_records')(a,b,incoming,control,count)

    def commit(cr,nr,out):
        _,_,amount,ca,cb,failed = decision(cr,nr)
        rows = jnp.arange(8)[:,None]
        lane = jnp.arange(128)[None] == 0
        result = cr[...]
        result = result + ((rows == 2)&lane).astype(jnp.uint32)*ca.astype(jnp.uint32)*amount
        result = result + ((rows == 3)&lane).astype(jnp.uint32)*cb.astype(jnp.uint32)*amount
        current = jnp.where(ca,jnp.uint32(0),jnp.where(cb,jnp.uint32(1),cr[6,0]))
        result = jnp.where((rows == 6)&lane,current,result)
        out[...] = jnp.where((rows == 7)&lane,failed.astype(jnp.uint32),result)

    updated = pl.pallas_call(commit,out_shape=jax.ShapeDtypeStruct(control.shape,jnp.uint32),
        in_specs=(pl.BlockSpec(control.shape),pl.BlockSpec(count.shape)),
        out_specs=pl.BlockSpec(control.shape),grid=(),interpret=interpret,
        name='beam_collector_next_control')(control,count)
    return aa,bb,updated


def pallas_collector_append_group(a,b,incoming,control,count,*,interpret=False):
    """Serialized multi-tile group: preflight the whole group before appending.

    Functional correctness implementation, not an in-place resident scheduler.
    The full returned tuple must complete before publication to any consumer.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    if (incoming.ndim != 2 or incoming.shape[0] != 8
            or incoming.shape[1] < 128 or incoming.shape[1] % 128
            or control.shape != (8,128) or count.shape != (1,128)
            or a.ndim != 2 or a.shape[0] != 8 or b.shape != a.shape
            or a.shape[1] < 128 or a.shape[1] % 128):
        raise ValueError('invalid group geometry')
    if any(x.dtype != jnp.uint32 for x in (a,b,incoming,control,count)):
        raise ValueError('collector requires uint32')
    capacity = a.shape[1]

    def reserve(c,n,out):
        amount = n[0,0]
        ua,ub = c[0,0]+c[2,0],c[1,0]+c[3,0]
        fa = (c[4,0] == 0)&(ua <= capacity)&(amount <= capacity-ua)
        fb = (c[5,0] == 0)&(ub <= capacity)&(amount <= capacity-ub)
        active = (amount != 0)&(c[7,0] == 0)
        ca = fa&((c[6,0] == 0)|~fb)
        chosen = jnp.where(ca,jnp.uint32(0),jnp.uint32(1))
        failed = (c[7,0] != 0)|(amount > incoming.shape[1])|(active&~fa&~fb)
        rows, lane = jnp.arange(8)[:,None],jnp.arange(128)[None] == 0
        result = jnp.where((rows == 6)&lane&active&~failed,chosen,c[...])
        out[...] = jnp.where((rows == 7)&lane,failed.astype(jnp.uint32),result)

    control = pl.pallas_call(reserve,out_shape=jax.ShapeDtypeStruct(control.shape,jnp.uint32),
        in_specs=(pl.BlockSpec(control.shape),pl.BlockSpec(count.shape)),
        out_specs=pl.BlockSpec(control.shape),grid=(),interpret=interpret,
        name='beam_collector_reserve_whole_group')(control,count)
    for offset in range(0,incoming.shape[1],128):
        def tile_amount(n,out,offset=offset):
            # Signed clamping avoids unsigned underflow for exhausted tiles.
            remaining = n[0,0].astype(jnp.int32)-offset
            amount = jnp.minimum(jnp.maximum(remaining,0),128).astype(jnp.uint32)
            out[...] = (jnp.arange(128)[None] == 0).astype(jnp.uint32)*amount
        amount = pl.pallas_call(tile_amount,out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
            in_specs=(pl.BlockSpec(count.shape),),out_specs=pl.BlockSpec(count.shape),
            grid=(),interpret=interpret,name='beam_collector_tile_count')(count)
        a,b,control = pallas_collector_append(a,b,incoming[:,offset:offset+128],
                                             control,amount,interpret=interpret)
    return a,b,control
