"""Serialized collector reference and experimental Pallas building blocks.

One logical shard has two resident siblings. This module does not publish
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


def pallas_collector_partition(words,shards,count,*,shard_count,interpret=False):
    """Stable shard grouping with supplied shard IDs; metadata is unchanged.

    Diagnostic external-sort baseline; caller supplies valid IDs and count.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from .beam_external_sort import pallas_external_bitonic_sort
    if (words.ndim != 2 or words.shape[0] != 8 or words.shape[1] < 128
            or words.shape[1] > 16384 or words.shape[1] & (words.shape[1]-1)
            or shards.shape != (1,words.shape[1]) or count.shape != (1,128)
            or not isinstance(shard_count,int) or not 1 <= shard_count <= 256):
        raise ValueError('invalid partition geometry')
    if any(x.dtype != jnp.uint32 for x in (words,shards,count)):
        raise ValueError('partition requires uint32')
    n = words.shape[1]
    width = ((shard_count+128)//128)*128
    ws = pl.BlockSpec((8,128),lambda b:(0,b))
    ds = pl.BlockSpec((11,128),lambda b:(0,b))
    def prepare(w,s,c,out):
        idx = (pl.program_id(0)*128+jnp.arange(128)).astype(jnp.uint32)
        valid = idx < c[0,0]
        out[...] = jnp.concatenate((w[...],s[...],valid[None].astype(jnp.uint32),idx[None]))
    data = pl.pallas_call(prepare,out_shape=jax.ShapeDtypeStruct((11,n),jnp.uint32),
        in_specs=(ws,pl.BlockSpec((1,128),lambda b:(0,b)),pl.BlockSpec(count.shape)),
        out_specs=ds,grid=(n//128,),interpret=interpret,
        name='beam_collector_partition_prepare')(words,shards,count)
    data = pallas_external_bitonic_sort(data,key_planes=(9,8,10),validity_plane=9,interpret=interpret)
    def finish(d,out,totals):
        valid = d[9] != 0
        neutral = jnp.where(jnp.arange(8)[:,None] == 6,jnp.uint32(0xffffffff),jnp.uint32(0))
        out[...] = jnp.where(valid[None],d[:8],neutral)
        amounts = [jnp.sum((valid&(d[8] == s)).astype(jnp.int32)).astype(jnp.uint32)
                   for s in range(shard_count)]
        totals[...] = jnp.stack(amounts)[:,None]*(jnp.arange(128)[None] == 0).astype(jnp.uint32)
    grouped,totals = pl.pallas_call(finish,
        out_shape=(jax.ShapeDtypeStruct((8,n),jnp.uint32),jax.ShapeDtypeStruct((shard_count,n),jnp.uint32)),
        in_specs=(ds,),out_specs=(ws,pl.BlockSpec((shard_count,128),lambda b:(0,b))),
        grid=(n//128,),interpret=interpret,name='beam_collector_partition_finish')(data)
    def reduce(t,c,o):
        pos = jnp.arange(width)[None]
        counts,offsets = jnp.zeros((1,width),jnp.uint32),jnp.zeros((1,width),jnp.uint32)
        running = jnp.uint32(0)
        for s in range(shard_count):
            amount = jnp.sum(t[s].astype(jnp.int32)).astype(jnp.uint32)
            counts += (pos == s).astype(jnp.uint32)*amount
            running += amount
            offsets += (pos == s+1).astype(jnp.uint32)*running
        c[...],o[...] = counts,offsets
    shape = jax.ShapeDtypeStruct((1,width),jnp.uint32)
    counts,offsets = pl.pallas_call(reduce,out_shape=(shape,shape),
        in_specs=(pl.BlockSpec(totals.shape),),out_specs=(pl.BlockSpec(shape.shape),)*2,
        grid=(),interpret=interpret,name='beam_collector_partition_counts')(totals)
    return grouped,counts,offsets


def pallas_collector_hash_partition(words,count,*,shard_count,interpret=False):
    """Group already owner-routed metadata by the independent Hash128 shard salt.

    No owner/source/move payload is rewritten. This is a functional baseline,
    not concurrent collector publication or an in-place resident scatter.
    """
    from .beam_hash import pallas_route_hashes
    routing = pallas_route_hashes(words[:4],world_size=1,
                                  shard_count=shard_count,interpret=interpret)
    return pallas_collector_partition(words,routing[1:2],count,
                                      shard_count=shard_count,interpret=interpret)


def pallas_collector_preflight(controls,counts,*,capacity,interpret=False):
    """Read-only admission for an entire partition; no dirty publication.

    Controls are [shards,8,128] in the single-shard ABI. Counts are aligned
    [1,W]. Plan [shards,4,128] lane zero holds sibling, offset, amount, enabled.
    Any overflow/prior fatal zeros ALL plans. Consumers must depend on this
    complete result before writes. Valid resident counts are a caller contract.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    if (controls.ndim != 3 or controls.shape[1:] != (8,128)
            or not 1 <= controls.shape[0] <= 256
            or counts.ndim != 2 or counts.shape[0] != 1
            or counts.shape[1] < controls.shape[0] or counts.shape[1] % 128
            or not isinstance(capacity,int) or not 0 < capacity <= 0xffffffff
            or capacity % 128):
        raise ValueError('invalid partition admission geometry')
    if controls.dtype != jnp.uint32 or counts.dtype != jnp.uint32:
        raise ValueError('admission requires uint32')
    shard_count = controls.shape[0]
    def kernel(c,n,out,status):
        failed = jnp.bool_(False)
        plans = []
        for s in range(shard_count):
            ua,ub = c[s,0,0]+c[s,2,0],c[s,1,0]+c[s,3,0]
            amount = n[0,s]
            fa = (c[s,4,0] == 0)&(ua <= capacity)&(amount <= capacity-ua)
            fb = (c[s,5,0] == 0)&(ub <= capacity)&(amount <= capacity-ub)
            active = amount != 0
            ca = fa&((c[s,6,0] == 0)|~fb)
            failed |= (c[s,7,0] != 0)|(active&~fa&~fb)
            entry = jnp.stack((jnp.where(ca,jnp.uint32(0),jnp.uint32(1)),
                               jnp.where(ca,ua,ub),amount,active.astype(jnp.uint32)))
            plans.append(jnp.where(active,entry,jnp.uint32(0)))
        lane = (jnp.arange(128,dtype=jnp.int32) == 0).astype(jnp.uint32)
        out[...] = jnp.where(failed,jnp.uint32(0),jnp.stack(plans)[:,:,None]*lane)
        status[...] = failed.astype(jnp.uint32)*lane[None]
    shape = (shard_count,4,128)
    return pl.pallas_call(kernel,
        out_shape=(jax.ShapeDtypeStruct(shape,jnp.uint32),jax.ShapeDtypeStruct((1,128),jnp.uint32)),
        in_specs=(pl.BlockSpec(controls.shape),pl.BlockSpec(counts.shape)),
        out_specs=(pl.BlockSpec(shape),pl.BlockSpec((1,128))),grid=(),
        interpret=interpret,name='beam_collector_partition_preflight')(controls,counts)
