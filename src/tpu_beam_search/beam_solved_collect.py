"""Serialized functional solved collector; no concurrent publication guarantee."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_collect_solved(arena, control, records, found, *,
                          stop_on_found, interpret=False):
    """Ten record planes: CandidateMeta eight, depth, suffix ID.

    Control [4,128] lane zero: attempted count, overflow, solved flag, stop.
    Caller supplies valid solved metadata/flags and guarantees uint32 attempted
    count will not wrap. Existing records and sticky flags are retained. Caller
    must await the whole call before exposing results; not a concurrent atomic
    append or a system-wide stop epoch. Full arenas reside in VMEM here, making
    this a small-capacity diagnostic baseline, not scalable HBM collection.
    """
    if (arena.ndim != 2 or arena.shape[0] != 10 or not arena.shape[1]
            or arena.shape[1]%128 or records.ndim != 2 or records.shape[0] != 10
            or not records.shape[1] or records.shape[1]%128
            or found.shape != (1,records.shape[1]) or control.shape != (4,128)
            or any(x.dtype != jnp.uint32 for x in (arena,control,records,found))):
        raise ValueError('invalid solved collector ABI')
    capacity,n = arena.shape[1],records.shape[1]
    def kernel(a,c,r,f,out,ctl):
        out[...] = a[...]
        ctl[...] = c[...]
        lanes = jnp.arange(n,dtype=jnp.int32)
        slots = jnp.arange(capacity,dtype=jnp.int32)
        controls = jnp.arange(128,dtype=jnp.int32)
        def candidate(index,_):
            hit = jnp.sum(jnp.where(lanes == index,f[0],jnp.uint32(0)).astype(jnp.int32)) != 0
            @pl.when(hit)
            def append():
                offset = ctl[0,0]
                for row in range(10):
                    value = jnp.sum(jnp.where(lanes == index,r[row],jnp.uint32(0)).astype(jnp.int32)).astype(jnp.uint32)
                    out[row,:] = jnp.where(slots.astype(jnp.uint32) == offset,value,out[row,:])
                first = ctl[2,0] == 0
                updated = (offset+jnp.uint32(1),
                           ctl[1,0] | (offset >= capacity).astype(jnp.uint32),
                           jnp.uint32(1),
                           ctl[3,0] | (first & jnp.bool_(stop_on_found)).astype(jnp.uint32))
                for row in range(4):
                    ctl[row,:] = jnp.where(controls == 0,updated[row],ctl[row,:])
        jax.lax.fori_loop(0,n,candidate,None)
    return pl.pallas_call(kernel,
        out_shape=(jax.ShapeDtypeStruct(arena.shape,jnp.uint32),
                   jax.ShapeDtypeStruct(control.shape,jnp.uint32)),
        interpret=interpret,name='beam_serialized_solved_collect')(arena,control,records,found)
