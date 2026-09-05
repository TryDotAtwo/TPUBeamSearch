"""Bounded-VMEM two-bucket K1 lookup; serialized DMA baseline."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from .beam_k1_keys import pallas_k1_keys


def pallas_k1_contains(hashes,table,count,*,bucket_count,interpret=False):
    """Table rows: fingerprint, Hash128 words0..3. Empty fingerprint is zero.

    Each bucket has four slots; allocation is padded to128 slots. Count must
    fit query capacity. Two aligned table DMA reads per valid query, not a
    coalesced-query implementation or physical performance claim.
    """
    keys = pallas_k1_keys(hashes,bucket_count=bucket_count,interpret=interpret)
    slots = max(128,4*bucket_count)
    if (table.shape != (5,slots) or table.dtype != jnp.uint32
            or count.shape != (1,) or count.dtype != jnp.uint32
            or slots > 0x7fffffff):
        raise ValueError('invalid K1 table/count ABI')
    capacity = hashes.shape[1]
    def kernel(h,k,t,n,out,window,sem):
        lanes = jnp.arange(128,dtype=jnp.int32)
        out[...] = jnp.zeros((1,128),jnp.uint32)
        def at_lane(row,index):
            return jnp.sum(jnp.where(lanes == index,row,jnp.uint32(0)).astype(jnp.int32)).astype(jnp.uint32)
        def query(index,_):
            @pl.when((pl.program_id(0)*128+index).astype(jnp.uint32) < n[0])
            def valid_query():
                fingerprint = at_lane(k[0],index)
                query_hash = tuple(at_lane(h[row],index) for row in range(4))
                hit = jnp.bool_(False)
                for choice in (1,2):
                    base = at_lane(k[choice],index)*jnp.uint32(4)
                    start = (base&jnp.uint32(0xffffff80)).astype(jnp.int32)
                    copy = pltpu.make_async_copy(t.at[:,pl.ds(start,128)],window,sem)
                    copy.start()
                    copy.wait()
                    local = (base&jnp.uint32(127)).astype(jnp.int32)
                    match = (lanes >= local)&(lanes < local+4)&(window[0] == fingerprint)
                    for row in range(4):
                        match &= window[row+1] == query_hash[row]
                    hit |= jnp.sum(match.astype(jnp.int32)) != 0
                out[...] = jnp.where((lanes == index)[None],hit.astype(jnp.uint32),out[...])
        jax.lax.fori_loop(0,128,query,None)
    return pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((1,capacity),jnp.uint32),
        in_specs=(pl.BlockSpec((4,128),lambda i:(0,i)),
            pl.BlockSpec((3,128),lambda i:(0,i)),
            pl.BlockSpec(memory_space=pltpu.HBM),pl.BlockSpec((1,))),
        out_specs=pl.BlockSpec((1,128),lambda i:(0,i)),grid=(capacity//128,),
        scratch_shapes=(pltpu.VMEM((5,128),jnp.uint32),pltpu.SemaphoreType.DMA),
        interpret=interpret,name='beam_k1_two_bucket_contains')(hashes,keys,table,count)
