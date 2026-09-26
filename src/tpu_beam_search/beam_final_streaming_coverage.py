"""Private target marks across epochs; serialized DMA, physical gate pending."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from .beam_final_error_summary import pallas_final_error_summary


def _geometry(seen,control,target_count):
    if (seen.ndim!=2 or seen.shape[0]!=1 or not 0<seen.shape[1]<1<<31
            or seen.shape[1]%128 or control.shape!=(2,128) or target_count.shape!=(1,)
            or any(x.dtype!=jnp.uint32 for x in (seen,control,target_count))):
        raise ValueError('invalid streaming coverage geometry')


def pallas_mark_final_targets(seen,control,targets,count,target_count,*,interpret=False):
    """Mark each target exactly once; error is sticky across calls.

    Marks are uint32 [1,capacity], zero-initialized once per private depth.
    Control [2,128] lane0 stores accepted count/error. The logical target count
    is frozen across epochs. Sequential grid programs read-modify-write one
    aligned128 mark tile, waiting each DMA before reuse. No atomics or parallel
    grid dimensions: concurrent writers to this arena are forbidden.

    Failure can leave a marked prefix PRIVATE; reject the entire depth rather
    than publish or retry with this state. Caller must merge transport errors
    and agree completion collectively. This is bounded-memory correctness
    machinery, not a throughput claim or a frontier scatter operation.
    """
    _geometry(seen,control,target_count)
    if (targets.ndim!=2 or targets.shape[0]!=1 or not targets.shape[1]
            or targets.shape[1]%128 or targets.shape[1]>=1<<31 or count.shape!=(1,)
            or targets.dtype!=jnp.uint32 or count.dtype!=jnp.uint32):
        raise ValueError('invalid target chunk geometry')
    capacity,n=seen.shape[1],targets.shape[1]
    def kernel(old,c,t,num,limit,out,status,tile,sem):
        index=pl.program_id(0)
        lanes=jnp.arange(128,dtype=jnp.int32)
        @pl.when(index==0)
        def initialize():
            bad=(c[1,0]!=0)|(num[0]>n)|(limit[0]>capacity)|(c[0,0]>limit[0])
            status[0,:]=jnp.where(lanes==0,c[0,0],jnp.uint32(0))
            status[1,:]=jnp.where(lanes==0,bad.astype(jnp.uint32),jnp.uint32(0))
        @pl.when((index.astype(jnp.uint32)<num[0])&(status[1,0]==0))
        def live():
            target=jnp.sum(jnp.where(lanes==index%128,t[0],jnp.uint32(0)).astype(jnp.int32)).astype(jnp.uint32)
            bad=(target>=limit[0])|(target>=capacity)|(status[0,0]>=limit[0])
            @pl.when(bad)
            def invalid():
                status[1,0]=jnp.uint32(1)
            @pl.when(~bad)
            def mark():
                block=(target//jnp.uint32(128)).astype(jnp.int32)
                load=pltpu.make_async_copy(out.at[pl.ds(block,1),:,:],tile,sem)
                load.start()
                load.wait()
                hit=lanes==(target%jnp.uint32(128)).astype(jnp.int32)
                duplicate=jnp.any(hit&(tile[0,0]!=0))
                status[1,0]=duplicate.astype(jnp.uint32)
                @pl.when(~duplicate)
                def store_mark():
                    tile[0,0,:]=jnp.where(hit,jnp.uint32(1),tile[0,0])
                    store=pltpu.make_async_copy(tile,out.at[pl.ds(block,1),:,:],sem)
                    store.start()
                    store.wait()
                    status[0,0]=status[0,0]+jnp.uint32(1)
    hbm=pl.BlockSpec(memory_space=pltpu.HBM)
    marks,status=pl.pallas_call(kernel,
        out_shape=(jax.ShapeDtypeStruct((capacity//128,1,128),jnp.uint32),
                   jax.ShapeDtypeStruct((2,128),jnp.uint32)),
        in_specs=(hbm,pl.BlockSpec((2,128)),pl.BlockSpec((1,128),lambda i:(0,i//128)),
                  pl.BlockSpec((1,)),pl.BlockSpec((1,))),
        out_specs=(hbm,pl.BlockSpec((2,128))),input_output_aliases={0:0},grid=(n,),
        scratch_shapes=(pltpu.VMEM((1,1,128),jnp.uint32),pltpu.SemaphoreType.DMA),
        interpret=interpret,name='beam_final_streaming_marks')(
            seen.reshape(capacity//128,1,128),control,targets,count,target_count)
    return marks.reshape(1,capacity),status


def pallas_finish_final_targets(seen,control,target_count,*,interpret=False):
    """Exact local prefix coverage, including bitmap and count consistency.

    Return normalized [1,128] error. Caller still agrees errors on all ranks
    and waits actual response/history consumers before publishing.
    """
    _geometry(seen,control,target_count)
    capacity=seen.shape[1]
    def check(m,c,n,out):
        index=pl.program_id(0)*128+jnp.arange(128)
        expected=(index.astype(jnp.uint32)<n[0]).astype(jnp.uint32)
        bad=(m[0]!=expected)|((index==0)&((c[1,0]!=0)|(c[0,0]!=n[0])|(n[0]>capacity)))
        out[...] = bad.astype(jnp.uint32)[None]
    tile=pl.BlockSpec((1,128),lambda i:(0,i))
    reasons=pl.pallas_call(check,out_shape=jax.ShapeDtypeStruct(seen.shape,jnp.uint32),
        in_specs=(tile,pl.BlockSpec((2,128)),pl.BlockSpec((1,))),out_specs=tile,
        grid=(capacity//128,),interpret=interpret,name='beam_final_streaming_coverage')(seen,control,target_count)
    summary=pallas_final_error_summary(reasons,interpret=interpret)
    def flag(s,out):
        out[...] = jnp.where(jnp.arange(128)[None]==0,(s[0,0]!=0).astype(jnp.uint32),jnp.uint32(0))
    return pl.pallas_call(flag,out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
        interpret=interpret,name='beam_final_streaming_coverage_flag')(summary)
