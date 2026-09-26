"""Checked history epochs into private target-indexed tiled SoA."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from .beam_final_error_summary import pallas_final_error_summary
from .beam_final_streaming_coverage import pallas_mark_final_targets
from .beam_s5_request import make_s5_request_call


def pallas_history_tiles_to_soa(history,*,interpret=False):
    """Explicit copy to the host publication ABI; not a zero-copy claim."""
    if (history.ndim!=3 or not history.shape[0] or history.shape[1:]!=(5,128)
            or history.dtype!=jnp.uint32):
        raise ValueError('invalid tiled history geometry')
    def copy(source,out):
        out[...] = source[0]
    return pl.pallas_call(copy,
        out_shape=jax.ShapeDtypeStruct((5,history.shape[0]*128),jnp.uint32),
        in_specs=(pl.BlockSpec((1,5,128),lambda i:(i,0,0)),),
        out_specs=pl.BlockSpec((5,128),lambda i:(0,i)),grid=(history.shape[0],),
        interpret=interpret,name='beam_history_tiles_to_soa')(history)


def make_final_history_consumer(mesh,*,move_count,interpret=False):
    """Receive history independently of response arrival order.

    History [capacity/128,5,128] is zero-initialized private storage per depth.
    Preserve parent64 and original route. Reject malformed live records before
    marking targets; collectively reject duplicate/out-of-range targets before
    any history store. DMA read-modify-write is sequential, no concurrent arena
    writers. Parent bounds against the old SOURCE frontier remain a publication
    check; this function never narrows parent64 or rewrites source to destination.
    No publication, host transfer, physical alias or throughput claim is made.
    """
    if type(move_count) is not int or not 1<=move_count<=256:
        raise ValueError('invalid history move count')
    agree=make_s5_request_call(mesh,interpret=interpret)
    def call(history,seen,coverage,records,status,target_count,prior_error):
        if (history.ndim!=3 or history.shape[1:]!=(5,128) or not history.shape[0]
                or history.dtype!=jnp.uint32 or seen.shape!=(1,history.shape[0]*128)
                or records.ndim!=2 or records.shape[0]!=5 or not records.shape[1]
                or records.shape[1]%128 or records.dtype!=jnp.uint32
                or status.shape!=(2,128) or coverage.shape!=(2,128)
                or prior_error.shape!=(1,128)
                or any(x.dtype!=jnp.uint32 for x in (status,coverage,prior_error))):
            raise ValueError('invalid history consumer geometry')
        n=records.shape[1]
        def check(r,s,out):
            index=pl.program_id(0)*128+jnp.arange(128)
            bad=(r[4]!=1)|((r[2]>>jnp.uint32(16))>=mesh.size)|((r[2]&jnp.uint32(255))>=move_count)
            out[...] = ((index.astype(jnp.uint32)<s[0,0])&bad).astype(jnp.uint32)[None]
        reasons=pl.pallas_call(check,out_shape=jax.ShapeDtypeStruct((1,n),jnp.uint32),
            in_specs=(pl.BlockSpec((5,128),lambda i:(0,i)),pl.BlockSpec((2,128))),
            out_specs=pl.BlockSpec((1,128),lambda i:(0,i)),grid=(n//128,),
            interpret=interpret,name='beam_history_consumer_validate')(records,status)
        summary=pallas_final_error_summary(reasons,interpret=interpret)
        def merge(c,s,p,v,out):
            out[0,:]=c[0,:]
            bad=(c[1,0]!=0)|(s[1,0]!=0)|(p[0,0]!=0)|(v[0,0]!=0)
            out[1,:]=jnp.where(jnp.arange(128)==0,bad.astype(jnp.uint32),jnp.uint32(0))
        merge_call=pl.pallas_call(merge,out_shape=jax.ShapeDtypeStruct((2,128),jnp.uint32),
            interpret=interpret,name='beam_history_consumer_error')
        coverage=merge_call(coverage,status,prior_error,summary)
        seen,coverage=pallas_mark_final_targets(seen,coverage,records[3:4],status[0,:1],
            target_count,interpret=interpret)
        common=agree(coverage[1:2])
        def scatter(old,r,s,e,out,tile,sem):
            index=pl.program_id(0)
            @pl.when((index.astype(jnp.uint32)<s[0,0])&(e[0,0]==0))
            def live():
                lanes=jnp.arange(128)
                fields=jnp.sum(jnp.where((lanes==index%128)[None],r[...],jnp.uint32(0)).astype(jnp.int32),axis=1).astype(jnp.uint32)
                target=fields[3]
                block=(target//jnp.uint32(128)).astype(jnp.int32)
                load=pltpu.make_async_copy(out.at[pl.ds(block,1),:,:],tile,sem)
                load.start()
                load.wait()
                tile[0,:,:]=jnp.where((lanes==(target%jnp.uint32(128)).astype(jnp.int32))[None],fields[:,None],tile[0])
                store=pltpu.make_async_copy(tile,out.at[pl.ds(block,1),:,:],sem)
                store.start()
                store.wait()
        hbm=pl.BlockSpec(memory_space=pltpu.HBM)
        history=pl.pallas_call(scatter,out_shape=jax.ShapeDtypeStruct(history.shape,jnp.uint32),
            in_specs=(hbm,pl.BlockSpec((5,128),lambda i:(0,i//128)),pl.BlockSpec((2,128)),pl.BlockSpec((1,128))),
            out_specs=hbm,input_output_aliases={0:0},grid=(n,),
            scratch_shapes=(pltpu.VMEM((1,5,128),jnp.uint32),pltpu.SemaphoreType.DMA),
            interpret=interpret,name='beam_history_consumer_scatter')(history,records,status,common)
        coverage=merge_call(coverage,status,common,summary)
        return history,seen,coverage,common
    return call
