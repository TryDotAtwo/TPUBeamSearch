"""One physical S4 buffer commit: records, histogram, then release control."""
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def pallas_run_reserved_s4(resident,a,b,control,threshold,*,bins,interpret=False):
    """Execute one exclusively reserved physical buffer; no host count read.

    Caller sets processing before entry, excludes sibling S4 jobs and passes a
    captured threshold. Returns new resident/hist/control versions. Scratch is
    still functional intermediates, not a verified preallocated arena.
    """
    from .beam_external_sort import pallas_external_stream4_dedup
    from .beam_s4_histogram import pallas_score_histogram
    def input_count(c,out):
        total = c[0,0]+c[1,0]
        out[...] = (jnp.arange(128)[None] == 0).astype(jnp.uint32)*total
    total = pl.pallas_call(input_count,
        out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
        in_specs=(pl.BlockSpec((4,128)),),out_specs=pl.BlockSpec((1,128)),
        interpret=interpret,name='beam_s4_clean_dirty_count')(control)
    records,count = pallas_external_stream4_dedup(resident,total[:,0],threshold,interpret=interpret)
    histogram = pallas_score_histogram(records,count[:,0],bins=bins,interpret=interpret)
    return pallas_commit_s4(resident,a,b,control,records,histogram,count,interpret=interpret)


def pallas_commit_s4(resident,a,b,control,records,histogram,count,*,interpret=False):
    """Aliased resident/hist A/B/control outputs; control lane0 is
    [clean,dirty,processing,hist_active]. Caller reserves this physical buffer
    exclusively and supplies a complete clean result/histogram/count.
    No global queue, S5 snapshot concurrency or physical allocation proof.
    """
    if (resident.ndim != 2 or resident.shape[0] != 8 or resident.shape[1] % 128
            or not resident.shape[1] or records.shape != resident.shape
            or a.ndim != 2 or a.shape[0] != 1 or not a.shape[1] or a.shape[1] % 128
            or b.shape != a.shape or histogram.shape != a.shape
            or control.shape != (4,128) or count.shape != (1,128)
            or any(x.dtype != jnp.uint32 for x in (resident,a,b,control,records,histogram,count))):
        raise ValueError('invalid physical S4 commit ABI')
    def kernel(ri,ai,bi,ci,source,hist,n,ro,ao,bo,co,record_tile,hist_tile,state,num,sem):
        def copy(src,dst):
            op = pltpu.make_async_copy(src,dst,sem)
            op.start()
            op.wait()
        copy(ci,state)
        copy(n,num)
        active = state[3,0]&jnp.uint32(1)
        def record_body(i,_):
            section = pl.ds(i*128,128)
            copy(source.at[:,section],record_tile)
            copy(record_tile,ro.at[:,section])
        lax.fori_loop(0,resident.shape[1]//128,record_body,None)
        def hist_body(i,_):
            section = pl.ds(i*128,128)
            copy(hist.at[:,section],hist_tile)
            @pl.when(active == 0)
            def write_b():
                copy(hist_tile,bo.at[:,section])
            @pl.when(active != 0)
            def write_a():
                copy(hist_tile,ao.at[:,section])
        lax.fori_loop(0,a.shape[1]//128,hist_body,None)
        rows = jnp.arange(4,dtype=jnp.int32)[:,None]
        values = jnp.where(rows == 0,num[0,0],
            jnp.where(rows == 3,active^jnp.uint32(1),jnp.uint32(0)))
        state[...] = jnp.where(jnp.arange(128)[None] == 0,values,state[...])
        copy(state,co)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(kernel,
        out_shape=tuple(jax.ShapeDtypeStruct(x.shape,x.dtype) for x in (resident,a,b,control)),
        in_specs=(hbm,)*7,out_specs=(hbm,)*4,
        input_output_aliases={0:0,1:1,2:2,3:3},
        scratch_shapes=(pltpu.VMEM((8,128),jnp.uint32),pltpu.VMEM((1,128),jnp.uint32),
            pltpu.VMEM((4,128),jnp.uint32),pltpu.VMEM((1,128),jnp.uint32),pltpu.SemaphoreType.DMA),
        interpret=interpret,name='beam_s4_records_hist_release')(resident,a,b,control,records,histogram,count)
