"""Serialized all-pair S4 service followed by a coordinated S5 epoch.

Every rank calls the same service rounds, including empty and fatal ranks.
This is an integration baseline, not the complete depth or resident scheduler.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_s4_service import pallas_service_s4_pair
from .beam_s5_epoch import make_s5_epoch_call
from .beam_s5_request import make_s5_request_call


def _fatal(controls, *, interpret):
    def kernel(*refs):
        bad = jnp.uint32(0)
        for control in refs[:-1]:
            bad |= (control[7,0] != 0).astype(jnp.uint32)
        refs[-1][...] = jnp.where(jnp.arange(128)[None] == 0,bad,jnp.uint32(0))
    return pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
                          interpret=interpret,name='beam_s4_rank_fatal')(*controls)


def _with_fatal(control, error, *, interpret):
    def kernel(c,e,out):
        flag = ((c[7,0] != 0)|(e[0,0] != 0)).astype(jnp.uint32)
        at_flag = (jnp.arange(8)[:,None] == 7)&(jnp.arange(128)[None] == 0)
        out[...] = jnp.where(at_flag,flag,c[...])
    return pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct((8,128),jnp.uint32),
                          interpret=interpret,name='beam_s4_propagate_fatal')(control,error)


def _pack_histograms(pairs, *, interpret):
    physical = 2*len(pairs)
    width = pairs[0][3][0].shape[1]
    ha = tuple(h for pair in pairs for h in pair[3])
    hb = tuple(h for pair in pairs for h in pair[4])
    def pack(*refs):
        for index in range(physical):
            refs[-2][index:index+1,:] = refs[index][...]
            refs[-1][index:index+1,:] = refs[physical+index][...]
    source = pl.BlockSpec((1,128),lambda tile:(0,tile))
    dest = pl.BlockSpec((physical,128),lambda tile:(0,tile))
    arrays = pl.pallas_call(pack,
        out_shape=(jax.ShapeDtypeStruct((physical,width),jnp.uint32),)*2,
        in_specs=(source,)*(2*physical),out_specs=(dest,dest),grid=(width//128,),
        interpret=interpret,name='beam_s4_committed_histogram_rows')(*ha,*hb)
    def versions(*refs):
        lanes = jnp.arange(128)[None]
        result = jnp.zeros((1,128),jnp.uint32)
        for index,ref in enumerate(refs[:-1]):
            result |= jnp.where(lanes == 2*index,ref[0,0],jnp.uint32(0))
            result |= jnp.where(lanes == 2*index+1,ref[0,1],jnp.uint32(0))
        refs[-1][...] = result
    active = pl.pallas_call(versions,out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
        interpret=interpret,name='beam_s4_committed_histogram_versions')(*(p[5] for p in pairs))
    return *arrays,active


def make_s4_s5_service_call(mesh, *, bins,period,clean_ready_threshold,
                            dirty_trigger,force_dirty=False,force_clean=False,
                            interpret=False,explicit_hbm_output=False):
    """Pairs: (resident A/B, control8, hist A/B tuples, hist-active lanes).

    At most 64 logical pairs fit the existing histogram control ABI. Local S4
    work contains no collective; the fatal agreement is unconditional. Only a
    common fatal result skips S5, uniformly on all ranks. A healthy rank with
    zero local jobs still participates in S5's request collective.

    Caller freezes collectors/readers during this round and keeps private
    outputs alive. This does not publish frontier/history or roll back private
    work after another rank fails. S5 update-count overflow remains a caller
    bound. Histogram packing is functional copying, not a residency claim.
    """
    agree = make_s5_request_call(mesh,interpret=interpret)
    s5 = make_s5_epoch_call(mesh,bins=bins,period=period,interpret=interpret,
                            explicit_hbm_output=explicit_hbm_output)
    def call(pairs,epoch,slots,beam,force):
        pairs = tuple(pairs)
        if not 1 <= len(pairs) <= 64:
            raise ValueError('one to 64 logical S4 pairs required')
        def capture(a,b,index,out):
            selected = jnp.where(index[0,0] == 0,a[...],b[...])
            out[0] = jnp.where(selected[1,0] != 0,selected[0,0],jnp.uint32(0xffffffff))
        threshold = pl.pallas_call(capture,out_shape=jax.ShapeDtypeStruct((1,),jnp.uint32),
            interpret=interpret,name='beam_s4_capture_service_threshold')(*slots)
        local_error = _fatal(tuple(p[2] for p in pairs),interpret=interpret)
        updated,jobs = [],[]
        for pair in pairs:
            a,b,control,ha,hb,active = pair
            control = _with_fatal(control,local_error,interpret=interpret)
            result = pallas_service_s4_pair(a,b,control,ha,hb,active,epoch,threshold,
                bins=bins,clean_ready_threshold=clean_ready_threshold,
                dirty_trigger=dirty_trigger,force_dirty=force_dirty,
                force_clean=force_clean,interpret=interpret)
            updated.append(result[:6])
            epoch = result[6]
            jobs.append(result[7])
            local_error = _fatal((result[2],),interpret=interpret)
        common_error = agree(local_error)
        # All ranks see the same decision; never branch here on a local job.
        def publish(_):
            hist_a,hist_b,active = _pack_histograms(updated,interpret=interpret)
            next_a,next_b,next_active,next_epoch = s5(
                hist_a,hist_b,active,*slots,beam,epoch,force)
            return (next_a,next_b,next_active),next_epoch
        next_slots,next_epoch = jax.lax.cond(common_error[0,0] == 0,publish,
                                             lambda _:(slots,epoch),None)
        final_pairs = tuple((p[0],p[1],_with_fatal(p[2],common_error,interpret=interpret),
                             p[3],p[4],p[5]) for p in updated)
        return final_pairs,next_epoch,next_slots,common_error,tuple(jobs)
    return call
