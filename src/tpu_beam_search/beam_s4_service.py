"""Serialized collector-control -> S4 commit -> S5 job-accounting bridge.

This services one logical sibling pair, not an entire depth or concurrent
scheduler. It inherits the diagnostic sort/histogram capacity bounds.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .beam_s4_ready import pallas_claim_ready
from .beam_s4_commit import pallas_run_reserved_s4


def pallas_service_s4_pair(a,b,controls,hist_a,hist_b,active,epoch,threshold,*,
                           bins,clean_ready_threshold,dirty_trigger,
                           force_dirty=False,force_clean=False,interpret=False):
    """Run at most one S4 job without extracting runtime counts on the host.

    Hist A/B are tuples of two [1,padded_bins] arrays, one per physical sibling.
    Active [1,128] lanes 0/1 select their committed versions. Epoch [4,128]
    follows S5's ABI. Its completed-job counter advances only after the physical
    commit's returned control, whose stores follow records and histogram DMA.

    Caller excludes collector/S4/S5 concurrent access and retains all returned
    arrays until completion. This functional composition does not establish
    whole-pair residency, scratch aliasing, distributed stop, or DMA overlap.
    A saturated job counter rejects admission and sets sticky fatal, never wraps.
    """
    hist_a,hist_b = tuple(hist_a),tuple(hist_b)
    width = ((bins+127)//128)*128
    if (a.ndim != 2 or a.shape[0] != 8 or b.shape != a.shape
            or controls.shape != (8,128) or active.shape != (1,128)
            or epoch.shape != (4,128) or threshold.shape != (1,)
            or len(hist_a) != 2 or len(hist_b) != 2
            or any(x.shape != (1,width) for x in (*hist_a,*hist_b))
            or any(x.dtype != jnp.uint32 for x in
                   (a,b,controls,active,epoch,threshold,*hist_a,*hist_b))):
        raise ValueError('invalid serialized S4 pair state')
    claimed,job = pallas_claim_ready(controls,capacity=a.shape[1],
        clean_ready_threshold=clean_ready_threshold,dirty_trigger=dirty_trigger,
        force_dirty=force_dirty,force_clean=force_clean,interpret=interpret)

    def guard(old,new,j,e,out,job_out):
        overflow = (j[0,0] != 0)&(e[0,0] == jnp.uint32(0xffffffff))
        fatal_lane = (jnp.arange(8)[:,None] == 7)&(jnp.arange(128)[None] == 0)
        rejected = jnp.where(fatal_lane,jnp.uint32(1),old[...])
        out[...] = jnp.where(overflow,rejected,new[...])
        job_out[...] = jnp.where(overflow,jnp.uint32(0),j[...])
    claimed,job = pl.pallas_call(guard,
        out_shape=(jax.ShapeDtypeStruct((8,128),jnp.uint32),
                   jax.ShapeDtypeStruct((2,128),jnp.uint32)),
        interpret=interpret,name='beam_s4_pair_counter_admission')(
            controls,claimed,job,epoch)

    def run_selected(slot):
        def run(_):
            def extract(c,h,out):
                values = jnp.stack((c[slot,0],c[2+slot,0],c[4+slot,0],h[0,slot]))
                out[...] = jnp.where(jnp.arange(128)[None] == 0,
                                     values[:,None],jnp.uint32(0))
            physical = pl.pallas_call(extract,
                out_shape=jax.ShapeDtypeStruct((4,128),jnp.uint32),
                interpret=interpret,name='beam_s4_pair_reservation')(claimed,active)
            resident,ha,hb,complete = pallas_run_reserved_s4(
                (a,b)[slot],hist_a[slot],hist_b[slot],physical,threshold,
                bins=bins,interpret=interpret)

            def publish(c,h,e,done,co,ho,eo):
                rows = jnp.arange(8)[:,None]
                lane = jnp.arange(128)[None] == 0
                value = jnp.where(rows == slot,done[0,0],
                    jnp.where(rows == 2+slot,done[1,0],
                    jnp.where(rows == 4+slot,done[2,0],c[...])))
                co[...] = jnp.where(lane,value,c[...])
                ho[...] = jnp.where(jnp.arange(128)[None] == slot,done[3,0],h[...])
                eo[...] = jnp.where((jnp.arange(4)[:,None] == 0)&lane,
                                    e[0,0]+jnp.uint32(1),e[...])
            control,new_active,new_epoch = pl.pallas_call(publish,
                out_shape=(jax.ShapeDtypeStruct((8,128),jnp.uint32),
                           jax.ShapeDtypeStruct((1,128),jnp.uint32),
                           jax.ShapeDtypeStruct((4,128),jnp.uint32)),
                interpret=interpret,name='beam_s4_pair_completed_job')(
                    claimed,active,epoch,complete)
            residents = (resident,b) if slot == 0 else (a,resident)
            new_ha = (ha,hist_a[1]) if slot == 0 else (hist_a[0],ha)
            new_hb = (hb,hist_b[1]) if slot == 0 else (hist_b[0],hb)
            return (*residents,control,new_ha,new_hb,new_active,new_epoch,job)
        return run

    def run(_):
        return jax.lax.cond(job[1,0] == 0,run_selected(0),run_selected(1),None)
    return jax.lax.cond(job[0,0] != 0,run,
        lambda _:(a,b,claimed,hist_a,hist_b,active,epoch,job),None)
