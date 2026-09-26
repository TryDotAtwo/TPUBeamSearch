"""Frozen resident prefixes to exact capped metadata, before materialization."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_final_resident import pallas_final_resident_view
from .beam_final_phase import pallas_final_phase_masks
from .beam_final_scan import pallas_final_phase_scan
from .beam_final_count_exchange import make_final_count_exchange
from .beam_final_prefix import pallas_final_prefixes
from .beam_final_cap import pallas_final_cap
from .beam_final_indices import pallas_final_indices
from .beam_final_compact import pallas_final_compact
from .beam_s5_request import make_s5_request_call


def make_final_selection_call(mesh, *, interpret=False):
    """All ranks use the same supplied final threshold and beam for one epoch.

    Returns packed meta/index/valid, global keep, common error, rank counts.
    Resident dirty/busy/fatal errors participate in unconditional agreement;
    any error rejects all output candidates, including healthy ranks. L>K is
    a threshold invariant failure, never permission to truncate the less phase.
    This does not derive the final threshold, flush work, compute destination
    boundaries, materialize states or publish history. HBM bitonic compaction
    remains the existing diagnostic O(N log^2 N) baseline.
    """
    agree = make_s5_request_call(mesh,interpret=interpret)
    exchange = make_final_count_exchange(mesh,interpret=interpret)
    def call(a,b,controls,threshold,beam,prior_error):
        meta,clean,local_error = pallas_final_resident_view(
            a,b,controls,prior_error,interpret=interpret)
        common_error = agree(local_error)
        masks = pallas_final_phase_masks(meta[:,6,:],clean,threshold,interpret=interpret)
        ordinal,local_counts = pallas_final_phase_scan(masks,interpret=interpret)
        counts = exchange(local_counts)
        bases,totals = pallas_final_prefixes(counts,world_size=mesh.size,interpret=interpret)
        keep,cap_error = pallas_final_cap(totals,beam,interpret=interpret)
        def errors(a,b,out):
            out[...] = jnp.where(jnp.arange(128)[None] == 0,
                ((a[0,0] != 0) | (b[0,0] != 0)).astype(jnp.uint32),jnp.uint32(0))
        error = pl.pallas_call(errors,
            out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
            interpret=interpret,name='beam_final_selection_error')(common_error,cap_error)
        indices,valid = pallas_final_indices(ordinal,bases,keep,error,
            rank=0 if mesh.size == 1 else None,interpret=interpret)
        # Explicit tiled Pallas layout conversion to the compactor ABI.
        def planes(x,out):
            out[:,0,:] = x[0,:,:]
        shaped = pl.pallas_call(planes,
            out_shape=jax.ShapeDtypeStruct((8,meta.shape[0],meta.shape[2]),jnp.uint32),
            in_specs=(pl.BlockSpec((1,8,128),lambda s,t:(s,0,t)),),
            out_specs=pl.BlockSpec((8,1,128),lambda s,t:(0,s,t)),
            grid=(meta.shape[0],meta.shape[2]//128),interpret=interpret,
            name='beam_final_compaction_layout')(meta)
        packed = pallas_final_compact(shaped,indices,valid,interpret=interpret)
        return packed,keep,error,counts
    return call
