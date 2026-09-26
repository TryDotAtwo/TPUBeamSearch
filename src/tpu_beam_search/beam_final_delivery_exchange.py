"""Paired request/history routing epochs; returned data stays private."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_final_group import pallas_group_final_records
from .beam_final_intervals import pallas_final_rank_intervals
from .beam_final_chunk import pallas_pack_final_chunk
from .beam_final_exchange import make_final_chunk_exchange
from .beam_final_receive import pallas_compact_final_received


def pallas_prepare_delivery_exchange(plan,*,world_size,interpret=False):
    """Group request by parent source and history by final destination.

    Any invalid rank in either route blocks both through shared preparation
    error. Selected source metadata is not rewritten to match the destination.
    """
    requests = pallas_group_final_records(plan.requests,plan.sources,plan.valid,interpret=interpret)
    history = pallas_group_final_records(plan.history,plan.destinations,plan.valid,interpret=interpret)
    ri = pallas_final_rank_intervals(requests[4:5],requests[6:7],world_size=world_size,interpret=interpret)
    hi = pallas_final_rank_intervals(history[5:6],history[7:8],world_size=world_size,interpret=interpret)
    def merge(p,r,h,out):
        bad = (p[0,0] != 0) | (r[2,0] != 0) | (h[2,0] != 0)
        out[...] = jnp.where(jnp.arange(128)[None] == 0,bad.astype(jnp.uint32),jnp.uint32(0))
    error = pl.pallas_call(merge,out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
        interpret=interpret,name='beam_final_delivery_route_error')(plan.error,ri,hi)
    return requests,ri,history,hi,error


def make_delivery_chunk_call(mesh,*,interpret=False):
    """One common epoch: request exchange, then history exchange.

    Every rank calls the same epoch schedule, including exhausted/error ranks.
    Request error gates the history transfer too. Returned request snapshots
    MUST be consumed with the final common error, since a later history error
    invalidates this whole delivery epoch. Requests remain source-major for
    materialization; history is compacted at its final destination. No frontier
    or host history is published here; no overlap or cross-epoch reuse claim.
    """
    request_exchange = make_final_chunk_exchange(mesh,planes=4,interpret=interpret)
    history_exchange = make_final_chunk_exchange(mesh,planes=5,interpret=interpret)
    def call(requests,request_intervals,history,history_intervals,error,chunk):
        if requests.ndim != 2 or requests.shape[0] != 7 or history.ndim != 2 or history.shape[0] != 8:
            raise ValueError('invalid prepared delivery plane counts')
        payload,control = pallas_pack_final_chunk(requests[:4],request_intervals,chunk,
            world_size=mesh.size,prior_error=error,interpret=interpret)
        snapshots,counts,common = request_exchange(payload,control)
        payload,control = pallas_pack_final_chunk(history[:5],history_intervals,chunk,
            world_size=mesh.size,prior_error=common,interpret=interpret)
        history_snapshots,history_counts,common = history_exchange(payload,control)
        received,status = pallas_compact_final_received(history_snapshots,history_counts,
            common,interpret=interpret)
        return snapshots,counts,received,status,common
    return call
