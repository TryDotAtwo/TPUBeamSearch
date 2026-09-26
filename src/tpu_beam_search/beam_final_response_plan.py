"""Consume a delivery epoch with its final error before authorizing responses."""
import jax.numpy as jnp
from .beam_final_receive import pallas_materialize_final_snapshots
from .beam_final_response_routing import pallas_prepare_final_response_exchange


def pallas_final_response_plan(parents,generators,snapshots,counts,error,target_counts,
                               *,state_len,world_size,interpret=False):
    """Return grouped response planes, intervals and local error.

    `error` MUST be the final common error from the paired request/history
    delivery epoch, not the earlier request-only result. Both receive and
    materialization errors gate routing. `target_counts` contains logical
    destination frontier capacities, never received chunk sizes. Parents and
    generators must already have the materializer's aligned tile width.

    A request epoch can produce world_size*128 responses for one destination.
    Its caller must execute up to world_size response chunks on EVERY rank
    (or collectively agree a smaller bound), even for empty/error ranks.
    Use make_final_response_chunk_call for collective error agreement and
    transfer. This preparation neither transfers nor publishes a frontier;
    coverage, accumulated history and scratch lifetime remain caller-owned.
    """
    if (type(world_size) is not int or not 1<=world_size<=128
            or snapshots.shape!=(world_size,4,128)
            or target_counts.shape!=(world_size,) or target_counts.dtype!=jnp.uint32):
        raise ValueError('invalid final response plan geometry')
    wire,validation,control,requests = pallas_materialize_final_snapshots(
        parents,generators,snapshots,counts,error,target_counts[:1],
        state_len=state_len,interpret=interpret,return_counts=target_counts)
    return pallas_prepare_final_response_exchange(wire,requests,control,validation,
        world_size=world_size,interpret=interpret)
