"""Single-host completed-result publication boundary for the beam caller.

Device coverage/collective error and DMA ordering remain producer obligations.
This module waits actual returned arrays and stages host history; it does not
infer remote DMA completion from an error flag or invent a device barrier.
"""
from dataclasses import dataclass

import jax
import numpy as np

from .beam_history import RankHistoryStore, decode_history_soa


@dataclass(frozen=True)
class PublishedBeamDepth:
    depth: int
    frontiers: tuple
    counts: tuple[int, ...]
    history: RankHistoryStore


def commit_final_publication(current, *, frontier_by_rank, history_by_rank,
                             target_counts, common_error, completed_work,
                             move_count):
    """Prepare one frontier/history handle, retaining ``current`` on device error.

    All frontier candidates must be private and the old frontier must remain
    undonated. Producers must return actual results depending on every required
    send/receive/store/consumer completion in ``completed_work``. Waiting those
    leaves does not cover unregistered asynchronous work. Coverage and clean
    frontier padding must already be included in the device acceptance gate.

    Only counts, errors and history records are copied to the host. A malformed
    late history entry raises before any public layer or frontier is changed.
    The single-host caller publishes by replacing its enclosing state handle
    with this return value; this is not a multi-host distributed commit.
    """
    frontier_by_rank, history_by_rank = tuple(frontier_by_rank), tuple(history_by_rank)
    world_size = len(current.frontiers)
    if (world_size == 0 or len(current.counts) != world_size
            or len(frontier_by_rank) != world_size or len(history_by_rank) != world_size
            or not isinstance(current.depth, int) or current.depth < 0):
        raise ValueError('publication must contain every rank of one depth')
    for old, candidate, count in zip(current.frontiers, frontier_by_rank, current.counts, strict=True):
        if (not isinstance(candidate, jax.Array) or old.ndim != 2 or candidate.ndim != 2
                or candidate.dtype != np.uint8 or old.dtype != np.uint8
                or candidate.shape[1] != old.shape[1]
                or not isinstance(count, int) or not 0 <= count <= old.shape[0]):
            raise ValueError('invalid persistent frontier publication geometry')

    # Wait success and failure paths alike before releasing private results.
    jax.block_until_ready((frontier_by_rank, history_by_rank, target_counts,
                           common_error, completed_work))
    host_counts, host_error = map(np.asarray, jax.device_get((target_counts, common_error)))
    if (host_counts.shape != (world_size,) or host_counts.dtype != np.uint32
            or host_error.shape != (world_size, 1, 128) or host_error.dtype != np.uint32):
        raise ValueError('invalid publication count/error controls')
    if np.any(host_error):
        return current
    counts = tuple(int(value) for value in host_counts)
    if any(count > frontier.shape[0] for count, frontier in zip(counts, frontier_by_rank, strict=True)):
        raise ValueError('published count exceeds frontier capacity')
    host_history = jax.device_get(history_by_rank)

    def checked_records(records):
        for target, entry in decode_history_soa(np.asarray(records), world_size=world_size,
                                                move_count=move_count):
            source = entry.route_packed >> 16
            if entry.parent_idx >= current.counts[source]:
                raise ValueError('history parent exceeds previous logical frontier')
            yield target, entry

    history = current.history.stage_all_rank_layer(
        (checked_records(records) for records in host_history),
        target_counts=counts, depth=current.depth,
    )
    return PublishedBeamDepth(current.depth + 1, frontier_by_rank, counts, history)
