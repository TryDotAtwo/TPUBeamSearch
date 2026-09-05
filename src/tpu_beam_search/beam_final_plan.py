"""Compose balanced destinations with source-routed final materialize requests."""
from .beam_final_balance import pallas_final_balance
from .beam_final_request import pallas_final_requests


def pallas_final_plan(meta, indices, boundaries, *, world_size, interpret=False):
    """Return request words, source rank keys, and selected-row validity.

    Global indices already encode the source algorithm's less/equal phases
    and exact cap. Boundaries are agreed across ranks. This function does not
    choose winners or exchange data. Caller MUST compact by returned validity
    before sending; inactive request words are unspecified and must not be
    consumed. Parent/source/owner metadata is never rewritten by balancing.
    """
    if meta.ndim != 2 or indices.ndim != 2 or meta.shape[1] != indices.shape[1]:
        raise ValueError('final metadata and global indices must align')
    ranks, local, valid = pallas_final_balance(indices,boundaries,
        world_size=world_size,interpret=interpret)
    requests, sources = pallas_final_requests(meta,local,ranks,interpret=interpret)
    return requests,sources,valid
