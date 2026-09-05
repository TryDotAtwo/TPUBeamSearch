"""Host inverse-neighborhood preparation, preserving source BFS/hash semantics."""
from dataclasses import dataclass
from types import MappingProxyType
from .beam_k1_table import K1Table, prepare_k1_table
from .beam_types import _integer
import numpy as np


@dataclass(frozen=True)
class K1Neighborhood:
    table: K1Table
    suffix_by_hash: object


def prepare_k1_neighborhood(central,generators,zobrist,*,state_len,radius,max_entries,bucket_count):
    """Inverse BFS; first Hash128 visit owns its suffix, as in the GPU builder.

    Suffix moves return a predecessor to the central state. Hash collisions are
    deduplicated by hash, not by full state, matching the source contract.
    max_entries=0 disables that cap; fixed bucket capacity still must fit.
    This is initialization, not device search or a CPU replacement for it.
    """
    central,generators,zobrist = map(np.asarray,(central,generators,zobrist))
    radius = _integer(radius,0,12)
    max_entries = _integer(max_entries,0,2**64-1)
    if central.ndim != 1 or not central.size or central.dtype != np.uint8:
        raise ValueError('central must be a nonempty uint8 state')
    width = central.size
    state_len = _integer(state_len,1,width)
    if (generators.ndim != 2 or not 1 <= generators.shape[0] <= 32
            or generators.shape[1] != width or generators.dtype != np.int32
            or not np.all(np.sort(generators,axis=1) == np.arange(width)[None])
            or np.any(generators[:,:state_len] >= state_len) or np.any(central[state_len:])):
        raise ValueError('generators must preserve the logical state and zero padding')
    if (zobrist.ndim != 2 or zobrist.shape[0] != 4 or zobrist.shape[1]%width
            or not zobrist.shape[1] or zobrist.dtype != np.uint32):
        raise ValueError('invalid Zobrist word table')
    classes = zobrist.shape[1]//width
    if np.any(central >= classes):
        raise ValueError('state values exceed Zobrist classes')
    def state_hash(state):
        return tuple(map(int,np.bitwise_xor.reduce(zobrist[:,np.arange(width)*classes+state],axis=1)))
    suffixes = {}
    frontier = []
    if radius:
        key = state_hash(central)
        suffixes[key] = ()
        frontier = [(central.copy(),())]
    for _ in range(radius):
        following = []
        for state,suffix in frontier:
            for move,generator in enumerate(generators):
                parent = np.zeros_like(state)
                parent[generator] = state
                parent[state_len:] = 0
                key = state_hash(parent)
                if key in suffixes:
                    continue
                if max_entries and len(suffixes) >= max_entries:
                    raise ValueError('K1 neighborhood exceeded max entries')
                chain = (move,*suffix)
                suffixes[key] = chain
                following.append((parent,chain))
        frontier = following
        if not frontier:
            break
    hashes = np.asarray(list(suffixes),np.uint32).reshape(-1,4).T
    table = prepare_k1_table(hashes,bucket_count=bucket_count)
    return K1Neighborhood(table,MappingProxyType(suffixes))
