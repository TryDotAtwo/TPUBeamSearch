"""Host reconstruction over rank-local history; no TPU transfer implementation."""
from dataclasses import dataclass
from array import array
from typing import Callable


@dataclass(frozen=True)
class HistoryEntry:
    parent_idx: int
    route_packed: int

    def __post_init__(self):
        if not isinstance(self.parent_idx,int) or not 0 <= self.parent_idx < 1 << 64:
            raise ValueError('history parent must fit uint64')
        if not isinstance(self.route_packed,int) or not 0 <= self.route_packed < 1 << 32:
            raise ValueError('history route must fit uint32')


@dataclass(frozen=True)
class HistoryPath:
    moves: tuple[int,...]
    parent_indices: tuple[int,...]


class RankHistoryStore:
    """Append-only host SoA history: uint64 parents and uint32 routes.

    Not a distributed service or concurrent writer. The caller publishes each
    rank's layer only after its transfer completes, and coordinates layer
    readiness before reconstruction. Failed validation publishes no layer.
    """

    def __init__(self, *, world_size: int):
        if not isinstance(world_size,int) or not 1 <= world_size <= 65536:
            raise ValueError('invalid history world_size')
        if array('Q').itemsize != 8 or array('I').itemsize != 4:
            raise RuntimeError('host array word sizes do not match history ABI')
        self._layers = [[] for _ in range(world_size)]

    def _rank(self, rank):
        if not isinstance(rank,int) or not 0 <= rank < len(self._layers):
            raise IndexError('history rank out of range')
        return self._layers[rank]

    def append_rank_layer(self, rank, records, *, target_count):
        layers = self._rank(rank)
        if not isinstance(target_count,int) or not 0 <= target_count < 1 << 32:
            raise ValueError('invalid history target_count')
        parents = array('Q',[0]) * target_count
        routes = array('I',[0]) * target_count
        seen = bytearray(target_count)
        count = 0
        for target,entry in records:
            if not isinstance(target,int) or not 0 <= target < target_count:
                raise ValueError('history target out of range')
            if seen[target]:
                raise ValueError('duplicate history target')
            if not isinstance(entry,HistoryEntry):
                raise TypeError('history records require HistoryEntry')
            parents[target],routes[target] = entry.parent_idx,entry.route_packed
            seen[target] = 1
            count += 1
        if count != target_count:
            raise ValueError('missing history target')
        layers.append((parents,routes))

    def read_entry(self, rank, layer, index):
        layers = self._rank(rank)
        if not isinstance(layer,int) or not 0 <= layer < len(layers):
            raise IndexError('history layer out of range')
        parents,routes = layers[layer]
        if not isinstance(index,int) or not 0 <= index < len(parents):
            raise IndexError('history parent index out of range')
        return HistoryEntry(parents[index],routes[index])


def reconstruct_history(solved: HistoryEntry, *, depth: int, world_size: int,
                        move_count: int,
                        read_entry: Callable[[int,int,int],HistoryEntry]) -> HistoryPath:
    """Follow (source rank, previous layer, local parent), returning forward order.

    The service must validate layer/index bounds and retain immutable history
    through this call. Depth counts generated moves; zero yields an empty path.
    K1/K2 suffixes and actual puzzle replay are separate caller responsibilities.
    No owner/target rank substitution or parent-index truncation is permitted.
    """
    if not isinstance(depth,int) or depth < 0:
        raise ValueError('history depth must be nonnegative')
    if not 1 <= world_size <= 65536 or not 1 <= move_count <= 256:
        raise ValueError('invalid history rank/move geometry')
    moves, parents = [], []
    cursor = solved
    for remaining in range(depth,0,-1):
        if not isinstance(cursor,HistoryEntry):
            raise TypeError('history service must return HistoryEntry')
        move = cursor.route_packed & 255
        if move >= move_count:
            raise ValueError('history move exceeds move_count')
        moves.append(move)
        parents.append(cursor.parent_idx)
        if remaining > 1:
            rank = cursor.route_packed >> 16
            if rank >= world_size:
                raise ValueError('history source rank exceeds world_size')
            cursor = read_entry(rank,remaining-2,cursor.parent_idx)
    return HistoryPath(tuple(reversed(moves)),tuple(reversed(parents)))
