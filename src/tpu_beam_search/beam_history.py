"""Host reconstruction over rank-local history; no TPU transfer implementation."""
from dataclasses import dataclass
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
