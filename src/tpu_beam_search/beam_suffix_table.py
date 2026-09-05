"""Immutable K2 chain preparation, outside the device search hot path."""
from dataclasses import dataclass
import numpy as np
from .beam_types import _integer
from .tpu_layout import pad_to_multiple


@dataclass(frozen=True)
class K2SuffixTable:
    words: np.ndarray
    count: int
    move_count: int
    radius: int


def prepare_k2_suffix_table(*,move_count,radius,max_count=0):
    """Source BFS chain order; uint32 low/high packed moves and length planes.

    Low5bits are the first move. Padding is not another empty suffix: consumers
    must use count and scan IDs1..count-1. Radius0 returns the canonical empty
    list entry; the runtime must separately bypass K2 when disabled, as CUDA
    does. This prepares immutable data, not K1 lookup or a Pallas suffix scan.
    """
    move_count = _integer(move_count,1,32)
    radius = _integer(radius,0,3)
    max_count = _integer(max_count,0,2**64-1)
    count = sum(move_count**depth for depth in range(radius+1))
    if max_count and count > max_count:
        raise ValueError('K2 suffix count exceeds configured limit')
    words = np.zeros((3,pad_to_multiple(count,128)),np.uint32)
    first,last = 0,1
    write = 1
    for depth in range(radius):
        for parent in range(first,last):
            packed = int(words[0,parent]) | (int(words[1,parent])<<32)
            for move in range(move_count):
                child = packed | (move<<(5*depth))
                words[0,write] = child & 0xffffffff
                words[1,write] = child>>32
                words[2,write] = depth+1
                write += 1
        first,last = last,write
    words.flags.writeable = False
    return K2SuffixTable(words,count,move_count,radius)
