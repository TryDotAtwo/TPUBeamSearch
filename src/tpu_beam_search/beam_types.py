"""Lossless host preparation for the TPU beam's uint32 SoA wire format.

These helpers run before device execution or during diagnostics, not in the
search hot path. Bit significance is independent of host byte order.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .tpu_layout import pad_to_multiple


def _integer(value, minimum, maximum):
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError('expected an integer')
    if not minimum <= int(value) <= maximum:
        raise ValueError('integer outside representable range')
    return int(value)


@dataclass(frozen=True)
class BeamStorage:
    STATE_LEN: int
    MOVE_COUNT: int
    capacity: int
    WORLD_SIZE: int = 8

    def __post_init__(self):
        _integer(self.STATE_LEN, 1, 2**31 - 5)
        _integer(self.MOVE_COUNT, 1, 256)
        _integer(self.capacity, 1, 2**32 - 128)
        _integer(self.WORLD_SIZE, 1, 256)

    @property
    def STATE_STORAGE_LEN(self):
        return pad_to_multiple(self.STATE_LEN + 4, 16)

    @property
    def response_index_offset(self):
        return self.STATE_LEN

    @property
    def candidate_capacity(self):
        return pad_to_multiple(self.capacity, 128)

    @property
    def metadata_bytes(self):
        return self.candidate_capacity * 8 * 4


def pack_candidates(hashes, parents, scores, routes, *, capacity):
    capacity = _integer(capacity, 1, 2**32 - 128)
    fields = [list(field) for field in (hashes, parents, scores, routes)]
    count = len(fields[0])
    if count > capacity or any(len(field) != count for field in fields):
        raise ValueError('field lengths must match and fit logical capacity')
    # Validate before allocating or converting to fixed-width NumPy values.
    fields = [[_integer(v, 0, 2**bits - 1) for v in field]
              for field, bits in zip(fields, (128, 64, 32, 32))]
    words = np.zeros((8, pad_to_multiple(capacity, 128)), np.uint32)
    words[6, :] = np.uint32(0xffffffff)
    for plane in range(4):
        words[plane, :count] = [(value >> (32 * plane)) & 0xffffffff for value in fields[0]]
    for plane in range(2):
        words[4 + plane, :count] = [(value >> (32 * plane)) & 0xffffffff for value in fields[1]]
    words[6, :count], words[7, :count] = fields[2], fields[3]
    return words


def unpack_candidates(words, *, count):
    words = np.asarray(words)
    if words.ndim != 2 or words.shape[0] != 8 or words.dtype != np.uint32:
        raise ValueError('metadata must be uint32 [8, capacity]')
    count = _integer(count, 0, words.shape[1])
    hashes = [sum(int(words[p, i]) << (32 * p) for p in range(4)) for i in range(count)]
    parents = [int(words[4, i]) | (int(words[5, i]) << 32) for i in range(count)]
    return hashes, parents, words[6, :count].tolist(), words[7, :count].tolist()
