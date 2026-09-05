"""Host preparation of a fixed K1 bucket arena; no implicit growth or spill."""
from dataclasses import dataclass
import numpy as np
from .beam_types import _integer


@dataclass(frozen=True)
class K1Table:
    words: np.ndarray
    bucket_count: int
    count: int


def _distribution(lo,hi,salt):
    mask = (1<<64)-1
    def mix(x):
        x = ((x^(x>>30))*0xbf58476d1ce4e5b9)&mask
        x = ((x^(x>>27))*0x94d049bb133111eb)&mask
        return x^(x>>31)
    return mix(lo^(((hi<<32)|(hi>>32))&mask)^salt^mix((hi+0x9e3779b97f4a7c15)&mask))


def prepare_k1_table(hashes,*,bucket_count):
    """Ordered neighborhood Hash128 columns, packed using source first-free order.

    The neighborhood builder supplies the entries; inverse BFS and CPU suffix
    reconstruction are separate. This implements the source fixed-arena mode:
    failure raises, never grows the arena or drops a hash. Rows are fingerprint
    then Hash128 words0..3; unused slots and allocation padding remain zero.
    """
    bucket_count = _integer(bucket_count,1,2**28)
    if bucket_count&(bucket_count-1):
        raise ValueError('K1 bucket count must be a power of two')
    hashes = np.asarray(hashes)
    if hashes.ndim != 2 or hashes.shape[0] != 4 or hashes.dtype != np.uint32:
        raise ValueError('K1 entries must be uint32 Hash128 columns')
    slots = bucket_count*4
    if hashes.shape[1] > slots:
        raise ValueError('K1 entries do not fit fixed bucket arena')
    words = np.zeros((5,max(128,slots)),np.uint32)
    for column in range(hashes.shape[1]):
        lo = int(hashes[0,column])|(int(hashes[1,column])<<32)
        hi = int(hashes[2,column])|(int(hashes[3,column])<<32)
        mixed = _distribution(lo,hi,0xa4093822299f31d0)
        fingerprint = ((mixed^(mixed>>32))&0xffffffff) or 1
        placed = False
        for salt in (0x082efa98ec4e6c89,0x452821e638d01377):
            base = (_distribution(lo,hi,salt)&(bucket_count-1))*4
            for slot in range(base,base+4):
                if words[0,slot] == 0:
                    words[0,slot] = fingerprint
                    words[1:,slot] = hashes[:,column]
                    placed = True
                    break
            if placed:
                break
        if not placed:
            raise ValueError('K1 entries do not fit fixed bucket arena')
    words.flags.writeable = False
    return K1Table(words,bucket_count,hashes.shape[1])
