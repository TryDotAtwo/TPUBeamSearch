"""Host reference for serialized collector reservation, not TPU execution.

One logical shard has two resident siblings. This reference does not publish
dirty counts to concurrent consumers; a device implementation must complete
record stores before committing counts. No group splitting or spill fallback.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Reservation:
    buffer: int | None
    offset: int
    dirty: tuple[int, int]
    fatal_overflow: bool


def reserve_group(*, capacity, clean, dirty, processing, current, amount):
    """Choose current if writable and fitting, otherwise its sole sibling.

    Inputs are validated host scalars; fatal overflow leaves counts untouched.
    Empty groups are no-ops even if both buffers are full or processing.
    """
    if (not isinstance(capacity, int) or not 0 < capacity <= 0xffffffff
            or not isinstance(amount, int) or not 0 <= amount <= 0xffffffff
            or current not in (0,1)
            or len(clean) != 2 or len(dirty) != 2 or len(processing) != 2):
        raise ValueError('invalid collector geometry')
    if any(not isinstance(x,int) or x < 0 for x in (*clean,*dirty)):
        raise ValueError('invalid resident counts')
    if any(clean[i]+dirty[i] > capacity for i in (0,1)):
        raise ValueError('resident counts exceed capacity')
    if amount == 0:
        return Reservation(None,0,tuple(dirty),False)
    for index in (current,1-current):
        used = clean[index]+dirty[index]
        if not processing[index] and amount <= capacity-used:
            updated = list(dirty)
            updated[index] += amount
            return Reservation(index,used,tuple(updated),False)
    return Reservation(None,0,tuple(dirty),True)
