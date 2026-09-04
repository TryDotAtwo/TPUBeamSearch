"""Host-side state contract for the bounded TPU remote-DMA ring.

This model is a race/ordering oracle for the later Pallas kernel.  It performs
no transfer and is not TPU execution evidence.
"""
from dataclasses import dataclass


@dataclass
class _Slot:
    epoch: int | None = None
    ready: bool = False
    started: bool = False
    send_done: bool = False
    recv_done: bool = False
    consumed: bool = False
    acknowledged: bool = True
    count: int = 0


class RemoteDmaRingModel:
    """Checks source/destination lifetimes for a fixed set of transfer slots."""

    def __init__(self, *, slot_count: int):
        if slot_count < 2:
            raise ValueError('slot_count must be at least two')
        self._slots = [_Slot() for _ in range(slot_count)]
        self._epochs: dict[int, _Slot] = {}

    def _slot(self, slot: int) -> _Slot:
        if not 0 <= slot < len(self._slots):
            raise ValueError('slot out of range')
        return self._slots[slot]

    @staticmethod
    def _require_epoch(state: _Slot, epoch: int) -> None:
        if state.epoch != epoch:
            raise RuntimeError('epoch does not own slot')

    def publish_ready(self, *, epoch: int, slot: int) -> None:
        state = self._slot(slot)
        if state.epoch is not None and not state.acknowledged:
            raise RuntimeError('previous use is not acknowledged')
        state.epoch = epoch
        state.ready = True
        state.started = state.send_done = state.recv_done = state.consumed = False
        state.acknowledged = False
        state.count = 0
        self._epochs[epoch] = state

    def start(self, *, epoch: int, slot: int, count: int) -> None:
        state = self._slot(slot)
        if not state.ready:
            raise RuntimeError('destination is not ready')
        self._require_epoch(state, epoch)
        if count < 0:
            raise ValueError('count must be nonnegative')
        state.count = count
        state.started = True
        if count == 0:
            # The rank still participates in the epoch, but must not wait for a
            # DMA that no sender starts. No semaphore receives phantom bytes.
            state.send_done = state.recv_done = state.consumed = state.acknowledged = True

    def wait_send(self, *, epoch: int, slot: int) -> None:
        state = self._slot(slot)
        self._require_epoch(state, epoch)
        if not state.started or state.count == 0:
            raise RuntimeError('no nonzero DMA send to wait for')
        state.send_done = True

    def wait_recv(self, *, epoch: int, slot: int) -> None:
        state = self._slot(slot)
        self._require_epoch(state, epoch)
        if not state.started or state.count == 0:
            raise RuntimeError('no nonzero DMA receive to wait for')
        state.recv_done = True

    def consume(self, *, epoch: int, slot: int) -> None:
        state = self._slot(slot)
        self._require_epoch(state, epoch)
        if not state.recv_done:
            raise RuntimeError('receive is not complete')
        state.consumed = True

    def ack(self, *, epoch: int, slot: int) -> None:
        state = self._slot(slot)
        self._require_epoch(state, epoch)
        if not state.send_done or not state.consumed:
            raise RuntimeError('send and consumption must complete before ack')
        state.acknowledged = True

    def can_reuse_source(self, slot: int) -> bool:
        state = self._slot(slot)
        return state.epoch is None or state.send_done or state.acknowledged

    def can_reuse_destination(self, slot: int) -> bool:
        state = self._slot(slot)
        return state.epoch is None or state.acknowledged

    def epoch_complete(self, epoch: int) -> bool:
        state = self._epochs.get(epoch)
        return state is not None and state.started and state.acknowledged
