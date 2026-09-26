import pytest

from tpu_beam_search.beam_dma_ring import RemoteDmaRingModel


def test_nonzero_transfer_requires_published_destination_readiness():
    ring = RemoteDmaRingModel(slot_count=2)
    with pytest.raises(RuntimeError, match='not ready'):
        ring.start(epoch=0, slot=0, count=7)


def test_source_and_destination_reuse_have_separate_gates():
    ring = RemoteDmaRingModel(slot_count=2)
    ring.publish_ready(epoch=0, slot=0)
    ring.start(epoch=0, slot=0, count=7)
    assert not ring.can_reuse_source(0)
    assert not ring.can_reuse_destination(0)
    ring.wait_send(epoch=0, slot=0)
    assert ring.can_reuse_source(0)
    assert not ring.can_reuse_destination(0)
    ring.wait_recv(epoch=0, slot=0)
    ring.consume(epoch=0, slot=0)
    assert not ring.can_reuse_destination(0)
    ring.ack(epoch=0, slot=0)
    assert ring.can_reuse_destination(0)


def test_zero_count_epoch_participates_without_dma_or_semaphore_wait():
    ring = RemoteDmaRingModel(slot_count=2)
    ring.publish_ready(epoch=0, slot=0)
    ring.start(epoch=0, slot=0, count=0)
    assert ring.epoch_complete(0)
    assert ring.can_reuse_source(0)
    assert ring.can_reuse_destination(0)


def test_multiple_wraps_require_ack_before_slot_reuse():
    ring = RemoteDmaRingModel(slot_count=2)
    for epoch in range(4):
        slot = epoch % 2
        ring.publish_ready(epoch=epoch, slot=slot)
        ring.start(epoch=epoch, slot=slot, count=epoch + 1)
        ring.wait_send(epoch=epoch, slot=slot)
        ring.wait_recv(epoch=epoch, slot=slot)
        ring.consume(epoch=epoch, slot=slot)
        if epoch == 0:
            with pytest.raises(RuntimeError, match='not acknowledged'):
                ring.publish_ready(epoch=2, slot=0)
        ring.ack(epoch=epoch, slot=slot)
        assert ring.epoch_complete(epoch)


def test_completed_epoch_stays_complete_after_slot_wrap():
    ring = RemoteDmaRingModel(slot_count=2)
    ring.publish_ready(epoch=0, slot=0)
    ring.start(epoch=0, slot=0, count=0)
    ring.publish_ready(epoch=1, slot=1)
    ring.start(epoch=1, slot=1, count=0)
    ring.publish_ready(epoch=2, slot=0)
    assert ring.epoch_complete(0)
    assert ring.epoch_complete(1)
    assert not ring.epoch_complete(2)


def test_duplicate_start_and_duplicate_epoch_are_rejected():
    ring = RemoteDmaRingModel(slot_count=2)
    ring.publish_ready(epoch=0, slot=0)
    ring.start(epoch=0, slot=0, count=1)
    with pytest.raises(RuntimeError, match='already started'):
        ring.start(epoch=0, slot=0, count=1)
    ring.wait_send(epoch=0, slot=0)
    ring.wait_recv(epoch=0, slot=0)
    ring.consume(epoch=0, slot=0)
    ring.ack(epoch=0, slot=0)
    with pytest.raises(RuntimeError, match='epoch'):
        ring.publish_ready(epoch=0, slot=0)


def test_stale_wait_and_ack_cannot_mutate_reused_slot():
    ring = RemoteDmaRingModel(slot_count=2)
    ring.publish_ready(epoch=0, slot=0)
    ring.start(epoch=0, slot=0, count=0)
    ring.publish_ready(epoch=1, slot=1)
    ring.start(epoch=1, slot=1, count=0)
    ring.publish_ready(epoch=2, slot=0)
    for operation in (ring.wait_send, ring.wait_recv, ring.ack):
        with pytest.raises(RuntimeError, match='epoch'):
            operation(epoch=0, slot=0)


def test_free_slot_can_advance_while_another_epoch_remains_unfinished():
    ring = RemoteDmaRingModel(slot_count=2)
    ring.publish_ready(epoch=0, slot=0)
    ring.start(epoch=0, slot=0, count=1)
    ring.publish_ready(epoch=1, slot=1)
    ring.start(epoch=1, slot=1, count=0)
    for epoch in range(2, 102):
        ring.publish_ready(epoch=epoch, slot=1)
        ring.start(epoch=epoch, slot=1, count=0)
        assert ring.epoch_complete(epoch)
    assert not ring.epoch_complete(0)
    ring.wait_send(epoch=0, slot=0)
    ring.wait_recv(epoch=0, slot=0)
    ring.consume(epoch=0, slot=0)
    ring.ack(epoch=0, slot=0)
    assert all(ring.epoch_complete(epoch) for epoch in range(102))
