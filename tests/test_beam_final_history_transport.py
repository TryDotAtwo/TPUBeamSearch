"""History routing composition with simulated links, not physical TPU DMA."""
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('late_corruption',[False,True])
def test_history_transport_preserves_parent64_route_and_target_across_sources(late_corruption):
    # Catches parent-high truncation, source/destination substitution and
    # publication by receive order rather than target index.
    from tpu_beam_search.beam_final_history import pallas_final_history_records
    from tpu_beam_search.beam_final_group import pallas_group_final_records
    from tpu_beam_search.beam_final_intervals import pallas_final_rank_intervals
    from tpu_beam_search.beam_final_chunk import pallas_pack_final_chunk
    from tpu_beam_search.beam_final_receive import pallas_compact_final_received
    from tpu_beam_search.beam_history import RankHistoryStore, HistoryEntry, decode_history_soa

    rows = (
        ((7, 0xffffffff, 0x00000217, 1, 2), (9, 0x80000000, 0x00000301, 0, 0)),
        ((11, 0x12345678, 0x00010102, 0, 2),),
    )
    sent, lengths = [], []
    for source_rows in rows:
        meta = np.full((8,128),0xdeadbeef,np.uint32)
        targets = np.zeros((1,128),np.uint32)
        ranks = np.full((1,128),0xffffffff,np.uint32)
        valid = np.zeros((1,128),np.uint32)
        for slot,(low,high,route,target,rank) in zip((3,127),source_rows):
            meta[4,slot],meta[5,slot],meta[7,slot] = low,high,route
            targets[0,slot],ranks[0,slot],valid[0,slot] = target,rank,1
        records = pallas_final_history_records(*map(jnp.asarray,(meta,targets,valid)),interpret=True)
        grouped = pallas_group_final_records(records,jnp.asarray(ranks),jnp.asarray(valid),interpret=True)
        intervals = pallas_final_rank_intervals(grouped[5:6],grouped[7:8],world_size=3,interpret=True)
        chunks,control = pallas_pack_final_chunk(grouped[:5],intervals,jnp.array([0],jnp.uint32),
                                               world_size=3,interpret=True)
        assert not np.asarray(control[:,1]).any()
        sent.append(np.asarray(chunks))
        lengths.append(np.asarray(control[:,0:1]))

    expected = (
        (HistoryEntry(0x8000000000000009,0x00000301),),
        (),
        (HistoryEntry(0x123456780000000b,0x00010102),HistoryEntry(0xffffffff00000007,0x00000217)),
    )
    store = RankHistoryStore(world_size=3)
    received = []
    for rank,want in enumerate(expected):
        # Only the link is simulated: source-major snapshots are delivered
        # unchanged to the actual Pallas receive compaction.
        snapshots = jnp.asarray(np.stack([x[rank] for x in sent]))
        counts = jnp.asarray(np.stack([x[rank] for x in lengths]))
        packed,control = pallas_compact_final_received(snapshots,counts,
            jnp.zeros((1,128),jnp.uint32),interpret=True)
        packed,control = map(np.asarray,(packed,control))
        assert int(control[1,0]) == 0
        assert int(control[0,0]) == len(want)
        assert not packed[:,len(want):].any()
        received.append(packed)
    def publish():
        store.append_all_rank_layer(
            [decode_history_soa(x,world_size=3,move_count=24) for x in received],
            target_counts=[len(x) for x in expected],depth=0)
    if late_corruption:
        received[-1]=received[-1].copy()
        saved=received[-1][2,1]
        received[-1][2,1]=3<<16  # impossible original source, at the last rank
        with pytest.raises(ValueError):
            publish()
        for rank in (0,2):
            with pytest.raises(IndexError):
                store.read_entry(rank,0,0)
        received[-1][2,1]=saved
    publish()
    for rank,want in enumerate(expected):
        assert tuple(store.read_entry(rank,0,i) for i in range(len(want))) == want
