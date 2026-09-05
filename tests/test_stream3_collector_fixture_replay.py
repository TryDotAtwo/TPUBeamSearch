"""CPU interpreter with explicitly simulated network; NOT a physical RDMA test."""
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from tpu_beam_search.beam_external_sort import pallas_external_stream3
from tpu_beam_search.beam_stream3 import pallas_stream3_wire_slots
from tpu_beam_search.beam_collector import pallas_collect
from tpu_beam_search.beam_receive_batch import pallas_collect_received


def test_source_fixture_replays_all_ranks_through_simulated_exchange():
    path = Path(__file__).parent/'fixtures/stream3_collector/stream3_collector_128.npz'
    with np.load(path,allow_pickle=False) as data:
        inputs = {k:data[k] for k in data.files}
    wires,wire_counts,locals_out = [],[],[]
    for rank in range(8):
        local,remote,lc,sc,so = pallas_external_stream3(
            *(jnp.asarray(inputs[k][rank]) for k in ('words','payload','counts','thresholds')),
            local_rank=rank,world_size=8,interpret=True)
        locals_out.append(pallas_collect(
            jnp.asarray(inputs['a'][rank]),jnp.asarray(inputs['b'][rank]),local,
            jnp.asarray(inputs['controls'][rank]),lc,interpret=True)[:3])
        wire,count = pallas_stream3_wire_slots(remote,sc,so,local_rank=rank,world_size=8,interpret=True)
        wires.append(np.asarray(wire))
        wire_counts.append(np.asarray(count))
    for rank in range(8):
        received = np.zeros((9,8,128),np.uint32)
        received[:2] = np.uint32(0xdeadbeef)
        counts = np.zeros((7,128),np.uint32)
        for epoch in range(7):
            source = (rank-epoch-1)%8
            received[2+epoch] = wires[source][epoch]
            counts[epoch] = wire_counts[source][epoch]
        rank_control = np.zeros((1,128),np.uint32)
        rank_control[0,0] = rank
        a,b,controls = locals_out[rank]
        actual = pallas_collect_received(a,b,jnp.asarray(received),controls,
            jnp.asarray(counts),jnp.asarray(rank_control),interpret=True)
        for i,value in enumerate(actual):
            np.testing.assert_array_equal(value,inputs[f'expected_{i}'][rank],err_msg=f'rank={rank} output={i}')
