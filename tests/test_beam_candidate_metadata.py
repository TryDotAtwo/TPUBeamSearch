import jax.numpy as jnp
import numpy as np
import pytest


def case(moves=30,count=3):
    hashes = np.arange(4*128,dtype=np.uint32).reshape(4,128)
    scores = np.arange(128,dtype=np.uint32)[None]
    valid = (np.arange(128)<count*moves).astype(np.uint32)[None]
    return [jnp.asarray(hashes),jnp.asarray(scores),jnp.asarray(valid),
            jnp.array([count],jnp.uint32),jnp.array([0xfffffffe,17],jnp.uint32),
            jnp.array([4096],jnp.uint32)]


def test_metadata_keeps_parent_carry_payload_order_and_neutral_tail():
    from tpu_beam_search.beam_candidate_metadata import pallas_candidate_metadata
    args = case()
    words,payload,count,error = pallas_candidate_metadata(*args,move_count=30,interpret=True)
    want = np.zeros((8,128),np.uint32)
    want[6] = 0xffffffff
    want[:4,:90] = args[0][:,:90]
    for i in range(90):
        parent = (17<<32)+0xfffffffe+i//30
        want[4,i],want[5,i] = parent&0xffffffff,parent>>32
        want[6,i],want[7,i] = i,i%30
    np.testing.assert_array_equal(words,want)
    np.testing.assert_array_equal(np.asarray(payload)[0,:90],np.arange(4096,4186))
    assert not np.asarray(payload)[0,90:].any()
    assert int(count[0]) == 90 and int(error[0,0]) == 0


@pytest.mark.parametrize('fault',['parent_overflow','payload_overflow','valid_gap','valid_tail','count'])
def test_invalid_identity_batch_cannot_be_admitted_to_s3(fault):
    from tpu_beam_search.beam_candidate_metadata import pallas_candidate_metadata
    args = case(24,2)
    if fault == 'parent_overflow': args[4] = jnp.array([0xffffffff,0xffffffff],jnp.uint32)
    if fault == 'payload_overflow': args[5] = jnp.array([0xfffffff0],jnp.uint32)
    if fault == 'valid_gap': args[2] = args[2].at[0,1].set(0)
    if fault == 'valid_tail': args[2] = args[2].at[0,100].set(1)
    if fault == 'count': args[3] = jnp.array([6],jnp.uint32)
    _,_,count,error = pallas_candidate_metadata(*args,move_count=24,interpret=True)
    assert int(count[0]) == 0 and int(error[0,0]) > 0


def test_empty_batch_ignores_padding_address_wrap():
    from tpu_beam_search.beam_candidate_metadata import pallas_candidate_metadata
    args = case(24,0)
    args[4],args[5] = jnp.full((2,),0xffffffff,jnp.uint32),jnp.full((1,),0xffffffff,jnp.uint32)
    _,payload,count,error = pallas_candidate_metadata(*args,move_count=24,interpret=True)
    assert int(count[0]) == 0 and int(error[0,0]) == 0
    assert not np.asarray(payload).any()


@pytest.mark.parametrize('moves', [24, 30])
def test_real_stream2_hashes_keep_global_identity_across_candidate_tiles(moves):
    # Catches resetting parent/move/payload at the second 128-candidate tile.
    from tpu_beam_search.beam_stream2 import pallas_hash_goal
    from tpu_beam_search.beam_candidate_metadata import pallas_candidate_metadata
    parents = np.tile(np.array([[0, 1, 2, 0]], np.uint8), (7, 1))
    generators = np.tile(np.array([[1, 0, 2, 3]], np.int32), (moves, 1))
    table = np.arange(48, dtype=np.uint32).reshape(4, 4, 3)
    table[:, 3] = 0
    count = jnp.array([6], jnp.uint32)
    hashes, goals, valid = pallas_hash_goal(
        jnp.asarray(parents), jnp.asarray(generators),
        jnp.array([1, 0, 2, 0], jnp.uint8),
        jnp.asarray(table.reshape(4, -1)), count, interpret=True)
    capacity = hashes.shape[1]
    words, payload, admitted, error = pallas_candidate_metadata(
        hashes, jnp.full((1, capacity), 17, jnp.uint32), valid, count,
        jnp.array([0xfffffffd, 2], jnp.uint32),
        jnp.array([8192], jnp.uint32), move_count=moves, interpret=True)
    assert int(admitted[0]) == 6*moves and int(error[0, 0]) == 0
    expected_hash = table[:, 0, 1] ^ table[:, 1, 0] ^ table[:, 2, 2]
    for lane in (0, 127, 128, 6*moves-1):
        parent = (2 << 32) + 0xfffffffd + lane // moves
        np.testing.assert_array_equal(np.asarray(words)[:, lane], [
            *expected_hash, parent & 0xffffffff, parent >> 32, 17, lane % moves])
        assert int(payload[0, lane]) == 8192 + lane
        assert int(goals[0, lane]) == 1
    assert not np.asarray(valid)[0, 6*moves:].any()
    assert not np.asarray(payload)[0, 6*moves:].any()
