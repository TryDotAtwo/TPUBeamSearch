import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize('initial,stop_on_found', [(0,False),(126,True),(130,True)])
def test_bounded_solved_collection_counts_attempts_and_preserves_existing(initial,stop_on_found):
    from tpu_beam_search.beam_solved_collect import pallas_collect_solved
    records = np.arange(1280,dtype=np.uint32).reshape(10,128)
    arena = np.full((10,128),77,np.uint32)
    control = np.zeros((4,128),np.uint32)
    control[0,0] = initial
    flags = np.zeros((1,128),np.uint32)
    flags[0,[0,3,7,127]] = 1
    want_arena,want_control = arena.copy(),control.copy()
    for index in (0,3,7,127):
        offset = int(want_control[0,0])
        want_control[0,0] += 1
        if offset < 128:
            want_arena[:,offset] = records[:,index]
        else:
            want_control[1,0] = 1
        if not want_control[2,0]:
            want_control[2,0] = 1
            if stop_on_found:
                want_control[3,0] = 1
    actual = pallas_collect_solved(*map(jnp.asarray,(arena,control,records,flags)),
        stop_on_found=stop_on_found,interpret=True)
    for got,want in zip(actual,(want_arena,want_control),strict=True):
        np.testing.assert_array_equal(got,want)


def test_empty_and_previously_found_do_not_republish_stop():
    from tpu_beam_search.beam_solved_collect import pallas_collect_solved
    arena = np.full((10,128),0xffffffff,np.uint32)
    control = np.zeros((4,128),np.uint32)
    control[2,0] = 1
    records = np.zeros((10,128),np.uint32)
    flags = np.zeros((1,128),np.uint32)
    empty = pallas_collect_solved(*map(jnp.asarray,(arena,control,records,flags)),
        stop_on_found=True,interpret=True)
    np.testing.assert_array_equal(empty[0],arena)
    np.testing.assert_array_equal(empty[1],control)
    flags[0,0] = 1
    actual = pallas_collect_solved(*map(jnp.asarray,(arena,control,records,flags)),
        stop_on_found=True,interpret=True)
    assert int(actual[1][0,0]) == 1
    assert int(actual[1][3,0]) == 0
