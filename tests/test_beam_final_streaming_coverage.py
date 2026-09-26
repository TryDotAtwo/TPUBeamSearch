import numpy as np
import jax.numpy as jnp
import pytest


def update(seen,control,values,logical=3):
    from tpu_beam_search.beam_final_streaming_coverage import pallas_mark_final_targets
    targets = np.full((1,128),0xffffffff,np.uint32)
    targets[0,:len(values)] = values
    return pallas_mark_final_targets(seen,control,jnp.asarray(targets),
        jnp.array([len(values)],jnp.uint32),jnp.array([logical],jnp.uint32),interpret=True)


def test_streaming_coverage_tracks_out_of_order_targets_across_epochs():
    from tpu_beam_search.beam_final_streaming_coverage import pallas_finish_final_targets
    seen=jnp.zeros((1,256),jnp.uint32)
    control=jnp.zeros((2,128),jnp.uint32)
    seen,control=update(seen,control,[2,0])
    assert int(pallas_finish_final_targets(seen,control,jnp.array([3],jnp.uint32),interpret=True)[0,0])==1
    seen,control=update(seen,control,[])
    seen,control=update(seen,control,[1])
    np.testing.assert_array_equal(np.asarray(seen)[0,:4],[1,1,1,0])
    assert int(control[0,0])==3 and int(control[1,0])==0
    assert not np.asarray(pallas_finish_final_targets(seen,control,jnp.array([3],jnp.uint32),interpret=True)).any()


@pytest.mark.parametrize('values',[[1],[2,2],[3],[0xffffffff]])
def test_duplicate_or_out_of_range_target_poison_is_sticky(values):
    from tpu_beam_search.beam_final_streaming_coverage import pallas_finish_final_targets
    seen,control=update(jnp.zeros((1,128),jnp.uint32),jnp.zeros((2,128),jnp.uint32),[1])
    seen,control=update(seen,control,values)
    assert int(control[1,0])==1
    saved=np.asarray(seen).copy()
    seen,control=update(seen,control,[0,2])
    np.testing.assert_array_equal(seen,saved)
    assert int(pallas_finish_final_targets(seen,control,jnp.array([3],jnp.uint32),interpret=True)[0,0])==1


def test_streaming_coverage_rejects_count_overflow_without_writes():
    from tpu_beam_search.beam_final_streaming_coverage import pallas_mark_final_targets
    seen,control=pallas_mark_final_targets(jnp.zeros((1,128),jnp.uint32),
        jnp.zeros((2,128),jnp.uint32),jnp.zeros((1,128),jnp.uint32),
        jnp.array([129],jnp.uint32),jnp.array([3],jnp.uint32),interpret=True)
    assert not np.asarray(seen).any()
    assert int(control[1,0])==1 and int(control[0,0])==0


def test_target_mark_dma_crosses_tile_boundary_without_touching_neighbors():
    seen,control=update(jnp.zeros((1,256),jnp.uint32),
        jnp.zeros((2,128),jnp.uint32),[128,127],logical=129)
    expected=np.zeros((1,256),np.uint32)
    expected[0,127:129]=1
    np.testing.assert_array_equal(seen,expected)
    assert int(control[0,0])==2 and int(control[1,0])==0


def test_empty_frontier_passes_but_corrupted_marks_do_not():
    from tpu_beam_search.beam_final_streaming_coverage import pallas_finish_final_targets
    seen,control=update(jnp.zeros((1,128),jnp.uint32),
        jnp.zeros((2,128),jnp.uint32),[],logical=0)
    assert not np.asarray(pallas_finish_final_targets(seen,control,jnp.array([0],jnp.uint32),interpret=True)).any()
    seen=seen.at[0,127].set(1)
    assert int(pallas_finish_final_targets(seen,control,jnp.array([0],jnp.uint32),interpret=True)[0,0])==1


def test_serial_mark_dma_detects_same_tile_duplicate_with_race_checks():
    from jax.experimental.pallas import tpu as pltpu
    from tpu_beam_search.beam_final_streaming_coverage import pallas_mark_final_targets
    targets=jnp.zeros((1,128),jnp.uint32).at[0,:3].set(jnp.array([7,8,7],jnp.uint32))
    seen,control=pallas_mark_final_targets(jnp.zeros((1,128),jnp.uint32),
        jnp.zeros((2,128),jnp.uint32),targets,jnp.array([3],jnp.uint32),
        jnp.array([9],jnp.uint32),interpret=pltpu.InterpretParams(detect_races=True))
    expected=np.zeros((1,128),np.uint32)
    expected[0,7:9]=1
    np.testing.assert_array_equal(seen,expected)
    assert int(control[0,0])==2 and int(control[1,0])==1
