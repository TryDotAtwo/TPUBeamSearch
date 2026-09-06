import pytest


def test_three_layouts_share_prefix_without_overlapping_stream_survivors():
    from tpu_beam_search.beam_scratch import plan_scratch
    p = plan_scratch(common_bytes=513,select_temp_bytes=1,materialize_temp_bytes=1025,
                     stream_temp_bytes=4097,stream_persistent_bytes=513)
    assert p.common == (0,1024)
    assert p.select_temp == (1024,512)
    assert p.materialize_temp == (1024,1536)
    assert p.final_budget_bytes == 2560
    assert p.stream_temp == (0,4608)
    assert p.stream_persistent == (4608,1024)
    assert p.phase_bytes == (5632,1536,2560)
    assert p.pool_bytes == 5632
    assert p.pool_shape == (11,128)  # uint32;512-byte address granule


def test_small_stream_temporaries_do_not_enter_reserved_final_prefix():
    from tpu_beam_search.beam_scratch import plan_scratch
    p = plan_scratch(common_bytes=1024,select_temp_bytes=512,materialize_temp_bytes=0,
                     stream_temp_bytes=512,stream_persistent_bytes=512)
    assert p.stream_persistent == (1536,512)
    assert p.pool_bytes == 2048
    assert p.phase_bytes == (2048,1536,1024)


@pytest.mark.parametrize('bad',[-1,True,1.5,2**63])
def test_invalid_geometry_rejected_before_allocation(bad):
    from tpu_beam_search.beam_scratch import plan_scratch
    with pytest.raises(ValueError):
        plan_scratch(common_bytes=bad,select_temp_bytes=0,materialize_temp_bytes=0,
                     stream_temp_bytes=0,stream_persistent_bytes=0)
