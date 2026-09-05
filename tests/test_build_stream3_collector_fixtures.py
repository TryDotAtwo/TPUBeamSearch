import importlib.util
import os
import numpy as np
import pytest


@pytest.mark.skipif(not os.environ.get('BEAM_ROUTE_ORACLE'),reason='route oracle not configured')
@pytest.mark.parametrize('amount',[128,140])
def test_reference_admission_uses_full_group_and_preserves_failed_buffers(amount):
    assert importlib.util.find_spec('benchmarks.build_stream3_collector_fixtures') is not None
    from benchmarks.build_stream3_collector_fixtures import reference_admit
    a = np.zeros((1,8,128),np.uint32)
    b = a.copy()
    c = np.zeros((1,8,128),np.uint32)
    records = np.arange(8*amount,dtype=np.uint32).reshape(8,amount)
    aa,bb,cc,fatal = reference_admit(os.environ['BEAM_ROUTE_ORACLE'],a,b,c,records)
    expected = a.copy()
    expected_control = c.copy()
    if amount == 128:
        expected[0] = records
        expected_control[0,2,0] = 128
    else:
        expected_control[0,7,0] = 1
    np.testing.assert_array_equal(aa,expected)
    np.testing.assert_array_equal(bb,b)
    np.testing.assert_array_equal(cc,expected_control)
    assert fatal == (amount == 140)
