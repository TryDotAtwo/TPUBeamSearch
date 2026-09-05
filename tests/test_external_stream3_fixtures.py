import importlib.util
import os
import numpy as np
import pytest


@pytest.mark.skipif(not os.environ.get('BEAM_SOURCE_ORACLE'), reason='source oracle required')
def test_fixture_oracle_empty_has_aligned_zero_counts_and_neutral_records():
    assert importlib.util.find_spec('benchmarks.build_external_stream3_fixtures') is not None
    from benchmarks.build_external_stream3_fixtures import query_oracle
    result = query_oracle(os.environ['BEAM_SOURCE_ORACLE'],
        np.zeros((8,256),np.uint32),np.zeros((1,256),np.uint32),0,0xffffffff,3)
    assert len(result) == 5
    for value in result[:2]:
        expected = np.zeros((8,256),np.uint32)
        expected[6] = np.uint32(0xffffffff)
        np.testing.assert_array_equal(value,expected)
    for value in result[2:]:
        np.testing.assert_array_equal(value,np.zeros((1,128),np.uint32))
