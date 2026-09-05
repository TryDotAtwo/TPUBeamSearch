import numpy as np


def test_wire_and_reduction_recovery_are_matched_nonzero_first():
    from benchmarks.beam_s5_request_probe import recovery_fixtures
    wire = recovery_fixtures('wire')
    reduction = recovery_fixtures('reduction')
    assert len(wire) == len(reduction) == 8
    assert np.any(wire[0][1]) and not np.any(wire[1][1])
    for (_,source,expected),(_,summands,total) in zip(wire,reduction,strict=True):
        np.testing.assert_array_equal(expected,summands)
        for rank in range(8):
            for offset in range(8):
                np.testing.assert_array_equal(expected[rank,2*offset:2*offset+2],source[(rank-offset)%8])
        values = summands[:,0::2].astype(np.uint64)+(summands[:,1::2].astype(np.uint64)<<np.uint64(32))
        summed = values.sum(axis=1,dtype=np.uint64)
        np.testing.assert_array_equal(total[:,0],summed.astype(np.uint32))
        np.testing.assert_array_equal(total[:,1],(summed>>np.uint64(32)).astype(np.uint32))
