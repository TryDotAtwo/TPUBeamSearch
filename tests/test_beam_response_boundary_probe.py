import numpy as np


def test_host_adapter_and_group_oracles_preserve_129_shuffled_records():
    from benchmarks.beam_response_boundary_probe import host_inputs, host_grouped
    wire, planes, ranks, valid = host_inputs()
    assert wire.shape == (8, 2048, 128)
    assert planes.shape == (8, 32, 2048)
    np.testing.assert_array_equal(planes[0, :, :129].T,
        wire[0, :129].copy().view('<u4').reshape(129, 32))
    expected = host_grouped(planes, ranks, valid)
    assert expected.shape == (8, 35, 2048)
    for device in range(8):
        assert valid[device, 0].sum() == 129
        np.testing.assert_array_equal(expected[device, :32, :129], planes[device, :, :129])
        np.testing.assert_array_equal(expected[device, 32, :129], device)
        np.testing.assert_array_equal(expected[device, 33], np.arange(2048))
        np.testing.assert_array_equal(expected[device, 34, :129], 1)
        np.testing.assert_array_equal(expected[device, 34, 129:], 0)


def test_boundary_assessment_requires_shape_dtype_hash_and_per_device_exactness():
    from benchmarks.beam_response_boundary_probe import assess
    expected = np.zeros((8, 32, 128), np.uint32)
    assert assess(expected, expected)['exact']
    corrupted = expected.copy()
    corrupted[5, 2, 9] = np.uint32(0x00ff0000)
    result = assess(corrupted, expected)
    assert not result['exact']
    assert result['mismatches'] == [0, 0, 0, 0, 0, 1, 0, 0]
    assert result['first_mismatches'] == [[5, 2, 9, 16711680, 0]]
    assert not assess(expected.astype(np.int32), expected)['exact']
    assert not assess(expected[:, :, :1], expected)['exact']
