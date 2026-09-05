import importlib.util
import numpy as np


def test_request_probe_covers_every_requester_and_reset_epochs():
    assert importlib.util.find_spec('benchmarks.beam_s5_request_probe') is not None
    from benchmarks.beam_s5_request_probe import fixtures
    cases = fixtures()
    assert len(cases) == 20
    for index,(name,request,expected) in enumerate(cases):
        assert request.shape == expected.shape == (8,1,128)
        phase = index%10
        want = np.zeros_like(request)
        want[:,0,0] = int(phase != 0)
        np.testing.assert_array_equal(expected,want)
        assert np.count_nonzero(request[:,0,0]) == (0 if phase == 0 else 8 if phase == 9 else 1)
        if 1 <= phase <= 8:
            assert request[phase-1,0,0] == 1
        assert not np.any(request[:,:,1:])
        assert name.startswith(f'round{index//10}_')


def test_histogram_probe_has_exact_uint64_global_reference_and_zero_reset():
    from benchmarks import beam_s5_request_probe as module
    assert hasattr(module,'histogram_fixtures')
    cases = module.histogram_fixtures()
    assert len(cases) == 6
    for index,(_,local,expected) in enumerate(cases):
        assert local.shape == expected.shape == (8,2,256)
        values = local[:,0].astype(np.uint64)+(local[:,1].astype(np.uint64)<<np.uint64(32))
        summed = values.sum(axis=0,dtype=np.uint64)
        decoded = expected[:,0].astype(np.uint64)+(expected[:,1].astype(np.uint64)<<np.uint64(32))
        np.testing.assert_array_equal(decoded,np.broadcast_to(summed,(8,256)))
        if index%3 == 0:
            assert not local.any() and not expected.any()
        else:
            assert np.any(expected[:,1] != 0)
