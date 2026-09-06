import numpy as np


def test_fixture_routes_source_to_destination_and_global_error():
    from benchmarks.beam_final_exchange_probe import fixtures
    cases=fixtures()
    assert len(cases)==16
    assert {name for name,_,_ in cases} >= {'round0_empty','round0_one_to_all','round0_all_to_one','round0_bad_count','round0_error'}
    for name,(payload,controls),(wire,counts,error) in cases:
        if 'bad_count' in name or name.endswith('_error'):
            assert not wire.any() and not counts.any()
            np.testing.assert_array_equal(error[:,0,0],1)
            continue
        assert not error.any()
        for destination in range(8):
            for source in range(8):
                count=int(controls[source,destination,0,0])
                assert counts[destination,source,0,0]==count
                np.testing.assert_array_equal(wire[destination,source,:,:count],payload[source,destination,:,:count])
                assert not wire[destination,source,:,count:].any()
