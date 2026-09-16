import numpy as np


def test_destination_targets_cover_across_sources_and_epochs():
    from benchmarks import beam_response_epoch_fixture as fixture
    assert hasattr(fixture,'publication_fixtures')
    seen_names=[]
    for name,inputs,counts in fixture.publication_fixtures():
        seen_names.append(name)
        wire,requests,control,validation=inputs
        collected=[[] for _ in range(8)]
        for epoch in range(3):
            received,status=fixture.expected_epoch(wire,requests,control[:,0,0],
                (control[:,1,0]!=0)|(validation[:,0,0]!=0),epoch)
            assert not status[:,1].any()
            for rank in range(8):
                for row in received[rank,:int(status[rank,0,0])]:
                    collected[rank].append(int.from_bytes(row[120:124].tobytes(),'little'))
                    assert np.all(row[124:]==255)
        for rank in range(8):
            assert sorted(collected[rank])==list(range(int(counts[rank])))
    assert set(seen_names)=={'empty','self','cycle','one_to_all','all_to_one','uneven','recovery'}
