import numpy as np
import pytest


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


@pytest.mark.parametrize('fault',['none','duplicate','missing','out_of_range','late_error'])
def test_routed_fixture_feeds_pallas_whole_depth_coverage(fault):
    import jax.numpy as jnp
    from types import SimpleNamespace
    from benchmarks.beam_response_epoch_fixture import publication_fixtures,expected_epoch
    from tpu_beam_search.beam_final_response import pallas_unpack_response
    from tpu_beam_search.beam_final_agreement import make_final_coverage_agreement
    _,inputs,counts=next(x for x in publication_fixtures() if x[0]=='self')
    wire,requests,control,validation=inputs
    assembled=np.zeros((256,128),np.uint8)
    cursor=0
    for epoch in range(3):
        received,status=expected_epoch(wire,requests,control[:,0,0],np.zeros(8,bool),epoch)
        n=int(status[0,0,0])
        assembled[cursor:cursor+n]=received[0,:n]
        cursor+=n
    assert cursor==129
    if fault=='duplicate':
        assembled[128,120:124]=assembled[0,120:124]
    elif fault=='out_of_range':
        assembled[128,120:124]=list((129).to_bytes(4,'little'))
    _,targets=pallas_unpack_response(jnp.asarray(assembled),state_len=120,interpret=True)
    valid=jnp.asarray((np.arange(256)<cursor).astype(np.uint32)[None,:])
    if fault=='missing':
        valid=valid.at[0,128].set(0)
    prior=jnp.zeros((1,128),jnp.uint32).at[0,0].set(int(fault=='late_error'))
    error,_=make_final_coverage_agreement(SimpleNamespace(size=1),interpret=True)(
        targets,valid,jnp.asarray(counts[:1]),prior)
    assert bool(np.asarray(error)[0,0])==(fault!='none')
