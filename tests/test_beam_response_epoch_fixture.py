import numpy as np


def test_fixture_matrix_has_boundary_counts_and_poisoned_tails():
    from benchmarks.beam_response_epoch_fixture import fixtures
    cases=dict(fixtures())
    assert {'empty','self','cycle','one_to_all','all_to_one','uneven',
            'materialization_error','receive_error','bad_rank','reserved','recovery'}<=cases.keys()
    wire,requests,control,validation=cases['cycle']
    assert wire.shape==(8,2048,128)
    assert np.all(control[:,0,0]==129)
    for source in range(8):
        assert np.all((requests[source,3,:129]&65535)==(source+1)%8)
        assert np.all(requests[source,:,129:]==0xffffffff)
    assert not validation.any()
    assert not np.array_equal(cases['uneven'][0],cases['recovery'][0])
    _,requests,control,_=cases['uneven']
    counts=[]
    for source in range(8):
        live=requests[source,3,:int(control[source,0,0])]&65535
        counts.extend(np.count_nonzero(live==destination) for destination in range(8))
    assert {0,1,127,128,129}<=set(counts)


def test_uneven_input_requires_real_grouping_and_keeps_payload_identity():
    from benchmarks.beam_response_epoch_fixture import fixtures
    wire,requests,control,_=dict(fixtures())['uneven']
    for source in range(8):
        n=int(control[source,0,0])
        destinations=(requests[source,3,:n]&65535).astype(np.int64)
        assert np.any(np.diff(destinations)<0), 'already grouped input hides grouping bugs'
        original_slots=requests[source,0,:n].astype(np.uint32)
        offsets=original_slots[:,None]*128+np.arange(128,dtype=np.uint32)[None,:]
        expected=((offsets//7+offsets%13+source*29+5*17)%256).astype(np.uint8)
        np.testing.assert_array_equal(wire[source,:n],expected)


def test_literal_source_order_chunk_boundary_and_global_failure():
    from benchmarks.beam_response_epoch_fixture import expected_epoch
    wire=np.zeros((8,256,128),np.uint8)
    requests=np.zeros((8,4,256),np.uint32)
    counts=np.zeros(8,np.uint32)
    errors=np.zeros(8,np.uint32)
    counts[0],counts[7]=129,1
    requests[0,3,:129]=3
    requests[7,3,0]=3
    wire[0,:128]=11
    wire[0,128]=22
    wire[7,0]=77
    first,status=expected_epoch(wire,requests,counts,errors,0)
    assert status[3,0,0]==129
    np.testing.assert_array_equal(first[3,:128],np.full((128,128),11,np.uint8))
    assert np.all(first[3,128]==77) and not first[3,129:].any()
    second,status=expected_epoch(wire,requests,counts,errors,1)
    assert status[3,0,0]==1 and np.all(second[3,0]==22)
    assert not second[3,1:].any()
    errors[2]=7
    empty,status=expected_epoch(wire,requests,counts,errors,2)
    assert not empty.any() and not status[:,0].any()
    np.testing.assert_array_equal(status[:,1,0],np.ones(8,np.uint32))
