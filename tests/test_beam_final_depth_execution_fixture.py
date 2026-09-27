import numpy as np


def test_physical_depth_fixture_routes_each_request_to_a_distinct_remote_parent():
    from benchmarks.beam_final_depth_execution import make_inputs,expected,WORLD,STATE_LEN
    a,b,controls,threshold,beam,prior,parents,generators,state=make_inputs()
    frontier,history=expected()
    assert a.shape==(WORLD,1,8,128) and parents.shape==(WORLD,128,256)
    assert controls.shape==(WORLD,1,8,128)
    assert all(int(a[rank,0,7,0]>>16)==(rank+1)%WORLD for rank in range(WORLD))
    assert all(int(frontier[rank,0,0])==int(parents[(rank+1)%WORLD,0,0])
               for rank in range(WORLD))
    assert all(not np.asarray(frontier[rank,0,STATE_LEN:]).any()
               for rank in range(WORLD))
    assert all(int(history[rank,0,2,0]>>16)==(rank+1)%WORLD
               for rank in range(WORLD))
    assert all(int(beam[rank,0,0])==WORLD and int(threshold[rank,0])==5
               for rank in range(WORLD))
    assert all(not np.asarray(value).any() for value in
               (b[:,:,:6],prior,state.error))
