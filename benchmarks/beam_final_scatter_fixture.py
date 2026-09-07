"""Literal NumPy input/output fixtures shared by standalone and chained gates."""
import numpy as np


def make_case(count,*,failure=None):
    if count not in (0,1,127,128,129) or failure not in (None,'count_overflow','target_overflow'):
        raise ValueError('invalid scatter fixture')
    parents = (np.arange(7*128).reshape(7,128)%256).astype(np.uint8)
    perm = np.arange(127,-1,-1,dtype=np.int32)[None,:]
    requests = np.zeros((4,256),np.uint32)
    requests[0,:count] = np.arange(count)%7
    requests[2,:count] = (np.arange(count)*3)%256
    wire = np.zeros((256,128),np.uint8)
    frontier = np.full((256,128),173,np.uint8)
    expected = frontier.copy()
    for i in range(count):
        target = int(requests[2,i])
        wire[i,:120] = parents[i%7,perm[0,:120]]
        wire[i,120:124] = np.frombuffer(target.to_bytes(4,'little'),np.uint8)
        expected[target,:120] = wire[i,:120]
        expected[target,120:] = 0
    n = np.array([count],np.uint32)
    if failure == 'count_overflow':
        n[0] = 257
    elif failure == 'target_overflow':
        if not count:
            raise ValueError('target overflow needs a live record')
        wire[0,120:124] = np.frombuffer((256).to_bytes(4,'little'),np.uint8)
    if failure:
        expected = frontier.copy()
    return dict(name=f'count{count}_{failure or "valid"}',
                inputs=(parents,perm,requests,n,frontier,wire),expected=expected,
                expected_error=int(failure is not None),failure=failure)
