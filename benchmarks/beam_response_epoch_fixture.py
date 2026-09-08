"""Host byte oracle from original requests; no production routing helpers."""
import numpy as np


def fixtures():
    """Runtime inputs for eight ranks; invalid tails are deliberately poisoned."""
    names=('empty','self','cycle','one_to_all','all_to_one','uneven',
           'materialization_error','receive_error','bad_rank','reserved','recovery')
    for case_id,name in enumerate(names):
        requests=np.full((8,4,2048),0xffffffff,np.uint32)
        wire=np.empty((8,2048,128),np.uint8)
        control=np.zeros((8,2,128),np.uint32)
        validation=np.zeros((8,2,128),np.uint32)
        for source in range(8):
            values=np.arange(2048*128,dtype=np.uint32).reshape(2048,128)
            wire[source]=((values//7+values%13+source*29+case_id*17)%256).astype(np.uint8)
            cursor=0
            for destination in range(8):
                if name=='empty':
                    count=0
                elif name=='self':
                    count=129 if destination==source else 0
                elif name=='cycle':
                    count=129 if destination==(source+1)%8 else 0
                elif name=='one_to_all':
                    count=129 if source==3 else 0
                elif name=='all_to_one':
                    count=129 if destination==5 else 0
                else:
                    count=(0,1,127,128,129)[(source+destination)%5]
                for slot in range(cursor,cursor+count):
                    requests[source,:,slot]=(slot,0,slot,destination | ((slot%24)<<16))
                cursor+=count
            control[source,0,0]=cursor
            # Move records and bytes together; do not pre-group the input that
            # is intended to validate device grouping and stable ordering.
            order=np.random.default_rng(4200+case_id*8+source).permutation(cursor)
            requests[source,:,:cursor]=requests[source,:,:cursor][:,order]
            wire[source,:cursor]=wire[source,:cursor][order]
        if name=='materialization_error':
            validation[2,0,0]=1
        elif name=='receive_error':
            control[2,1,0]=1
        elif name=='bad_rank':
            requests[2,3,0]=8
        elif name=='reserved':
            requests[2,3,0]|=np.uint32(1<<24)
        yield name,(wire,requests,control,validation)


def expected_epoch(wire,requests,counts,errors,epoch):
    ranks,capacity,width=wire.shape
    result=np.zeros((ranks,128*ranks,width),np.uint8)
    status=np.zeros((ranks,2,128),np.uint32)
    bad=bool(np.any(errors) or np.any(counts>capacity))
    for source in range(ranks):
        for slot in range(min(int(counts[source]),capacity)):
            route=int(requests[source,3,slot])
            bad |= (route & 65535)>=ranks or (route>>24)!=0
    if bad:
        status[:,1,0]=1
        return result,status
    for destination in range(ranks):
        cursor=0
        for source in range(ranks):
            slots=[slot for slot in range(int(counts[source]))
                   if int(requests[source,3,slot]) & 65535 == destination]
            selected=slots[128*epoch:128*(epoch+1)]
            for slot in selected:
                result[destination,cursor]=wire[source,slot]
                cursor+=1
        status[destination,0,0]=cursor
    return result,status
