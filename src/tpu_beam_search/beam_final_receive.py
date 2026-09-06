"""Compact source-major final snapshots to a dense valid prefix."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_external_sort import pallas_external_bitonic_sort


def pallas_compact_final_received(snapshots,counts,error,*,interpret=False):
    """Return payload[p,Npow2] and total/error [2,128], lane0 controls.

    One exchange chunk per source, capacity128. Preserve source/slot order.
    Invalid tails are zero. This external-sort baseline retains all snapshots
    until completion; it is not a streaming consumer or a speed result.
    """
    if (snapshots.ndim!=3 or not 1<=snapshots.shape[0]<=128
            or not snapshots.shape[1] or snapshots.shape[2]!=128
            or counts.shape!=(snapshots.shape[0],1,128) or error.shape!=(1,128)
            or any(x.dtype!=jnp.uint32 for x in (snapshots,counts,error))):
        raise ValueError('invalid final receive ABI')
    ranks,planes,_=snapshots.shape
    n=1<<(ranks*128-1).bit_length()
    def summarize(c,e,out):
        bad=(e[0,0]!=0)|jnp.any(c[:,0,0]>128)
        total=jnp.sum(c[:,0,0])
        lanes=jnp.arange(128)
        out[0,:]=jnp.where((lanes==0)&(~bad),total,jnp.uint32(0))
        out[1,:]=jnp.where(lanes==0,bad.astype(jnp.uint32),jnp.uint32(0))
    control=pl.pallas_call(summarize,out_shape=jax.ShapeDtypeStruct((2,128),jnp.uint32),
        interpret=interpret,name='beam_final_receive_control')(counts,error)
    def prepare(x,c,s,out):
        source=pl.program_id(0)
        safe=jnp.minimum(source,ranks-1)
        lanes=jnp.arange(128,dtype=jnp.uint32)
        valid=(source<ranks)&(lanes<c[safe,0,0])&(s[1,0]==0)
        out[:planes,:]=jnp.where(valid[None,:],x[0],jnp.uint32(0))
        out[planes,:]=valid.astype(jnp.uint32)
        out[planes+1,:]=source.astype(jnp.uint32)*jnp.uint32(128)+lanes
    records=pl.pallas_call(prepare,out_shape=jax.ShapeDtypeStruct((planes+2,n),jnp.uint32),
        in_specs=(pl.BlockSpec((1,planes,128),lambda i:(jnp.minimum(i,ranks-1),0,0)),
            pl.BlockSpec(counts.shape),pl.BlockSpec((2,128))),
        out_specs=pl.BlockSpec((planes+2,128),lambda i:(0,i)),grid=(n//128,),
        interpret=interpret,name='beam_final_receive_prepare')(snapshots,counts,control)
    sorted_records=pallas_external_bitonic_sort(records,key_planes=(planes,planes+1),validity_plane=planes,interpret=interpret)
    def finish(r,out):
        out[...] = jnp.where((r[planes,:]!=0)[None,:],r[:planes,:],jnp.uint32(0))
    packed=pl.pallas_call(finish,out_shape=jax.ShapeDtypeStruct((planes,n),jnp.uint32),
        in_specs=(pl.BlockSpec((planes+2,128),lambda i:(0,i)),),
        out_specs=pl.BlockSpec((planes,128),lambda i:(0,i)),grid=(n//128,),
        interpret=interpret,name='beam_final_receive_finish')(sorted_records)
    return packed,control


def pallas_materialize_final_snapshots(parents,generators,snapshots,counts,error,target_count,*,state_len,interpret=False):
    """Compact requests and materialize without a host count/record readback.

    Return wire, validation summary, receive controls and packed requests.
    Caller must preserve and collectively gate both error summaries before
    sending responses. Packed requests retain return-rank/move alongside each
    response and must stay live through response routing. No publication here.
    """
    from .beam_final_materialize import pallas_materialize_final
    requests,control=pallas_compact_final_received(snapshots,counts,error,interpret=interpret)
    wire,validation=pallas_materialize_final(parents,generators,requests,
        control[0,:1],target_count,state_len=state_len,interpret=interpret)
    return wire,validation,control,requests
