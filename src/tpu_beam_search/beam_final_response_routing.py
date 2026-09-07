"""Prepare response routing masks; no transport, collective or publication."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_final_error_summary import pallas_final_error_summary


def pallas_final_response_routing(requests,control,validation,*,world_size,interpret=False):
    """Return return-rank, validity and normalized local error flag.

    Inputs are packed requests and both summaries from materialize snapshots.
    A local error suppresses the whole batch. Every rank must still execute
    common error agreement and the same transport epochs, including zero count.
    This does not validate per-destination target capacities or drain DMA.
    """
    if (requests.ndim!=2 or requests.shape[0]!=4 or not requests.shape[1]
            or requests.shape[1]%128 or requests.shape[1]>=1<<31
            or control.shape!=(2,128) or validation.shape!=(2,128)
            or type(world_size) is not int or not 1<=world_size<=65536
            or any(x.dtype!=jnp.uint32 for x in (requests,control,validation))):
        raise ValueError('invalid final response routing ABI')
    n=requests.shape[1]
    def check(r,c,v,out):
        index=pl.program_id(0).astype(jnp.uint32)*128+jnp.arange(128,dtype=jnp.uint32)
        word=r[3,:]
        malformed=((word&jnp.uint32(65535))>=world_size)|(word>>jnp.uint32(24)!=0)
        prior=(c[1,0]!=0)|(v[0,0]!=0)|(c[0,0]>n)
        out[...] = (((index<c[0,0])&malformed)|((index==0)&prior)).astype(jnp.uint32)[None,:]
    reasons=pl.pallas_call(check,out_shape=jax.ShapeDtypeStruct((1,n),jnp.uint32),
        in_specs=(pl.BlockSpec((4,128),lambda i:(0,i)),pl.BlockSpec((2,128)),pl.BlockSpec((2,128))),
        out_specs=pl.BlockSpec((1,128),lambda i:(0,i)),grid=(n//128,),
        interpret=interpret,name='beam_final_response_route_check')(requests,control,validation)
    summary=pallas_final_error_summary(reasons,interpret=interpret)
    def flag(s,out):
        out[...] = jnp.where(jnp.arange(128)[None,:]==0,(s[0,0]!=0).astype(jnp.uint32),jnp.uint32(0))
    error=pl.pallas_call(flag,out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
        interpret=interpret,name='beam_final_response_route_error')(summary)
    def prepare(r,c,e,ranks,valid):
        index=pl.program_id(0).astype(jnp.uint32)*128+jnp.arange(128,dtype=jnp.uint32)
        live=(index<c[0,0])&(e[0,0]==0)
        ranks[...] = jnp.where(live,r[3,:]&jnp.uint32(65535),jnp.uint32(0))[None,:]
        valid[...] = live.astype(jnp.uint32)[None,:]
    ranks,valid=pl.pallas_call(prepare,
        out_shape=(jax.ShapeDtypeStruct((1,n),jnp.uint32),)*2,
        in_specs=(pl.BlockSpec((4,128),lambda i:(0,i)),pl.BlockSpec((2,128)),pl.BlockSpec((1,128))),
        out_specs=(pl.BlockSpec((1,128),lambda i:(0,i)),)*2,grid=(n//128,),
        interpret=interpret,name='beam_final_response_route_prepare')(requests,control,error)
    return ranks,valid,error


def pallas_prepare_final_response_exchange(wire,requests,control,validation,*,world_size,interpret=False):
    """Group materialized bytes by return rank and produce send intervals.

    Returns grouped payload/rank/ordinal/validity, intervals and local error.
    Invalid payload remains private; intervals/validity authorize no sends on
    error. Caller must agree errors collectively before transport. No frontier
    publication or slot reuse is implied by this preparation completing.
    """
    from .beam_final_transport import pallas_wire_to_planes
    from .beam_final_group import pallas_group_final_records
    from .beam_final_intervals import pallas_final_rank_intervals
    if (wire.ndim!=2 or requests.ndim!=2 or wire.shape[0]!=requests.shape[1]
            or type(world_size) is not int or not 1<=world_size<=128):
        raise ValueError('incompatible response exchange geometry')
    ranks,valid,error=pallas_final_response_routing(requests,control,validation,
        world_size=world_size,interpret=interpret)
    payload=pallas_wire_to_planes(wire,interpret=interpret)
    grouped=pallas_group_final_records(payload,ranks,valid,interpret=interpret)
    planes=payload.shape[0]
    intervals=pallas_final_rank_intervals(grouped[planes:planes+1],
        grouped[planes+2:planes+3],world_size=world_size,interpret=interpret)
    return grouped,intervals,error
