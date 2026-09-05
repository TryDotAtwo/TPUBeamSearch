"""FinalResponse byte packing at logical STATE_LEN; no frontier scatter yet."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _check(states,state_len):
    if (states.ndim != 2 or states.dtype != jnp.uint8 or not states.shape[0]
            or states.shape[0]%128 or states.shape[1]%128
            or not isinstance(state_len,int) or not 0 < state_len <= states.shape[1]-4):
        raise ValueError('invalid aligned final response state ABI')


def pallas_pack_response(states,targets,*,state_len,interpret=False):
    """Use TPU tile-padded rows; compact persistent/transport width is separate."""
    _check(states,state_len)
    if targets.shape != (1,states.shape[0]) or targets.dtype != jnp.uint32:
        raise ValueError('invalid response target ABI')
    width = states.shape[1]
    def kernel(s,t,out):
        positions = jnp.arange(width)[None,:]
        value = jnp.where(positions < state_len,s[...],jnp.uint8(0))
        for byte in range(4):
            encoded = ((t[0]>>jnp.uint32(byte*8))&jnp.uint32(255)).astype(jnp.uint8)
            value = jnp.where(positions == state_len+byte,encoded[:,None],value)
        out[...] = value
    tile = pl.BlockSpec((128,width),lambda i:(i,0))
    return pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct(states.shape,jnp.uint8),
        in_specs=(tile,pl.BlockSpec((1,128),lambda i:(0,i))),out_specs=tile,
        grid=(states.shape[0]//128,),interpret=interpret,name='beam_final_response_pack')(states,targets)


def pallas_unpack_response(wire,*,state_len,interpret=False):
    _check(wire,state_len)
    width = wire.shape[1]
    def kernel(s,out,target):
        decoded = jnp.zeros((128,),jnp.uint32)
        for byte in range(4):
            decoded |= s[:,state_len+byte].astype(jnp.uint32)<<jnp.uint32(byte*8)
        target[...] = decoded[None,:]
        out[...] = jnp.where(jnp.arange(width)[None,:] < state_len,s[...],jnp.uint8(0))
    tile = pl.BlockSpec((128,width),lambda i:(i,0))
    return pl.pallas_call(kernel,
        out_shape=(jax.ShapeDtypeStruct(wire.shape,jnp.uint8),
                   jax.ShapeDtypeStruct((1,wire.shape[0]),jnp.uint32)),
        in_specs=(tile,),out_specs=(tile,pl.BlockSpec((1,128),lambda i:(0,i))),
        grid=(wire.shape[0]//128,),interpret=interpret,name='beam_final_response_unpack')(wire)
