"""FinalRequest wire words in aligned SoA; selection and exchange are separate."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_final_requests(meta,target_local,return_rank,*,interpret=False):
    """Return request4xN and source_rank1xN.

    Request words: parent low/high, target local, return_rank16|move8<<16.
    Last byte is zero. Caller validates return ranks fit16 bits, moves are
    legal, parents/targets are in range; only selected valid rows may be sent.
    No parent cast/truncation or source/owner/return conflation is permitted.
    """
    if (meta.ndim != 2 or meta.shape[0] != 8 or not meta.shape[1]
            or meta.shape[1]%128 or target_local.shape != (1,meta.shape[1])
            or return_rank.shape != target_local.shape
            or any(x.dtype != jnp.uint32 for x in (meta,target_local,return_rank))):
        raise ValueError('invalid final request SoA ABI')
    def kernel(m,t,r,out,source):
        out[0,:],out[1,:] = m[4,:],m[5,:]
        out[2,:] = t[0,:]
        out[3,:] = r[0,:] | ((m[7,:]&jnp.uint32(255))<<jnp.uint32(16))
        source[...] = (m[7,:]>>jnp.uint32(16))[None,:]
    return pl.pallas_call(kernel,
        out_shape=tuple(jax.ShapeDtypeStruct((r,meta.shape[1]),jnp.uint32) for r in (4,1)),
        in_specs=tuple(pl.BlockSpec((r,128),lambda i:(0,i)) for r in (8,1,1)),
        out_specs=tuple(pl.BlockSpec((r,128),lambda i:(0,i)) for r in (4,1)),
        grid=(meta.shape[1]//128,),interpret=interpret,
        name='beam_final_request_words')(meta,target_local,return_rank)
