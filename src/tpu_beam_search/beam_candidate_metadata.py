"""Exact single-ring-slot S2/fixed-score -> S3 identity assembly."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_final_error_summary import pallas_final_error_summary


def pallas_candidate_metadata(hashes,scores,valid,parent_count,parent_base,payload_base,*,
                              move_count,interpret=False):
    """Return meta8, payload IDs, admitted child count and error summary.

    Child order is parent-major/move-minor. parent_base is uint32 low/high;
    payload_base identifies this slot in the S3 ring payload numbering. S3
    subsequently restores source/owner route fields; this stage writes move.
    Scores are supplied uint32 keys, not floating model outputs. Solved flags
    are deliberately not consumed: they bypass the ordinary pruning path.

    Validity must be exactly the logical child prefix. Any identity overflow,
    count/validity disagreement rejects the whole batch (count zero). Metadata
    must never be consumed ignoring that count/error result. No x64 is needed.
    """
    if (type(move_count) is not int or not 1 <= move_count <= 256
            or hashes.ndim != 2 or hashes.shape[0] != 4
            or not hashes.shape[1] or hashes.shape[1]%128 or hashes.shape[1] > 0x7fffffff
            or scores.shape != (1,hashes.shape[1]) or valid.shape != scores.shape
            or parent_count.shape != (1,) or parent_base.shape != (2,) or payload_base.shape != (1,)
            or any(x.dtype != jnp.uint32 for x in (hashes,scores,valid,parent_count,parent_base,payload_base))):
        raise ValueError('invalid candidate identity ABI')
    capacity = hashes.shape[1]
    def kernel(h,s,v,n,b,p,out,payload,reasons):
        lane = pl.program_id(0).astype(jnp.uint32)*jnp.uint32(128)+jnp.arange(128,dtype=jnp.uint32)
        bad_count = n[0] > jnp.uint32(capacity//move_count)
        expected = lane < n[0]*jnp.uint32(move_count)
        low = b[0]+lane//jnp.uint32(move_count)
        carry = (low < b[0]).astype(jnp.uint32)
        high = b[1]+carry
        identity = p[0]+lane
        parent_overflow = (carry != 0)&(b[1] == jnp.uint32(0xffffffff))
        payload_overflow = identity < p[0]
        reason = (bad_count.astype(jnp.uint32)
            | ((v[0] != expected.astype(jnp.uint32)).astype(jnp.uint32)<<jnp.uint32(1))
            | ((expected&parent_overflow).astype(jnp.uint32)<<jnp.uint32(2))
            | ((expected&payload_overflow).astype(jnp.uint32)<<jnp.uint32(3)))
        live = expected&(v[0] == 1)&~bad_count
        fields = jnp.concatenate((h[...],low[None],high[None],s[...],
                                  (lane%jnp.uint32(move_count))[None]),axis=0)
        neutral = jnp.where(jnp.arange(8)[:,None] == 6,jnp.uint32(0xffffffff),jnp.uint32(0))
        out[...] = jnp.where(live[None],fields,neutral)
        payload[...] = jnp.where(live[None],identity[None],jnp.uint32(0))
        reasons[...] = reason[None]
    tile = lambda rows:pl.BlockSpec((rows,128),lambda i:(0,i))
    words,payload,reasons = pl.pallas_call(kernel,
        out_shape=tuple(jax.ShapeDtypeStruct((rows,capacity),jnp.uint32) for rows in (8,1,1)),
        in_specs=(tile(4),tile(1),tile(1),pl.BlockSpec((1,)),pl.BlockSpec((2,)),pl.BlockSpec((1,))),
        out_specs=(tile(8),tile(1),tile(1)),grid=(capacity//128,),
        interpret=interpret,name='beam_candidate_identity')(hashes,scores,valid,parent_count,parent_base,payload_base)
    error = pallas_final_error_summary(reasons,interpret=interpret)
    def admit(n,e,out):
        out[0] = jnp.where(e[0,0] == 0,n[0]*jnp.uint32(move_count),jnp.uint32(0))
    count = pl.pallas_call(admit,out_shape=jax.ShapeDtypeStruct((1,),jnp.uint32),
        interpret=interpret,name='beam_candidate_identity_admission')(parent_count,error)
    return words,payload,count,error
