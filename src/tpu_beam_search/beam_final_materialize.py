"""Validated serialized parent DMA to final responses; physical gate pending."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from .beam_final_validation import pallas_validate_final_requests
from .beam_final_error_summary import pallas_final_error_summary
from .beam_stream2 import _take_clipped


def pallas_materialize_final(parents,generators,requests,count,target_count,*,state_len,interpret=False):
    """Whole invalid batch yields zero wire and error summary, no parent DMA.

    Caller validates permutations and return ranks and gates sending on errors.
    Parent arena is locally indexed and fits signed32. No response exchange or
    frontier scatter; one DMA per valid request is a diagnostic baseline.
    """
    if (parents.ndim != 2 or parents.dtype != jnp.uint8 or not 0 < parents.shape[0] < 0x7fffffff
            or parents.shape[1]%128 or not 0 < state_len <= parents.shape[1]-4
            or generators.ndim != 2 or generators.shape[1] != parents.shape[1]
            or generators.dtype != jnp.int32):
        raise ValueError('invalid final parent/generator ABI')
    reasons = pallas_validate_final_requests(requests,count,
        jnp.array([parents.shape[0],0],jnp.uint32),target_count,
        move_count=generators.shape[0],interpret=interpret)
    errors = pallas_final_error_summary(reasons,interpret=interpret)
    width,n = parents.shape[1],requests.shape[1]
    def kernel(p,g,r,c,e,out,staging,sem):
        index = pl.program_id(0)
        out[...] = jnp.zeros((1,width),jnp.uint8)
        @pl.when((index.astype(jnp.uint32) < c[0]) & (e[0,0] == 0))
        def valid():
            lanes = jnp.arange(128,dtype=jnp.int32)
            def scalar(row):
                return jnp.sum(jnp.where(lanes == index%128,r[row],jnp.uint32(0)).astype(jnp.int32)).astype(jnp.uint32)
            parent,target,packed = scalar(0),scalar(2),scalar(3)
            copy = pltpu.make_async_copy(p.at[pl.ds(parent.astype(jnp.int32),1),:],staging,sem)
            copy.start()
            copy.wait()
            move = ((packed>>jnp.uint32(16))&jnp.uint32(255)).astype(jnp.int32)
            selected = jnp.sum(jnp.where(jnp.arange(generators.shape[0])[:,None] == move,g[...],0),axis=0).astype(jnp.int32)
            child = _take_clipped(staging[0],selected)
            positions = jnp.arange(width)
            child = jnp.where(positions < state_len,child,jnp.uint8(0))
            for byte in range(4):
                child = jnp.where(positions == state_len+byte,
                    ((target>>jnp.uint32(byte*8))&jnp.uint32(255)).astype(jnp.uint8),child)
            out[...] = child[None,:]
    wire = pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct((n,width),jnp.uint8),
        in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),pl.BlockSpec(generators.shape),
                  pl.BlockSpec((4,128),lambda i:(0,i//128)),pl.BlockSpec((1,)),pl.BlockSpec((2,128))),
        out_specs=pl.BlockSpec((1,width),lambda i:(i,0)),grid=(n,),
        scratch_shapes=(pltpu.VMEM((1,width),jnp.uint8),pltpu.SemaphoreType.DMA),
        interpret=interpret,name='beam_final_validated_parent_dma')(parents,generators,requests,count,errors)
    return wire,errors
