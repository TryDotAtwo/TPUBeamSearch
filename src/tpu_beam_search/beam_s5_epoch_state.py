"""Local S5 epoch counters; transport and publication completion are external."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _check(state,flag):
    if state.shape != (4,128) or flag.shape != (1,) or state.dtype != jnp.uint32 or flag.dtype != jnp.uint32:
        raise ValueError('invalid S5 epoch state ABI')


def pallas_s5_local_request(state,force,*,period,interpret=False):
    """State lane0: completed jobs, update count, local/global request.

    Multi-rank caller supplies max(1, storage_shard_count) as period. A local
    zero does not permit skipping the request collective.
    """
    _check(state,force)
    if not isinstance(period,int) or not 1 <= period <= 0xffffffff:
        raise ValueError('S5 period must fit positive uint32')
    def kernel(s,f,out):
        request = (s[0,0] >= jnp.uint32(period)) | (f[0] != 0)
        out[...] = jnp.where(jnp.arange(128)[None] == 0,request.astype(jnp.uint32),jnp.uint32(0))
    return pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
        interpret=interpret,name='beam_s5_local_request')(state,force)


def pallas_s5_complete_epoch(state,published,*,interpret=False):
    """Call after common publication completion, never merely after request MAX.

    Zero preserves every counter. Nonzero increments updates and clears jobs
    and requests. Caller serializes state ownership and prevents counter wrap.
    This boolean is not itself a DMA fence or proof of distributed completion.
    """
    _check(state,published)
    def kernel(s,p,out):
        rows = jnp.arange(4)[:,None]
        changed = jnp.where(rows == 1,s[1,0]+jnp.uint32(1),jnp.uint32(0))
        out[...] = jnp.where((jnp.arange(128)[None] == 0)&(p[0] != 0),changed,s[...])
    return pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct((4,128),jnp.uint32),
        interpret=interpret,name='beam_s5_epoch_complete')(state,published)
