"""Local committed histogram sum; coordinated S5 transport remains separate."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_sum_committed_histograms(a,b,active,*,interpret=False):
    """Histograms [physical_shards,bins_padded], active lane per physical shard.

    Returns [low32/high32,bins_padded]. Caller freezes selected versions through
    snapshot completion; this call is not a concurrent-writer snapshot protocol.
    At most128 physical shards in this control ABI. No local host count read.
    """
    if (a.ndim != 2 or not 1 <= a.shape[0] <= 128 or not a.shape[1] or a.shape[1]%128
            or b.shape != a.shape or active.shape != (1,128)
            or any(x.dtype != jnp.uint32 for x in (a,b,active))):
        raise ValueError('invalid committed histogram geometry')
    shards,width = a.shape
    def kernel(ha,hb,index,out):
        low = jnp.zeros((128,),jnp.uint32)
        high = jnp.zeros((128,),jnp.uint32)
        for shard in range(shards):
            value = jnp.where((index[0,shard]&jnp.uint32(1)) == 0,ha[shard],hb[shard])
            next_low = low+value
            high += (next_low < low).astype(jnp.uint32)
            low = next_low
        out[...] = jnp.stack((low,high))
    tile = pl.BlockSpec((shards,128),lambda i:(0,i))
    return pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((2,width),jnp.uint32),
        in_specs=(tile,tile,pl.BlockSpec((1,128))),
        out_specs=pl.BlockSpec((2,128),lambda i:(0,i)),
        grid=(width//128,),interpret=interpret,name='beam_committed_histogram_uint64_pair')(a,b,active)
