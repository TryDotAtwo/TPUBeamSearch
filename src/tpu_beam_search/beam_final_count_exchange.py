"""Final phase-count gather reusing serialized S5 transport, not its sum."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_s5_histogram_exchange import make_s5_histogram_call


def pallas_rank_order_final_counts(wire, *, rank=None, interpret=False):
    """Offset slot d contains counts from (local_rank-d) modulo world.

    Reorder into uint32 [2,128] with less/equal counts indexed by source rank,
    as required by final rank prefixes. This is NOT a low/high histogram sum.
    Each wire pair carries two independent phase counts in lane zero. Padding
    is ignored and cleared. Synthetic wire tests do not establish remote DMA.
    """
    if (wire.ndim != 2 or wire.shape[1] != 128 or wire.shape[0]%2
            or not 2 <= wire.shape[0] <= 256 or wire.dtype != jnp.uint32):
        raise ValueError('invalid final count wire ABI')
    world = wire.shape[0]//2
    if rank is not None and (type(rank) is not int or not 0 <= rank < world):
        raise ValueError('rank outside count exchange world')
    def reorder(w,out):
        own = jax.lax.axis_index('core') if rank is None else jnp.int32(rank)
        lanes = jnp.arange(128)
        out[...] = jnp.zeros((2,128),jnp.uint32)
        for offset in range(world):
            source = jax.lax.rem(own-offset+world,jnp.int32(world))
            for phase in range(2):
                out[phase,:] = jnp.where(lanes == source,w[2*offset+phase,0],out[phase,:])
    return pl.pallas_call(reorder,
        out_shape=jax.ShapeDtypeStruct((2,128),jnp.uint32),
        interpret=interpret,name='beam_final_rank_order_counts')(wire)


def make_final_count_exchange(mesh, *, interpret=False):
    """All ranks enter with frozen local phase counts, including zero counts.

    The existing offset exchange completes its send/receive waits before
    reordering. No local conditional skips a collective. Caller separately
    agrees errors and supplies phase counts from the same final epoch.
    """
    exchange = make_s5_histogram_call(mesh,width=128,interpret=interpret,return_wire=True)
    def call(counts):
        return pallas_rank_order_final_counts(exchange(counts),
            rank=0 if mesh.size == 1 else None,interpret=interpret)
    return call
