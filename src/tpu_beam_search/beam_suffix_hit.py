"""Select the first successful K2 suffix without touching beam hashes."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_merge_suffix_hit(found_hash, found, suffix_ids, projected_hash,
                            projected_hit, valid, *, suffix_id, interpret=False):
    """Caller visits nonempty suffix IDs in ascending source BFS order.

    Initialize found from immediate/K1 membership, suffix IDs to zero, and
    found_hash from immediate hashes. Keep the separate immediate hash array
    for beam routing. Outputs here belong only to solved metadata. Invalid
    candidates retain their previous values; initialized flags must be zero.
    """
    capacity = found_hash.shape[1] if found_hash.ndim == 2 else 0
    if not capacity or capacity % 128 or not isinstance(suffix_id, int) or not 1 <= suffix_id <= 0xffffffff:
        raise ValueError('invalid suffix hit capacity/ID')
    inputs = (found_hash, found, suffix_ids, projected_hash, projected_hit, valid)
    rows = (4,1,1,4,1,1)
    if any(x.shape != (r,capacity) or x.dtype != jnp.uint32 for x,r in zip(inputs,rows)):
        raise ValueError('suffix hit arrays must be aligned uint32 SoA')

    def kernel(old_hash, old_hit, old_id, new_hash, new_hit, mask,
               out_hash, out_hit, out_id):
        take = (mask[...] != 0) & (old_hit[...] == 0) & (new_hit[...] != 0)
        out_hash[...] = jnp.where(take, new_hash[...], old_hash[...])
        out_hit[...] = jnp.where(take, jnp.uint32(1), old_hit[...])
        out_id[...] = jnp.where(take, jnp.uint32(suffix_id), old_id[...])

    return pl.pallas_call(kernel,
        out_shape=tuple(jax.ShapeDtypeStruct((r,capacity),jnp.uint32) for r in (4,1,1)),
        in_specs=tuple(pl.BlockSpec((r,128),lambda i:(0,i)) for r in rows),
        out_specs=tuple(pl.BlockSpec((r,128),lambda i:(0,i)) for r in (4,1,1)),
        grid=(capacity//128,),interpret=interpret,name='beam_first_suffix_hit')(*inputs)
