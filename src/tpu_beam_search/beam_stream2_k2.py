"""Diagnostic multi-dispatch K1/K2 Stream2, not a production suffix scan."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_stream2_k1 import pallas_hash_k1_goal
from .beam_suffix_projection import pallas_suffix_projection
from .beam_suffix_hit import pallas_merge_suffix_hit


def pallas_hash_k2_goal(parents, generators, central, zobrist_words, count,
                        table, suffix_words, *, bucket_count, suffix_count,
                        interpret=False):
    """Return immediate hashes, found, valid, solution hashes, suffix IDs.

    Enabled K1 table must include central. Suffix table uses validated source
    BFS ordering, count includes empty suffix zero. Static dispatch expansion
    intentionally diagnoses semantics, not scalable production throughput.
    No child-state arrays, host tensor arithmetic, or solved queue are used.
    """
    projection = pallas_suffix_projection(generators,suffix_words,
                                        count=suffix_count,interpret=interpret)
    immediate,found,valid = pallas_hash_k1_goal(
        parents,generators,central,zobrist_words,count,table,
        bucket_count=bucket_count,interpret=interpret)
    solution = immediate
    def zero(out):
        out[...] = jnp.zeros(out.shape,jnp.uint32)
    ids = pl.pallas_call(zero,out_shape=jax.ShapeDtypeStruct(found.shape,jnp.uint32),
                         interpret=interpret,name='beam_suffix_zero_ids')()
    for suffix in range(1,suffix_count):
        def compose(g,p,out):
            indices = jnp.broadcast_to(p[:,suffix][None,:],g.shape)
            out[...] = jnp.take_along_axis(g[...],indices,axis=1,
                                           mode='promise_in_bounds')
        composed = pl.pallas_call(compose,
            out_shape=jax.ShapeDtypeStruct(generators.shape,jnp.int32),
            interpret=interpret,name='beam_immediate_then_suffix')(generators,projection)
        projected,hit,_ = pallas_hash_k1_goal(
            parents,composed,central,zobrist_words,count,table,
            bucket_count=bucket_count,interpret=interpret)
        solution,found,ids = pallas_merge_suffix_hit(solution,found,ids,
            projected,hit,valid,suffix_id=suffix,interpret=interpret)
    return immediate,found,valid,solution,ids
