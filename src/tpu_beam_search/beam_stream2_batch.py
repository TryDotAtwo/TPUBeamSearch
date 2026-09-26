"""Checked S2 batch composition; diagnostic K2, not a complete depth runner."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .beam_candidate_metadata import pallas_candidate_metadata
from .beam_stream2 import pallas_hash_goal
from .beam_stream2_k1 import pallas_hash_k1_goal
from .beam_stream2_k2 import pallas_hash_k2_goal


class Stream2Batch(NamedTuple):
    words: jax.Array
    payload: jax.Array
    count: jax.Array
    error: jax.Array
    found: jax.Array
    solution_hashes: jax.Array
    suffix_ids: jax.Array


def pallas_stream2_batch(parents, generators, central, zobrist, parent_count,
                         scores, parent_base, payload_base, *, k1_table=None,
                         bucket_count=0, suffix_words=None, suffix_count=0,
                         interpret=False):
    """Preserve ordinary child identity separately from K1/K2 solution identity.

    Supplied scores are uint32 keys. No score threshold filters solved flags.
    A malformed batch rejects both channels; the caller must propagate error
    into coordinated fatal handling. Tables retain their existing validated
    host preparation contract. No solved queue, global stop or DMA drain is
    implemented here. K2 inherits static diagnostic suffix expansion.
    """
    if parent_count.shape != (1,) or parent_count.dtype != jnp.uint32:
        raise ValueError('parent_count must be uint32 [1]')
    if k1_table is None and (bucket_count or suffix_words is not None or suffix_count):
        raise ValueError('K2 requires an enabled K1 table')
    if k1_table is not None and (type(bucket_count) is not int or bucket_count <= 0):
        raise ValueError('enabled K1 requires bucket_count')
    if (suffix_words is None) != (suffix_count == 0):
        raise ValueError('suffix words/count must be supplied together')

    # Bound all S2 consumers before lookup. Metadata validation still sees the
    # original count and the safe S2 validity, so it diagnoses the rejected input.
    def bound_count(n, out):
        out[0] = jnp.where(n[0] <= parents.shape[0], n[0], jnp.uint32(0))
    safe_count = pl.pallas_call(bound_count,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.uint32), interpret=interpret,
        name='beam_s2_bound_parent_count')(parent_count)
    args = (parents, generators, central, zobrist, safe_count)
    if suffix_words is not None:
        hashes, found, valid, solution, ids = pallas_hash_k2_goal(
            *args, k1_table, suffix_words, bucket_count=bucket_count,
            suffix_count=suffix_count, interpret=interpret)
    else:
        if k1_table is None:
            hashes, found, valid = pallas_hash_goal(*args, interpret=interpret)
        else:
            hashes, found, valid = pallas_hash_k1_goal(
                *args, k1_table, bucket_count=bucket_count, interpret=interpret)
        solution = hashes
        def zero(out):
            out[...] = jnp.zeros(out.shape, jnp.uint32)
        ids = pl.pallas_call(zero,
            out_shape=jax.ShapeDtypeStruct(found.shape, jnp.uint32),
            interpret=interpret, name='beam_s2_empty_suffix')()
    words, payload, count, error = pallas_candidate_metadata(
        hashes, scores, valid, parent_count, parent_base, payload_base,
        move_count=generators.shape[0], interpret=interpret)

    def gate_solutions(flags, validity, err, out):
        out[...] = jnp.where((err[0, 0] == 0) & (validity[...] == 1),
                             flags[...], jnp.uint32(0))
    tile = pl.BlockSpec((1, 128), lambda i: (0, i))
    found = pl.pallas_call(gate_solutions,
        out_shape=jax.ShapeDtypeStruct(found.shape, jnp.uint32),
        in_specs=(tile, tile, pl.BlockSpec(error.shape)), out_specs=tile,
        grid=(found.shape[1]//128,), interpret=interpret,
        name='beam_s2_solution_admission')(found, valid, error)
    return Stream2Batch(words, payload, count, error, found, solution, ids)
