"""Common bounded epoch count for paired final request/history streams."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .beam_s5_request import make_s5_request_call


def make_final_epoch_schedule(mesh, *, request_capacity, history_capacity,
                              interpret=False):
    """Return a uniform round count and error on every rank.

    Both collectives run even when local inputs are malformed. A nonzero
    common error suppresses all rounds; no rank may leave the epoch loop early.
    """
    ranks = mesh.size
    if not isinstance(ranks, int) or not 1 <= ranks <= 128:
        raise ValueError('rank count must be in [1,128]')
    for capacity in (request_capacity, history_capacity):
        if (not isinstance(capacity, int) or capacity <= 0 or
                capacity % 128 or capacity >= 2**31):
            raise ValueError('interval capacity must be a positive aligned int32 bound')

    def local(requests, history, prior, rounds, error):
        lanes = jnp.arange(128)
        active = lanes < ranks
        request_start = requests[0, :]
        request_count = requests[1, :]
        history_start = history[0, :]
        history_count = history[1, :]
        request_ok = (request_start <= jnp.uint32(request_capacity)) & (
            request_count <= jnp.uint32(request_capacity) -
            jnp.minimum(request_start, jnp.uint32(request_capacity)))
        history_ok = (history_start <= jnp.uint32(history_capacity)) & (
            history_count <= jnp.uint32(history_capacity) -
            jnp.minimum(history_start, jnp.uint32(history_capacity)))
        bad = (jnp.any(active & ~(request_ok & history_ok)) |
               (requests[2, 0] != 0) | (history[2, 0] != 0) |
               (prior[0, 0] != 0))
        request_count = jnp.where(active & request_ok, request_count, jnp.uint32(0))
        history_count = jnp.where(active & history_ok, history_count, jnp.uint32(0))
        counts = jnp.maximum(request_count, history_count)
        epochs = counts // jnp.uint32(128) + (counts % jnp.uint32(128) != 0).astype(jnp.uint32)
        # The validated counts fit signed int32; Mosaic's reduction is signed.
        maximum = jnp.max(epochs.astype(jnp.int32)).astype(jnp.uint32)
        rounds[...] = jnp.where(lanes[None, :] == 0, maximum, jnp.uint32(0))
        error[...] = jnp.where(lanes[None, :] == 0, bad.astype(jnp.uint32), jnp.uint32(0))

    local_call = pl.pallas_call(local,
        out_shape=(jax.ShapeDtypeStruct((1, 128), jnp.uint32),
                   jax.ShapeDtypeStruct((1, 128), jnp.uint32)),
        interpret=interpret, name='beam_final_epoch_local')
    agree = make_s5_request_call(mesh, interpret=interpret)

    def gate(rounds, error, out):
        out[...] = jnp.where(error[0, 0] == 0, rounds[...], jnp.uint32(0))

    gate_call = pl.pallas_call(gate,
        out_shape=jax.ShapeDtypeStruct((1, 128), jnp.uint32),
        interpret=interpret, name='beam_final_epoch_gate')

    def call(request_intervals, history_intervals, prior_error):
        for value in (request_intervals, history_intervals):
            if value.shape != (3, 128) or value.dtype != jnp.uint32:
                raise ValueError('intervals must be uint32 [3,128]')
        if prior_error.shape != (1, 128) or prior_error.dtype != jnp.uint32:
            raise ValueError('prior error must be uint32 [1,128]')
        local_rounds, local_error = local_call(request_intervals,
                                                history_intervals, prior_error)
        common_error = agree(local_error)
        common_rounds = agree(local_rounds)
        return gate_call(common_rounds, common_error)[0, :1], common_error

    return call
