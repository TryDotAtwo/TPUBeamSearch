from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def pallas_apply_all_moves(
    parents,
    generators,
    *,
    MOVE_COUNT: int,
    STATE_STORAGE_LEN: int,
    interpret: bool = False,
):
    if generators.shape != (MOVE_COUNT, STATE_STORAGE_LEN):
        raise ValueError("generators shape must be (MOVE_COUNT, STATE_STORAGE_LEN)")
    if parents.shape[1] != STATE_STORAGE_LEN:
        raise ValueError("parents must use STATE_STORAGE_LEN columns")

    def kernel(parent_ref, generator_ref, output_ref):
        parent = jnp.broadcast_to(parent_ref[...], generator_ref.shape)
        output_ref[...] = jnp.take_along_axis(
            parent,
            generator_ref[...],
            axis=1,
        )

    parent_count = parents.shape[0]
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (parent_count * MOVE_COUNT, STATE_STORAGE_LEN), parents.dtype
        ),
        in_specs=[
            pl.BlockSpec((1, STATE_STORAGE_LEN), lambda parent, move: (parent, 0)),
            pl.BlockSpec((1, STATE_STORAGE_LEN), lambda parent, move: (move, 0)),
        ],
        out_specs=pl.BlockSpec(
            (1, STATE_STORAGE_LEN),
            lambda parent, move: (parent * MOVE_COUNT + move, 0),
        ),
        grid=(parent_count, MOVE_COUNT),
        interpret=interpret,
        name="stream1_apply_all_moves",
    )(parents, generators)


def _folded_input_kernel(
    state_ref,
    weight_ref,
    bias_ref,
    output_ref,
    accumulator_ref,
    *,
    STATE_LEN: int,
    NUM_CLASSES: int,
    bk: int,
    nsteps: int,
):
    k_step = pl.program_id(2)
    @pl.when(k_step == 0)
    def initialize():
        accumulator_ref[...] = jnp.zeros_like(accumulator_ref[...])

    flat_index = k_step * bk + jnp.arange(bk, dtype=jnp.int32)
    position = flat_index // NUM_CLASSES
    value = flat_index % NUM_CLASSES
    valid = position < STATE_LEN
    position = jnp.minimum(position, STATE_LEN - 1)
    states = state_ref[...].astype(jnp.int32)
    position_index = jnp.broadcast_to(position[None, :], (states.shape[0], bk))
    selected_state = jnp.take_along_axis(states, position_index, axis=1)
    one_hot = (selected_state == value[None, :]) & valid[None, :]
    accumulator_ref[...] += jnp.dot(
        one_hot.astype(jnp.bfloat16),
        weight_ref[...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(k_step == nsteps - 1)
    def finish():
        output_ref[...] = accumulator_ref[...] + bias_ref[...][None, :].astype(jnp.float32)


def pallas_folded_input_linear(
    states,
    weight,
    bias,
    *,
    STATE_LEN: int,
    NUM_CLASSES: int,
    bm: int = 128,
    bk: int = 128,
    bn: int = 256,
    interpret: bool = False,
):
    rows = states.shape[0]
    input_width = STATE_LEN * NUM_CLASSES
    output_width = weight.shape[1]
    if weight.shape[0] != input_width:
        raise ValueError("input weight rows must equal STATE_LEN * NUM_CLASSES")
    if bias.shape != (output_width,):
        raise ValueError("bias shape must equal output width")

    padded_rows = math.ceil(rows / bm) * bm
    padded_input = math.ceil(input_width / bk) * bk
    padded_output = math.ceil(output_width / bn) * bn
    states_padded = jnp.pad(states, ((0, padded_rows - rows), (0, 0)))
    weight_padded = jnp.pad(
        weight,
        ((0, padded_input - input_width), (0, padded_output - output_width)),
    )
    bias_padded = jnp.pad(bias, ((0, padded_output - output_width),))
    nsteps = padded_input // bk

    call = pl.pallas_call(
        functools.partial(
            _folded_input_kernel,
            STATE_LEN=STATE_LEN,
            NUM_CLASSES=NUM_CLASSES,
            bk=bk,
            nsteps=nsteps,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, states.shape[1]), lambda i, j, k: (i, 0)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
                pl.BlockSpec((bn,), lambda i, j, k: (j,)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(padded_rows // bm, padded_output // bn, nsteps),
        ),
        out_shape=jax.ShapeDtypeStruct(
            (padded_rows, padded_output), jnp.float32
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
        name="stream1_folded_input_linear",
    )
    return call(states_padded, weight_padded, bias_padded)[:rows, :output_width]


def _embedding_sum_kernel(
    states_ref,
    weight_ref,
    bias_ref,
    output_ref,
    accumulator_ref,
    *,
    STATE_LEN: int,
    NUM_CLASSES: int,
    weight_row_block: int,
):
    parent = pl.program_id(0)
    position = pl.program_id(2)

    @pl.when(position == 0)
    def initialize():
        accumulator_ref[...] = jnp.zeros_like(accumulator_ref[...])

    weight_row = position * NUM_CLASSES + states_ref[parent, position]
    local_row = weight_row % weight_row_block
    weight_tile = weight_ref[...]
    row_index = jnp.broadcast_to(local_row, (1, weight_tile.shape[1]))
    selected_weight = jnp.take_along_axis(weight_tile, row_index, axis=0)
    accumulator_ref[...] += selected_weight.astype(jnp.float32)

    @pl.when(position == STATE_LEN - 1)
    def finish():
        output_ref[...] = accumulator_ref[...] + bias_ref[...][None, :].astype(jnp.float32)


def pallas_embedding_sum_linear(
    states,
    weight,
    bias,
    *,
    STATE_LEN: int,
    NUM_CLASSES: int,
    bn: int = 128,
    interpret: bool = False,
):
    rows = states.shape[0]
    input_width = STATE_LEN * NUM_CLASSES
    output_width = weight.shape[1]
    if states.dtype != jnp.uint32:
        states = states.astype(jnp.uint32)
    if states.shape[1] < STATE_LEN:
        raise ValueError("states contain fewer than STATE_LEN columns")
    if weight.shape[0] != input_width:
        raise ValueError("input weight rows must equal STATE_LEN * NUM_CLASSES")
    if bias.shape != (output_width,):
        raise ValueError("bias shape must equal output width")
    if states.size * states.dtype.itemsize > 16 * 1024:
        raise ValueError("embedding-sum scalar-prefetch states exceed TPU SMEM")

    padded_output = math.ceil(output_width / bn) * bn
    weight_padded = jnp.pad(weight, ((0, 0), (0, padded_output - output_width)))
    bias_padded = jnp.pad(bias, ((0, padded_output - output_width),))

    def weight_index(parent, output_block, position, states_ref):
        row = position * NUM_CLASSES + states_ref[parent, position]
        return (row // weight_row_block, output_block)

    weight_row_block = 8
    padded_input = math.ceil(input_width / weight_row_block) * weight_row_block
    if padded_input != input_width:
        weight_padded = jnp.pad(
            weight_padded,
            ((0, padded_input - input_width), (0, 0)),
        )

    call = pl.pallas_call(
        functools.partial(
            _embedding_sum_kernel,
            STATE_LEN=STATE_LEN,
            NUM_CLASSES=NUM_CLASSES,
            weight_row_block=weight_row_block,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec((weight_row_block, bn), weight_index),
                pl.BlockSpec(
                    (bn,),
                    lambda parent, output_block, position, states_ref: (output_block,),
                ),
            ],
            out_specs=pl.BlockSpec(
                (1, bn),
                lambda parent, output_block, position, states_ref: (
                    parent,
                    output_block,
                ),
            ),
            scratch_shapes=[pltpu.VMEM((1, bn), jnp.float32)],
            grid=(rows, padded_output // bn, STATE_LEN),
        ),
        out_shape=jax.ShapeDtypeStruct((rows, padded_output), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
        name="stream1_embedding_sum_linear",
    )
    return call(states, weight_padded, bias_padded)[:, :output_width]
