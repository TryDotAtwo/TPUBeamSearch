from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .tpu_layout import pad_to_multiple, validate_matrix_tile


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
    relu: bool,
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
        value = accumulator_ref[...] + bias_ref[...][None, :].astype(jnp.float32)
        if relu:
            value = jnp.maximum(value, 0.0)
        output_ref[...] = value.astype(jnp.bfloat16)


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
    relu: bool = True,
    interpret: bool = False,
):
    rows = states.shape[0]
    input_width = STATE_LEN * NUM_CLASSES
    output_width = weight.shape[1]
    if weight.shape[0] != input_width:
        raise ValueError("input weight rows must equal STATE_LEN * NUM_CLASSES")
    if bias.shape != (output_width,):
        raise ValueError("bias shape must equal output width")

    if not interpret:
        validate_matrix_tile(bm=bm, bk=bk, bn=bn)

    padded_rows = pad_to_multiple(rows, bm)
    padded_input = pad_to_multiple(input_width, bk)
    padded_output = pad_to_multiple(output_width, bn)
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
            relu=relu,
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
            (padded_rows, padded_output), jnp.bfloat16
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
    weight_tile = weight_ref[...].astype(jnp.float32)
    row_index = jnp.broadcast_to(local_row, (1, weight_tile.shape[1]))
    selected_weight = jnp.take_along_axis(weight_tile, row_index, axis=0)
    accumulator_ref[...] += selected_weight

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
    if rows > 1:
        return jnp.concatenate(
            [
                pallas_embedding_sum_linear(
                    states[row : row + 1],
                    weight,
                    bias,
                    STATE_LEN=STATE_LEN,
                    NUM_CLASSES=NUM_CLASSES,
                    bn=bn,
                    interpret=interpret,
                )
                for row in range(rows)
            ],
            axis=0,
        )

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


def _dense_linear_kernel(
    input_ref,
    weight_ref,
    bias_ref,
    output_ref,
    accumulator_ref,
    *,
    nsteps: int,
    relu: bool,
):
    k_step = pl.program_id(2)

    @pl.when(k_step == 0)
    def initialize():
        accumulator_ref[...] = jnp.zeros_like(accumulator_ref[...])

    accumulator_ref[...] += jnp.dot(
        input_ref[...],
        weight_ref[...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(k_step == nsteps - 1)
    def finish():
        value = accumulator_ref[...] + bias_ref[...][None, :].astype(jnp.float32)
        if relu:
            value = jnp.maximum(value, 0.0)
        output_ref[...] = value.astype(jnp.bfloat16)


def pallas_dense_linear(
    values,
    weight,
    bias,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 256,
    relu: bool = False,
    interpret: bool = False,
):
    if not interpret:
        validate_matrix_tile(bm=bm, bk=bk, bn=bn)
    rows, input_width = values.shape
    if weight.shape[0] != input_width:
        raise ValueError("weight rows must equal input width")
    output_width = weight.shape[1]
    if bias.shape != (output_width,):
        raise ValueError("bias shape must equal output width")

    padded_rows = pad_to_multiple(rows, bm)
    padded_input = pad_to_multiple(input_width, bk)
    padded_output = pad_to_multiple(output_width, bn)
    values_padded = jnp.pad(
        values.astype(jnp.bfloat16),
        ((0, padded_rows - rows), (0, padded_input - input_width)),
    )
    weight_padded = jnp.pad(
        weight.astype(jnp.bfloat16),
        ((0, padded_input - input_width), (0, padded_output - output_width)),
    )
    bias_padded = jnp.pad(
        bias.astype(jnp.bfloat16), ((0, padded_output - output_width),)
    )
    nsteps = padded_input // bk

    call = pl.pallas_call(
        functools.partial(_dense_linear_kernel, nsteps=nsteps, relu=relu),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
                pl.BlockSpec((bn,), lambda i, j, k: (j,)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(padded_rows // bm, padded_output // bn, nsteps),
        ),
        out_shape=jax.ShapeDtypeStruct(
            (padded_rows, padded_output), jnp.bfloat16
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
        name="stream1_dense_linear",
    )
    return call(values_padded, weight_padded, bias_padded)[:rows, :output_width]


def _fused_folded_hidden_kernel(
    state_ref,
    input_weight_ref,
    input_bias_ref,
    hidden_weight_ref,
    hidden_bias_ref,
    output_ref,
    hidden_ref,
    input_accumulator_ref,
    hidden_accumulator_ref,
    *,
    STATE_LEN: int,
    NUM_CLASSES: int,
    bk_input: int,
    input_ksteps: int,
    hidden_ksteps: int,
    input_output_blocks: int,
    hidden_output_blocks: int,
):
    def input_body(weight_tile_ref, bias_tile_ref, hidden_tile_ref, accumulator_ref):
        k_step = pl.program_id(1)

        @pl.when(k_step == 0)
        def initialize():
            accumulator_ref[...] = jnp.zeros_like(accumulator_ref[...])

        flat_index = k_step * bk_input + jnp.arange(bk_input, dtype=jnp.int32)
        position = flat_index // NUM_CLASSES
        value = flat_index % NUM_CLASSES
        valid = position < STATE_LEN
        position = jnp.minimum(position, STATE_LEN - 1)
        states = state_ref[...].astype(jnp.int32)
        position_index = jnp.broadcast_to(
            position[None, :], (states.shape[0], bk_input)
        )
        selected_state = jnp.take_along_axis(states, position_index, axis=1)
        one_hot = (selected_state == value[None, :]) & valid[None, :]
        accumulator_ref[...] += jnp.dot(
            one_hot.astype(jnp.bfloat16),
            weight_tile_ref[...],
            preferred_element_type=jnp.float32,
        )

        @pl.when(k_step == input_ksteps - 1)
        def finish():
            value_with_bias = (
                accumulator_ref[...]
                + bias_tile_ref[...][None, :].astype(jnp.float32)
            )
            hidden_tile_ref[...] = jnp.maximum(value_with_bias, 0.0).astype(
                jnp.bfloat16
            )

    input_pipeline = pltpu.emit_pipeline(
        input_body,
        grid=(input_output_blocks, input_ksteps),
        in_specs=[
            pl.BlockSpec(
                (bk_input, input_weight_ref.shape[1] // input_output_blocks),
                lambda output_block, k_step: (k_step, output_block),
            ),
            pl.BlockSpec(
                (input_bias_ref.shape[0] // input_output_blocks,),
                lambda output_block, k_step: (output_block,),
            ),
        ],
        out_specs=pl.BlockSpec(
            (state_ref.shape[0], hidden_ref.shape[1] // input_output_blocks),
            lambda output_block, k_step: (0, output_block),
        ),
        dimension_semantics=("parallel", "arbitrary"),
    )
    input_pipeline(
        input_weight_ref,
        input_bias_ref,
        hidden_ref,
        scratches=(input_accumulator_ref,),
    )

    def hidden_body(
        hidden_tile_ref,
        weight_tile_ref,
        bias_tile_ref,
        output_tile_ref,
        accumulator_ref,
    ):
        k_step = pl.program_id(1)

        @pl.when(k_step == 0)
        def initialize():
            accumulator_ref[...] = jnp.zeros_like(accumulator_ref[...])

        accumulator_ref[...] += jnp.dot(
            hidden_tile_ref[...],
            weight_tile_ref[...],
            preferred_element_type=jnp.float32,
        )

        @pl.when(k_step == hidden_ksteps - 1)
        def finish():
            value_with_bias = (
                accumulator_ref[...]
                + bias_tile_ref[...][None, :].astype(jnp.float32)
            )
            output_tile_ref[...] = jnp.maximum(value_with_bias, 0.0).astype(
                jnp.bfloat16
            )

    hidden_pipeline = pltpu.emit_pipeline(
        hidden_body,
        grid=(hidden_output_blocks, hidden_ksteps),
        in_specs=[
            pl.BlockSpec(
                (state_ref.shape[0], hidden_ref.shape[1] // hidden_ksteps),
                lambda output_block, k_step: (0, k_step),
            ),
            pl.BlockSpec(
                (
                    hidden_weight_ref.shape[0] // hidden_ksteps,
                    hidden_weight_ref.shape[1] // hidden_output_blocks,
                ),
                lambda output_block, k_step: (k_step, output_block),
            ),
            pl.BlockSpec(
                (hidden_bias_ref.shape[0] // hidden_output_blocks,),
                lambda output_block, k_step: (output_block,),
            ),
        ],
        out_specs=pl.BlockSpec(
            (state_ref.shape[0], output_ref.shape[1] // hidden_output_blocks),
            lambda output_block, k_step: (0, output_block),
        ),
        dimension_semantics=("parallel", "arbitrary"),
    )
    hidden_pipeline(
        hidden_ref,
        hidden_weight_ref,
        hidden_bias_ref,
        output_ref,
        scratches=(hidden_accumulator_ref,),
    )


def pallas_fused_folded_hidden(
    states,
    input_weight,
    input_bias,
    hidden_weight,
    hidden_bias,
    *,
    STATE_LEN: int,
    NUM_CLASSES: int,
    bm: int = 128,
    bk_input: int = 128,
    bn_input: int = 256,
    bk_hidden: int = 128,
    bn_hidden: int = 512,
    interpret: bool = False,
):
    if not interpret:
        validate_matrix_tile(bm=bm, bk=bk_input, bn=bn_input)
        validate_matrix_tile(bm=bm, bk=bk_hidden, bn=bn_hidden)

    rows = states.shape[0]
    input_width = STATE_LEN * NUM_CLASSES
    if input_weight.shape[0] != input_width:
        raise ValueError("input weight rows must equal STATE_LEN * NUM_CLASSES")
    hidden_width = input_weight.shape[1]
    if input_bias.shape != (hidden_width,):
        raise ValueError("input bias shape must equal input-layer output width")
    if hidden_weight.shape[0] != hidden_width:
        raise ValueError("hidden weight rows must equal input-layer output width")
    output_width = hidden_weight.shape[1]
    if hidden_bias.shape != (output_width,):
        raise ValueError("hidden bias shape must equal hidden-layer output width")

    if interpret:
        virtual_one_hot = jax.nn.one_hot(
            states[:, :STATE_LEN].astype(jnp.int32),
            NUM_CLASSES,
            dtype=jnp.bfloat16,
        ).reshape(rows, input_width)
        hidden = jnp.maximum(
            virtual_one_hot.astype(jnp.float32)
            @ input_weight.astype(jnp.float32)
            + input_bias.astype(jnp.float32),
            0.0,
        ).astype(jnp.bfloat16)
        return jnp.maximum(
            hidden.astype(jnp.float32) @ hidden_weight.astype(jnp.float32)
            + hidden_bias.astype(jnp.float32),
            0.0,
        ).astype(jnp.bfloat16)

    padded_rows = pad_to_multiple(rows, bm)
    padded_input = pad_to_multiple(input_width, bk_input)
    padded_hidden = pad_to_multiple(
        hidden_width, max(bn_input, bk_hidden)
    )
    padded_output = pad_to_multiple(output_width, bn_hidden)
    states_padded = jnp.pad(states, ((0, padded_rows - rows), (0, 0)))
    input_weight_padded = jnp.pad(
        input_weight.astype(jnp.bfloat16),
        ((0, padded_input - input_width), (0, padded_hidden - hidden_width)),
    )
    input_bias_padded = jnp.pad(
        input_bias.astype(jnp.bfloat16), ((0, padded_hidden - hidden_width),)
    )
    hidden_weight_padded = jnp.pad(
        hidden_weight.astype(jnp.bfloat16),
        ((0, padded_hidden - hidden_width), (0, padded_output - output_width)),
    )
    hidden_bias_padded = jnp.pad(
        hidden_bias.astype(jnp.bfloat16), ((0, padded_output - output_width),)
    )

    input_ksteps = padded_input // bk_input
    hidden_ksteps = padded_hidden // bk_hidden
    input_output_blocks = padded_hidden // bn_input
    hidden_output_blocks = padded_output // bn_hidden
    call = pl.pallas_call(
        functools.partial(
            _fused_folded_hidden_kernel,
            STATE_LEN=STATE_LEN,
            NUM_CLASSES=NUM_CLASSES,
            bk_input=bk_input,
            input_ksteps=input_ksteps,
            hidden_ksteps=hidden_ksteps,
            input_output_blocks=input_output_blocks,
            hidden_output_blocks=hidden_output_blocks,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, states.shape[1]), lambda row_block: (row_block, 0)),
                pl.no_block_spec,
                pl.no_block_spec,
                pl.no_block_spec,
                pl.no_block_spec,
            ],
            out_specs=pl.BlockSpec(
                (bm, padded_output), lambda row_block: (row_block, 0)
            ),
            scratch_shapes=[
                pltpu.VMEM((bm, padded_hidden), jnp.bfloat16),
                pltpu.VMEM((bm, bn_input), jnp.float32),
                pltpu.VMEM((bm, bn_hidden), jnp.float32),
            ],
            grid=(padded_rows // bm,),
        ),
        out_shape=jax.ShapeDtypeStruct(
            (padded_rows, padded_output), jnp.bfloat16
        ),
        interpret=interpret,
        name="stream1_fused_folded_hidden",
    )
    return call(
        states_padded,
        input_weight_padded,
        input_bias_padded,
        hidden_weight_padded,
        hidden_bias_padded,
    )[:rows, :output_width]
