from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .tpu_layout import pad_to_multiple, validate_matrix_tile
from .stream1_architecture import InputEncodingKind, LayerNormStream1Weights, Stream1Architecture
from .stream1_pallas import pallas_dense_linear, pallas_folded_input_linear


def _validate_arithmetic(dense_rounding: str = "late", mean_mode: str = "sum_div"):
    if dense_rounding not in ("late", "bf16_before_bias"):
        raise ValueError("dense_rounding must be 'late' or 'bf16_before_bias'")
    if mean_mode not in ("sum_div", "jax"):
        raise ValueError("mean_mode must be 'sum_div' or 'jax'")


def _logical_mean(values, logical_width, mean_mode):
    # jnp.mean(BF16) divides its FP32 sum before rounding, unlike sum / width.
    if mean_mode == "jax":
        return (
            jnp.sum(values.astype(jnp.float32), axis=1, keepdims=True) / logical_width
        ).astype(values.dtype)
    return jnp.sum(values, axis=1, keepdims=True) / logical_width


def _dense_bias(accumulator, bias, dense_rounding):
    if dense_rounding == "bf16_before_bias":
        accumulator = accumulator.astype(jnp.bfloat16).astype(jnp.float32)
    return (accumulator + bias.astype(jnp.float32)).astype(jnp.bfloat16)


def _layernorm_dense_kernel(
    input_ref, weight_ref, bias_ref, output_ref, accumulator_ref,
    *, nsteps: int, dense_rounding: str,
):
    kstep = pl.program_id(2)

    @pl.when(kstep == 0)
    def initialize():
        accumulator_ref[...] = jnp.zeros_like(accumulator_ref[...])

    accumulator_ref[...] += jnp.dot(
        input_ref[...], weight_ref[...], preferred_element_type=jnp.float32
    )

    @pl.when(kstep == nsteps - 1)
    def finish():
        output_ref[...] = _dense_bias(
            accumulator_ref[...], bias_ref[...][None, :], dense_rounding
        )


def pallas_layernorm_dense(
    values,
    weight,
    bias,
    *,
    bm: int = 128,
    bk: int = 256,
    bn: int = 512,
    dense_rounding: str = "late",
    interpret: bool = False,
):
    """LN-only Dense experiment; legacy and dot-before-bias BF16 boundaries.

    Both modes use BF16 operands and FP32 tiled dot accumulation. The early
    mode rounds the *completed* reduction before bias, never each K tile.
    Interpreter matching does not establish TPU lowering equivalence.
    """
    _validate_arithmetic(dense_rounding=dense_rounding)
    if dense_rounding == "late":
        return pallas_dense_linear(
            values, weight, bias, bm=bm, bk=bk, bn=bn, relu=False, interpret=interpret
        )
    if values.ndim != 2 or weight.ndim != 2:
        raise ValueError("values and weight must be matrices")
    if min(bm, bk, bn) <= 0:
        raise ValueError("bm, bk, and bn must be positive")
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
        functools.partial(
            _layernorm_dense_kernel, nsteps=nsteps, dense_rounding=dense_rounding
        ),
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
        out_shape=jax.ShapeDtypeStruct((padded_rows, padded_output), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
        name="stream1_layernorm_dense",
    )
    return call(values_padded, weight_padded, bias_padded)[:rows, :output_width]


def _layer_norm_kernel(
    values_ref,
    scale_ref,
    bias_ref,
    output_ref,
    *,
    logical_width: int,
    epsilon: float,
    fp32_statistics: bool,
    mean_mode: str,
):
    values = values_ref[...]
    if fp32_statistics:
        values = values.astype(jnp.float32)
    columns = jnp.arange(values.shape[1], dtype=jnp.int32)
    valid = columns < logical_width
    masked_values = jnp.where(valid[None, :], values, 0.0)
    mean = _logical_mean(masked_values, logical_width, mean_mode)
    centered = jnp.where(valid[None, :], values - mean, 0.0)
    variance = _logical_mean(jnp.square(centered), logical_width, mean_mode)
    normalized = centered * jax.lax.rsqrt(variance + epsilon)
    scale = scale_ref[...][None, :]
    bias = bias_ref[...][None, :]
    if fp32_statistics:
        scale = scale.astype(jnp.float32)
        bias = bias.astype(jnp.float32)
    result = normalized * scale + bias
    output_ref[...] = jnp.where(valid[None, :], result, 0.0).astype(jnp.bfloat16)


def pallas_layer_norm(
    values,
    scale,
    bias,
    *,
    bm: int = 128,
    width_alignment: int = 128,
    epsilon: float = 1e-5,
    fp32_statistics: bool = True,
    mean_mode: str = "sum_div",
    interpret: bool = False,
):
    """Per-row LayerNorm, logical-width statistics and aligned BF16 storage.

    ``fp32_statistics`` selects statistic/affine computation dtype. For BF16
    computation, ``mean_mode='jax'`` rounds means after FP32 division; the
    legacy ``sum_div`` rounds the sum before division. Both mean and centered
    variance use the selected boundary.
    """

    _validate_arithmetic(mean_mode=mean_mode)
    if values.ndim != 2:
        raise ValueError("values must be a matrix")
    rows, logical_width = values.shape
    if scale.shape != (logical_width,) or bias.shape != (logical_width,):
        raise ValueError("scale and bias must be vectors matching values width")
    if bm <= 0 or width_alignment <= 0:
        raise ValueError("bm and width_alignment must be positive")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    padded_rows = pad_to_multiple(rows, bm)
    padded_width = pad_to_multiple(logical_width, width_alignment)
    values_padded = jnp.pad(
        values.astype(jnp.bfloat16),
        ((0, padded_rows - rows), (0, padded_width - logical_width)),
    )
    scale_padded = jnp.pad(
        scale.astype(jnp.bfloat16), ((0, padded_width - logical_width),)
    )
    bias_padded = jnp.pad(
        bias.astype(jnp.bfloat16), ((0, padded_width - logical_width),)
    )

    call = pl.pallas_call(
        functools.partial(
            _layer_norm_kernel,
            logical_width=logical_width,
            epsilon=epsilon,
            fp32_statistics=fp32_statistics,
            mean_mode=mean_mode,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(
                    (bm, padded_width), lambda row_block: (row_block, 0)
                ),
                pl.BlockSpec((padded_width,), lambda row_block: (0,)),
                pl.BlockSpec((padded_width,), lambda row_block: (0,)),
            ],
            out_specs=pl.BlockSpec(
                (bm, padded_width), lambda row_block: (row_block, 0)
            ),
            grid=(padded_rows // bm,),
        ),
        out_shape=jax.ShapeDtypeStruct(
            (padded_rows, padded_width), jnp.bfloat16
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
        interpret=interpret,
        name="stream1_layer_norm",
    )
    return call(values_padded, scale_padded, bias_padded)[:rows, :logical_width]


def _fused_dense_layer_norm_kernel(
    input_ref,
    weight_ref,
    bias_ref,
    scale_ref,
    beta_ref,
    skip_ref,
    output_ref,
    dense_ref,
    accumulator_ref,
    *,
    logical_output_width: int,
    ksteps: int,
    output_blocks: int,
    epsilon: float,
    add_skip: bool,
    relu: bool,
    fp32_statistics: bool,
    dense_rounding: str,
    mean_mode: str,
):
    def dense_body(
        input_tile_ref,
        weight_tile_ref,
        bias_tile_ref,
        dense_tile_ref,
        tile_accumulator_ref,
    ):
        kstep = pl.program_id(1)

        @pl.when(kstep == 0)
        def initialize():
            tile_accumulator_ref[...] = jnp.zeros_like(
                tile_accumulator_ref[...]
            )

        tile_accumulator_ref[...] += jnp.dot(
            input_tile_ref[...],
            weight_tile_ref[...],
            preferred_element_type=jnp.float32,
        )

        @pl.when(kstep == ksteps - 1)
        def finish():
            dense_tile_ref[...] = _dense_bias(
                tile_accumulator_ref[...], bias_tile_ref[...][None, :], dense_rounding
            )

    dense_pipeline = pltpu.emit_pipeline(
        dense_body,
        grid=(output_blocks, ksteps),
        in_specs=[
            pl.BlockSpec(
                (input_ref.shape[0], input_ref.shape[1] // ksteps),
                lambda output_block, kstep: (0, kstep),
            ),
            pl.BlockSpec(
                (
                    weight_ref.shape[0] // ksteps,
                    weight_ref.shape[1] // output_blocks,
                ),
                lambda output_block, kstep: (kstep, output_block),
            ),
            pl.BlockSpec(
                (bias_ref.shape[0] // output_blocks,),
                lambda output_block, kstep: (output_block,),
            ),
        ],
        out_specs=pl.BlockSpec(
            (dense_ref.shape[0], dense_ref.shape[1] // output_blocks),
            lambda output_block, kstep: (0, output_block),
        ),
        dimension_semantics=("parallel", "arbitrary"),
    )
    dense_pipeline(
        input_ref,
        weight_ref,
        bias_ref,
        dense_ref,
        scratches=(accumulator_ref,),
    )

    dense = dense_ref[...]
    if fp32_statistics:
        dense = dense.astype(jnp.float32)
    columns = jnp.arange(dense.shape[1], dtype=jnp.int32)
    valid = columns < logical_output_width
    masked = jnp.where(valid[None, :], dense, 0.0)
    mean = _logical_mean(masked, logical_output_width, mean_mode)
    centered = jnp.where(valid[None, :], dense - mean, 0.0)
    variance = _logical_mean(jnp.square(centered), logical_output_width, mean_mode)
    scale = scale_ref[...][None, :]
    beta = beta_ref[...][None, :]
    if fp32_statistics:
        scale = scale.astype(jnp.float32)
        beta = beta.astype(jnp.float32)
    result = centered * jax.lax.rsqrt(variance + epsilon) * scale + beta
    result = result.astype(jnp.bfloat16)
    if add_skip:
        skip = skip_ref[...]
        if fp32_statistics:
            result = result.astype(jnp.float32)
            skip = skip.astype(jnp.float32)
        result += skip
    if relu:
        result = jnp.maximum(result, 0.0)
    output_ref[...] = jnp.where(valid[None, :], result, 0.0).astype(jnp.bfloat16)


def pallas_fused_dense_layer_norm(
    values,
    weight,
    bias,
    scale,
    beta,
    *,
    skip=None,
    add_skip: bool = False,
    relu: bool = False,
    bm: int = 128,
    bk: int = 256,
    bn: int = 512,
    epsilon: float = 1e-5,
    fp32_statistics: bool = True,
    dense_rounding: str = "late",
    mean_mode: str = "sum_div",
    interpret: bool = False,
):
    """Fuse Dense, selected-statistics LayerNorm, optional skip and ReLU in VMEM."""

    _validate_arithmetic(dense_rounding, mean_mode)
    rows, input_width = values.shape
    if weight.shape[0] != input_width:
        raise ValueError("weight rows must equal input width")
    output_width = weight.shape[1]
    if bias.shape != (output_width,):
        raise ValueError("bias shape must equal output width")
    if scale.shape != (output_width,) or beta.shape != (output_width,):
        raise ValueError("scale and beta must match output width")
    if add_skip and (skip is None or skip.shape != (rows, output_width)):
        raise ValueError("skip must match dense output when add_skip is enabled")
    if not interpret:
        validate_matrix_tile(bm=bm, bk=bk, bn=bn)

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
    scale_padded = jnp.pad(
        scale.astype(jnp.bfloat16), ((0, padded_output - output_width),)
    )
    beta_padded = jnp.pad(
        beta.astype(jnp.bfloat16), ((0, padded_output - output_width),)
    )
    if skip is None:
        skip = jnp.zeros((rows, output_width), dtype=jnp.bfloat16)
    skip_padded = jnp.pad(
        skip.astype(jnp.bfloat16),
        ((0, padded_rows - rows), (0, padded_output - output_width)),
    )
    ksteps = padded_input // bk
    output_blocks = padded_output // bn
    call = pl.pallas_call(
        functools.partial(
            _fused_dense_layer_norm_kernel,
            logical_output_width=output_width,
            ksteps=ksteps,
            output_blocks=output_blocks,
            epsilon=epsilon,
            add_skip=add_skip,
            relu=relu,
            fp32_statistics=fp32_statistics,
            dense_rounding=dense_rounding,
            mean_mode=mean_mode,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(
                    (bm, padded_input), lambda row_block: (row_block, 0)
                ),
                pl.no_block_spec,
                pl.no_block_spec,
                pl.no_block_spec,
                pl.no_block_spec,
                pl.BlockSpec(
                    (bm, padded_output), lambda row_block: (row_block, 0)
                ),
            ],
            out_specs=pl.BlockSpec(
                (bm, padded_output), lambda row_block: (row_block, 0)
            ),
            scratch_shapes=[
                pltpu.VMEM((bm, padded_output), jnp.bfloat16),
                pltpu.VMEM((bm, bn), jnp.float32),
            ],
            grid=(padded_rows // bm,),
        ),
        out_shape=jax.ShapeDtypeStruct(
            (padded_rows, padded_output), jnp.bfloat16
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
        interpret=interpret,
        name="stream1_fused_dense_layer_norm",
    )
    return call(
        values_padded,
        weight_padded,
        bias_padded,
        scale_padded,
        beta_padded,
        skip_padded,
    )[:rows, :output_width]


def _fused_residual_block_kernel(
    input_ref,
    weight1_ref,
    bias1_ref,
    scale1_ref,
    beta1_ref,
    weight2_ref,
    bias2_ref,
    scale2_ref,
    beta2_ref,
    output_ref,
    hidden_ref,
    dense_ref,
    accumulator_ref,
    *,
    logical_width: int,
    ksteps: int,
    output_blocks: int,
    epsilon: float,
    fp32_statistics: bool,
    dense_rounding: str,
    mean_mode: str,
):
    def run_dense(source_ref, weight_ref, bias_ref):
        def dense_body(
            source_tile_ref,
            weight_tile_ref,
            bias_tile_ref,
            dense_tile_ref,
            tile_accumulator_ref,
        ):
            kstep = pl.program_id(1)

            @pl.when(kstep == 0)
            def initialize():
                tile_accumulator_ref[...] = jnp.zeros_like(
                    tile_accumulator_ref[...]
                )

            tile_accumulator_ref[...] += jnp.dot(
                source_tile_ref[...],
                weight_tile_ref[...],
                preferred_element_type=jnp.float32,
            )

            @pl.when(kstep == ksteps - 1)
            def finish():
                dense_tile_ref[...] = _dense_bias(
                    tile_accumulator_ref[...], bias_tile_ref[...][None, :], dense_rounding
                )

        pipeline = pltpu.emit_pipeline(
            dense_body,
            grid=(output_blocks, ksteps),
            in_specs=[
                pl.BlockSpec(
                    (source_ref.shape[0], source_ref.shape[1] // ksteps),
                    lambda output_block, kstep: (0, kstep),
                ),
                pl.BlockSpec(
                    (
                        weight_ref.shape[0] // ksteps,
                        weight_ref.shape[1] // output_blocks,
                    ),
                    lambda output_block, kstep: (kstep, output_block),
                ),
                pl.BlockSpec(
                    (bias_ref.shape[0] // output_blocks,),
                    lambda output_block, kstep: (output_block,),
                ),
            ],
            out_specs=pl.BlockSpec(
                (dense_ref.shape[0], dense_ref.shape[1] // output_blocks),
                lambda output_block, kstep: (0, output_block),
            ),
            dimension_semantics=("parallel", "arbitrary"),
        )
        pipeline(
            source_ref,
            weight_ref,
            bias_ref,
            dense_ref,
            scratches=(accumulator_ref,),
        )

    def normalize(dense, scale_ref, beta_ref):
        if fp32_statistics:
            dense = dense.astype(jnp.float32)
        columns = jnp.arange(dense.shape[1], dtype=jnp.int32)
        valid = columns < logical_width
        masked = jnp.where(valid[None, :], dense, 0.0)
        mean = _logical_mean(masked, logical_width, mean_mode)
        centered = jnp.where(valid[None, :], dense - mean, 0.0)
        variance = _logical_mean(jnp.square(centered), logical_width, mean_mode)
        scale = scale_ref[...][None, :]
        beta = beta_ref[...][None, :]
        if fp32_statistics:
            scale = scale.astype(jnp.float32)
            beta = beta.astype(jnp.float32)
        return (
            centered * jax.lax.rsqrt(variance + epsilon) * scale + beta
        ).astype(jnp.bfloat16)

    run_dense(input_ref, weight1_ref, bias1_ref)
    hidden_ref[...] = jnp.maximum(
        normalize(dense_ref[...], scale1_ref, beta1_ref), 0.0
    ).astype(jnp.bfloat16)
    run_dense(hidden_ref, weight2_ref, bias2_ref)
    branch = normalize(dense_ref[...], scale2_ref, beta2_ref)
    skip = input_ref[...]
    if fp32_statistics:
        branch = branch.astype(jnp.float32)
        skip = skip.astype(jnp.float32)
    columns = jnp.arange(branch.shape[1], dtype=jnp.int32)
    valid = columns < logical_width
    output_ref[...] = jnp.where(
        valid[None, :], jnp.maximum(skip + branch, 0.0), 0.0
    ).astype(jnp.bfloat16)


def pallas_fused_residual_block(
    values,
    block,
    *,
    bm: int = 128,
    bk: int = 256,
    bn: int = 512,
    epsilon: float = 1e-5,
    fp32_statistics: bool = True,
    dense_rounding: str = "late",
    mean_mode: str = "sum_div",
    interpret: bool = False,
):
    """Execute a complete two-dense residual block in one Pallas kernel."""

    _validate_arithmetic(dense_rounding, mean_mode)
    if values.ndim != 2:
        raise ValueError("values must be a matrix")
    rows, width = values.shape
    layers = (block.first, block.second)
    for layer in layers:
        if layer.dense.weight.shape != (width, width):
            raise ValueError("residual dense weights must be square and match values")
        if layer.dense.bias.shape != (width,):
            raise ValueError("residual dense bias must match values width")
        if layer.normalization.scale.shape != (width,):
            raise ValueError("residual LayerNorm scale must match values width")
        if layer.normalization.bias.shape != (width,):
            raise ValueError("residual LayerNorm bias must match values width")
    if not interpret:
        validate_matrix_tile(bm=bm, bk=bk, bn=bn)

    padded_rows = pad_to_multiple(rows, bm)
    padded_width = pad_to_multiple(width, math.lcm(bk, bn))
    values_padded = jnp.pad(
        values.astype(jnp.bfloat16),
        ((0, padded_rows - rows), (0, padded_width - width)),
    )

    def pad_matrix(value):
        return jnp.pad(
            value.astype(jnp.bfloat16),
            ((0, padded_width - width), (0, padded_width - width)),
        )

    def pad_vector(value):
        return jnp.pad(
            value.astype(jnp.bfloat16), ((0, padded_width - width),)
        )

    inputs = (
        values_padded,
        pad_matrix(block.first.dense.weight),
        pad_vector(block.first.dense.bias),
        pad_vector(block.first.normalization.scale),
        pad_vector(block.first.normalization.bias),
        pad_matrix(block.second.dense.weight),
        pad_vector(block.second.dense.bias),
        pad_vector(block.second.normalization.scale),
        pad_vector(block.second.normalization.bias),
    )
    ksteps = padded_width // bk
    output_blocks = padded_width // bn
    call = pl.pallas_call(
        functools.partial(
            _fused_residual_block_kernel,
            logical_width=width,
            ksteps=ksteps,
            output_blocks=output_blocks,
            epsilon=epsilon,
            fp32_statistics=fp32_statistics,
            dense_rounding=dense_rounding,
            mean_mode=mean_mode,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, padded_width), lambda row_block: (row_block, 0)),
                pl.no_block_spec,
                pl.no_block_spec,
                pl.no_block_spec,
                pl.no_block_spec,
                pl.no_block_spec,
                pl.no_block_spec,
                pl.no_block_spec,
                pl.no_block_spec,
            ],
            out_specs=pl.BlockSpec(
                (bm, padded_width), lambda row_block: (row_block, 0)
            ),
            scratch_shapes=[
                pltpu.VMEM((bm, padded_width), jnp.bfloat16),
                pltpu.VMEM((bm, padded_width), jnp.bfloat16),
                pltpu.VMEM((bm, bn), jnp.float32),
            ],
            grid=(padded_rows // bm,),
        ),
        out_shape=jax.ShapeDtypeStruct(
            (padded_rows, padded_width), jnp.bfloat16
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="stream1_fused_residual_block",
    )
    return call(*inputs)[:rows, :width]


def make_fused_virtual_one_hot_weight(
    embedding,
    input_weight,
    *,
    STATE_LEN: int,
):
    """Fold embedding and first dense weights outside the timed inference call."""

    num_classes, embed_dim = embedding.shape
    if input_weight.shape[0] != STATE_LEN * embed_dim:
        raise ValueError("input weight rows must equal STATE_LEN * EMBED_DIM")
    hidden = input_weight.shape[1]
    per_position = input_weight.reshape(STATE_LEN, embed_dim, hidden)
    return jnp.einsum(
        "ce,seh->sch",
        embedding.astype(jnp.float32),
        per_position.astype(jnp.float32),
    ).reshape(STATE_LEN * num_classes, hidden).astype(jnp.bfloat16)


def pallas_layernorm_input_prefix(
    states,
    weights: LayerNormStream1Weights,
    architecture: Stream1Architecture,
    *,
    input_encoding: InputEncodingKind,
    fused_input_weight=None,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
    bk_embedding: int | None = None,
    bn_embedding: int | None = None,
    bk_dense: int | None = None,
    bn_dense: int | None = None,
    fuse_dense_layer_norm: bool = False,
    fp32_statistics: bool = True,
    dense_rounding: str = "late",
    mean_mode: str = "sum_div",
    interpret: bool = False,
):
    """Execute encoding, first Dense, selected-statistics LayerNorm and ReLU."""

    _validate_arithmetic(dense_rounding, mean_mode)
    if (
        dense_rounding != "late"
        and input_encoding is not InputEncodingKind.EMBEDDING_GATHER
    ):
        raise ValueError("nondefault dense_rounding currently requires embedding_gather")
    logical_states = states[:, : architecture.STATE_LEN]
    embedding_bk = bk if bk_embedding is None else bk_embedding
    embedding_bn = bn if bn_embedding is None else bn_embedding
    dense_bk = bk if bk_dense is None else bk_dense
    dense_bn = bn if bn_dense is None else bn_dense
    if input_encoding is InputEncodingKind.EMBEDDING_GATHER:
        encoded = weights.embedding[logical_states.astype(jnp.int32)].reshape(
            states.shape[0], architecture.STATE_LEN * architecture.EMBED_DIM
        ).astype(jnp.bfloat16)
        if fuse_dense_layer_norm:
            return pallas_fused_dense_layer_norm(
                encoded,
                weights.input.dense.weight,
                weights.input.dense.bias,
                weights.input.normalization.scale,
                weights.input.normalization.bias,
                relu=True,
                bm=bm,
                bk=dense_bk,
                bn=dense_bn,
                epsilon=architecture.LAYER_NORM_EPSILON,
                fp32_statistics=fp32_statistics,
                dense_rounding=dense_rounding,
                mean_mode=mean_mode,
                interpret=interpret,
            )
        hidden = pallas_layernorm_dense(
            encoded,
            weights.input.dense.weight,
            weights.input.dense.bias,
            bm=bm,
            bk=dense_bk,
            bn=dense_bn,
            dense_rounding=dense_rounding,
            interpret=interpret,
        )
    elif input_encoding is InputEncodingKind.VIRTUAL_ONE_HOT_MXU:
        flattened_states = logical_states.reshape(-1, 1)
        zero_embedding_bias = jnp.zeros(
            (architecture.EMBED_DIM,), dtype=jnp.bfloat16
        )
        encoded = pallas_folded_input_linear(
            flattened_states,
            weights.embedding,
            zero_embedding_bias,
            STATE_LEN=1,
            NUM_CLASSES=architecture.NUM_CLASSES,
            bm=bm,
            bk=embedding_bk,
            bn=embedding_bn,
            relu=False,
            interpret=interpret,
        ).reshape(
            states.shape[0], architecture.STATE_LEN * architecture.EMBED_DIM
        )
        if fuse_dense_layer_norm:
            return pallas_fused_dense_layer_norm(
                encoded,
                weights.input.dense.weight,
                weights.input.dense.bias,
                weights.input.normalization.scale,
                weights.input.normalization.bias,
                relu=True,
                bm=bm,
                bk=dense_bk,
                bn=dense_bn,
                epsilon=architecture.LAYER_NORM_EPSILON,
                fp32_statistics=fp32_statistics,
                dense_rounding=dense_rounding,
                mean_mode=mean_mode,
                interpret=interpret,
            )
        hidden = pallas_dense_linear(
            encoded,
            weights.input.dense.weight,
            weights.input.dense.bias,
            bm=bm,
            bk=dense_bk,
            bn=dense_bn,
            relu=False,
            interpret=interpret,
        )
    elif input_encoding is InputEncodingKind.FUSED_VIRTUAL_ONE_HOT:
        if fused_input_weight is None:
            raise ValueError("fused_input_weight is required for fused encoding")
        hidden = pallas_folded_input_linear(
            logical_states,
            fused_input_weight,
            weights.input.dense.bias,
            STATE_LEN=architecture.STATE_LEN,
            NUM_CLASSES=architecture.NUM_CLASSES,
            bm=bm,
            bk=dense_bk,
            bn=dense_bn,
            relu=False,
            interpret=interpret,
        )
    else:
        raise ValueError(f"unsupported input encoding: {input_encoding}")

    normalized = pallas_layer_norm(
        hidden,
        weights.input.normalization.scale,
        weights.input.normalization.bias,
        bm=bm,
        epsilon=architecture.LAYER_NORM_EPSILON,
        fp32_statistics=fp32_statistics,
        mean_mode=mean_mode,
        interpret=interpret,
    )
    return jax.nn.relu(normalized).astype(jnp.bfloat16)


def _pallas_normalized_dense(
    values,
    layer,
    *,
    epsilon: float,
    relu: bool,
    bm: int,
    bk: int,
    bn: int,
    fp32_statistics: bool,
    dense_rounding: str,
    mean_mode: str,
    interpret: bool,
):
    dense = pallas_layernorm_dense(
        values,
        layer.dense.weight,
        layer.dense.bias,
        bm=bm,
        bk=bk,
        bn=bn,
        dense_rounding=dense_rounding,
        interpret=interpret,
    )
    normalized = pallas_layer_norm(
        dense,
        layer.normalization.scale,
        layer.normalization.bias,
        bm=bm,
        epsilon=epsilon,
        fp32_statistics=fp32_statistics,
        mean_mode=mean_mode,
        interpret=interpret,
    )
    return jax.nn.relu(normalized).astype(jnp.bfloat16) if relu else normalized


def stream1_layernorm_pallas_inference(
    states,
    weights: LayerNormStream1Weights,
    architecture: Stream1Architecture,
    *,
    input_encoding: InputEncodingKind | None = None,
    fused_input_weight=None,
    bm: int = 128,
    bk_input: int = 128,
    bn_input: int = 128,
    bk_hidden: int = 128,
    bn_hidden: int = 128,
    bk_output: int = 128,
    bn_output: int = 128,
    layernorm_fusion: str = "separate",
    fp32_statistics: bool = True,
    dense_rounding: str = "late",
    mean_mode: str = "sum_div",
    interpret: bool = False,
):
    """Complete correctness-first Pallas LayerNorm ResMLP inference."""

    _validate_arithmetic(dense_rounding, mean_mode)
    encoding = input_encoding or architecture.INPUT_ENCODING
    if layernorm_fusion not in ("separate", "per_layer", "per_block"):
        raise ValueError(
            "layernorm_fusion must be 'separate', 'per_layer', or 'per_block'"
        )
    hidden = pallas_layernorm_input_prefix(
        states,
        weights,
        architecture,
        input_encoding=encoding,
        fused_input_weight=fused_input_weight,
        bm=bm,
        bk=bk_input,
        bn=bn_input,
        fuse_dense_layer_norm=(
            layernorm_fusion in ("per_layer", "per_block")
            and encoding is not InputEncodingKind.FUSED_VIRTUAL_ONE_HOT
        ),
        fp32_statistics=fp32_statistics,
        dense_rounding=dense_rounding,
        mean_mode=mean_mode,
        interpret=interpret,
    )
    for block in weights.residuals:
        skip = hidden
        if layernorm_fusion == "per_block":
            hidden = pallas_fused_residual_block(
                hidden,
                block,
                bm=bm,
                bk=bk_hidden,
                bn=bn_hidden,
                epsilon=architecture.LAYER_NORM_EPSILON,
                fp32_statistics=fp32_statistics,
                dense_rounding=dense_rounding,
                mean_mode=mean_mode,
                interpret=interpret,
            )
            continue
        if layernorm_fusion == "per_layer":
            branch = pallas_fused_dense_layer_norm(
                hidden,
                block.first.dense.weight,
                block.first.dense.bias,
                block.first.normalization.scale,
                block.first.normalization.bias,
                relu=True,
                bm=bm,
                bk=bk_hidden,
                bn=bn_hidden,
                epsilon=architecture.LAYER_NORM_EPSILON,
                fp32_statistics=fp32_statistics,
                dense_rounding=dense_rounding,
                mean_mode=mean_mode,
                interpret=interpret,
            )
            hidden = pallas_fused_dense_layer_norm(
                branch,
                block.second.dense.weight,
                block.second.dense.bias,
                block.second.normalization.scale,
                block.second.normalization.bias,
                skip=skip,
                add_skip=True,
                relu=True,
                bm=bm,
                bk=bk_hidden,
                bn=bn_hidden,
                epsilon=architecture.LAYER_NORM_EPSILON,
                fp32_statistics=fp32_statistics,
                dense_rounding=dense_rounding,
                mean_mode=mean_mode,
                interpret=interpret,
            )
            continue
        branch = _pallas_normalized_dense(
            hidden,
            block.first,
            epsilon=architecture.LAYER_NORM_EPSILON,
            relu=True,
            bm=bm,
            bk=bk_hidden,
            bn=bn_hidden,
            fp32_statistics=fp32_statistics,
            dense_rounding=dense_rounding,
            mean_mode=mean_mode,
            interpret=interpret,
        )
        branch = _pallas_normalized_dense(
            branch,
            block.second,
            epsilon=architecture.LAYER_NORM_EPSILON,
            relu=False,
            bm=bm,
            bk=bk_hidden,
            bn=bn_hidden,
            fp32_statistics=fp32_statistics,
            dense_rounding=dense_rounding,
            mean_mode=mean_mode,
            interpret=interpret,
        )
        if fp32_statistics:
            skip = skip.astype(jnp.float32)
            branch = branch.astype(jnp.float32)
        hidden = jax.nn.relu(skip + branch).astype(jnp.bfloat16)
    return pallas_layernorm_dense(
        hidden,
        weights.output.weight,
        weights.output.bias,
        bm=bm,
        bk=bk_output,
        bn=bn_output,
        dense_rounding=dense_rounding,
        interpret=interpret,
    )
