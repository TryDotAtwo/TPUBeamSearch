from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .tpu_layout import pad_to_multiple, validate_matrix_tile
from .stream1_architecture import InputEncodingKind, LayerNormStream1Weights, Stream1Architecture
from .stream1_pallas import pallas_dense_linear, pallas_folded_input_linear


def _layer_norm_kernel(
    values_ref,
    scale_ref,
    bias_ref,
    output_ref,
    *,
    logical_width: int,
    epsilon: float,
):
    values = values_ref[...].astype(jnp.float32)
    columns = jnp.arange(values.shape[1], dtype=jnp.int32)
    valid = columns < logical_width
    masked_values = jnp.where(valid[None, :], values, 0.0)
    mean = jnp.sum(masked_values, axis=1, keepdims=True) / logical_width
    centered = jnp.where(valid[None, :], values - mean, 0.0)
    variance = jnp.sum(jnp.square(centered), axis=1, keepdims=True) / logical_width
    normalized = centered * jax.lax.rsqrt(variance + epsilon)
    result = (
        normalized * scale_ref[...][None, :].astype(jnp.float32)
        + bias_ref[...][None, :].astype(jnp.float32)
    )
    output_ref[...] = jnp.where(valid[None, :], result, 0.0).astype(jnp.bfloat16)


def pallas_layer_norm(
    values,
    scale,
    bias,
    *,
    bm: int = 128,
    width_alignment: int = 128,
    epsilon: float = 1e-5,
    interpret: bool = False,
):
    """Per-row LayerNorm with FP32 reductions and aligned BF16 storage."""

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
            dense_tile_ref[...] = (
                tile_accumulator_ref[...]
                + bias_tile_ref[...][None, :].astype(jnp.float32)
            ).astype(jnp.bfloat16)

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

    dense = dense_ref[...].astype(jnp.float32)
    columns = jnp.arange(dense.shape[1], dtype=jnp.int32)
    valid = columns < logical_output_width
    masked = jnp.where(valid[None, :], dense, 0.0)
    mean = jnp.sum(masked, axis=1, keepdims=True) / logical_output_width
    centered = jnp.where(valid[None, :], dense - mean, 0.0)
    variance = jnp.sum(jnp.square(centered), axis=1, keepdims=True) / logical_output_width
    result = (
        centered
        * jax.lax.rsqrt(variance + epsilon)
        * scale_ref[...][None, :].astype(jnp.float32)
        + beta_ref[...][None, :].astype(jnp.float32)
    )
    result = result.astype(jnp.bfloat16).astype(jnp.float32)
    if add_skip:
        result += skip_ref[...].astype(jnp.float32)
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
    interpret: bool = False,
):
    """Fuse dense output, FP32 LayerNorm, optional skip, and ReLU in VMEM."""

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
    interpret: bool = False,
):
    """Execute encoding, first dense, FP32 LayerNorm, and ReLU."""

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
    interpret: bool,
):
    dense = pallas_dense_linear(
        values,
        layer.dense.weight,
        layer.dense.bias,
        bm=bm,
        bk=bk,
        bn=bn,
        relu=False,
        interpret=interpret,
    )
    normalized = pallas_layer_norm(
        dense,
        layer.normalization.scale,
        layer.normalization.bias,
        bm=bm,
        epsilon=epsilon,
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
    interpret: bool = False,
):
    """Complete correctness-first Pallas LayerNorm ResMLP inference."""

    encoding = input_encoding or architecture.INPUT_ENCODING
    if layernorm_fusion not in ("separate", "per_layer"):
        raise ValueError("layernorm_fusion must be 'separate' or 'per_layer'")
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
            layernorm_fusion == "per_layer"
            and encoding is not InputEncodingKind.FUSED_VIRTUAL_ONE_HOT
        ),
        interpret=interpret,
    )
    for block in weights.residuals:
        skip = hidden
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
            interpret=interpret,
        )
        hidden = jax.nn.relu(
            skip.astype(jnp.float32) + branch.astype(jnp.float32)
        ).astype(jnp.bfloat16)
    return pallas_dense_linear(
        hidden,
        weights.output.weight,
        weights.output.bias,
        bm=bm,
        bk=bk_output,
        bn=bn_output,
        relu=False,
        interpret=interpret,
    )
