"""Opt-in LayerNorm arithmetic/predicate probes; never a production default.

``hlo_mixed`` is a hypothesis extracted from width-1024 HLO, not a claim of
equivalence to TPU machine arithmetic. Interpretation validates the expression
only. Named mask scopes and call names help identify sites in target dumps.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .tpu_layout import pad_to_multiple


_ARITHMETICS = ("legacy_bf16", "hlo_mixed")
_MASK_MODES = ("all", "none", "input", "center", "output", "fp32_where", "direct_2d")
_FULL_MASK_MODES = ("all", "fp32_where", "direct_2d")


def _validate_values(values):
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("values must be a nonempty matrix")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise ValueError("values must have a real floating dtype")


def _column_indices(shape, *, direct_2d):
    if direct_2d:
        # Compare an integer 2D iota, rather than broadcast a rank-one xi1.
        # A backend can canonicalize equivalent expressions; only target IR
        # inspection can establish whether this avoids its boolean layout path.
        return jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    return jnp.arange(shape[1], dtype=jnp.int32)


def _experimental_layer_norm_kernel(
    values_ref, scale_ref, bias_ref, output_ref, *, logical_width, epsilon,
    arithmetic, mask_mode,
):
    values = values_ref[...]
    if arithmetic == "hlo_mixed":
        values = values.astype(jnp.float32)
    if mask_mode != "none":
        columns = _column_indices(values.shape, direct_2d=mask_mode == "direct_2d")
        valid = columns < logical_width
        if mask_mode != "direct_2d":
            valid = valid[None, :]

    def mask(value, site):
        if mask_mode not in _FULL_MASK_MODES and mask_mode != site:
            return value
        with jax.named_scope(f"mask_{site}"):
            if mask_mode == "fp32_where":
                # Conversion back preserves legacy BF16 arithmetic after select.
                # In hlo_mixed, operands are already FP32: this arm is deliberately
                # equivalent to 'all', not an independent dtype intervention.
                return jnp.where(valid, value.astype(jnp.float32), 0.0).astype(value.dtype)
            return jnp.where(valid, value, 0.0)

    masked_values = mask(values, "input")
    if arithmetic == "legacy_bf16":
        # Exactly the existing sum_div BF16 expression, including three selects.
        mean = jnp.sum(masked_values, axis=1, keepdims=True) / logical_width
        centered = mask(values - mean, "center")
        variance = jnp.sum(jnp.square(centered), axis=1, keepdims=True) / logical_width
        normalized = centered * jax.lax.rsqrt(variance + epsilon)
        result = normalized * scale_ref[...][None, :] + bias_ref[...][None, :]
    else:
        mean = (jnp.sum(masked_values, axis=1, keepdims=True) / logical_width
                ).astype(jnp.bfloat16)
        centered = mask(values - mean.astype(jnp.float32), "center")
        variance = (jnp.sum(centered * centered, axis=1, keepdims=True) / logical_width
                    ).astype(jnp.bfloat16)
        eps = jnp.asarray(epsilon, jnp.bfloat16).astype(jnp.float32)
        inv = jax.lax.rsqrt(variance.astype(jnp.float32) + eps).astype(jnp.bfloat16)
        normalized = centered * inv.astype(jnp.float32)
        result = (normalized * scale_ref[...].astype(jnp.float32)[None, :]
                  + bias_ref[...].astype(jnp.float32)[None, :])
    output_ref[...] = mask(result, "output").astype(jnp.bfloat16)


def experimental_layer_norm(
    values, scale, bias, *, epsilon: float = 1e-5, bm: int = 128,
    alignment: int = 128, arithmetic: str = "legacy_bf16", mask_mode: str = "all",
    interpret: bool = False,
):
    """Return logical-shape BF16 output for an explicit experimental arm.

    Floating inputs/affine parameters are quantized to BF16 before the call.
    ``none/input/center/output`` are valid only with no *column* padding;
    ``input/center/output`` enable only the named original mask site. The other
    modes retain all three sites, exclude tails from statistics, and zero the
    invalid output columns in-kernel. Row padding is sliced away (as in the
    baseline); no fourth row-mask site is introduced into this isolation probe.

    ``legacy_bf16`` uses sum_div; ``hlo_mixed`` uses FP32 sums/divisions then
    BF16 mean/variance/inverse, FP32 centered values and affine arithmetic, and
    BF16-rounded epsilon promoted to FP32. No TPU equivalence is implied.
    """
    _validate_values(values)
    if arithmetic not in _ARITHMETICS:
        raise ValueError(f"arithmetic must be one of {_ARITHMETICS}")
    if mask_mode not in _MASK_MODES:
        raise ValueError(f"mask_mode must be one of {_MASK_MODES}")
    if not isinstance(bm, int) or bm <= 0:
        raise ValueError("bm must be a positive integer")
    if not isinstance(alignment, int) or alignment <= 0:
        raise ValueError("alignment must be a positive integer")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    rows, width = values.shape
    if scale.shape != (width,) or bias.shape != (width,):
        raise ValueError("scale and bias must be vectors matching values width")
    if not all(jnp.issubdtype(v.dtype, jnp.floating) for v in (scale, bias)):
        raise ValueError("scale and bias must have real floating dtypes")
    padded_rows = pad_to_multiple(rows, bm)
    padded_width = pad_to_multiple(width, alignment)
    if mask_mode not in _FULL_MASK_MODES and width != padded_width:
        raise ValueError("partial mask modes require unpadded columns: logical width == padded width")
    xp = jnp.pad(values.astype(jnp.bfloat16),
                 ((0, padded_rows - rows), (0, padded_width - width)))
    sp = jnp.pad(scale.astype(jnp.bfloat16), (0, padded_width - width))
    bp = jnp.pad(bias.astype(jnp.bfloat16), (0, padded_width - width))
    call = pl.pallas_call(
        functools.partial(_experimental_layer_norm_kernel, logical_width=width,
                          epsilon=epsilon, arithmetic=arithmetic, mask_mode=mask_mode),
        grid=(padded_rows // bm,),
        in_specs=[pl.BlockSpec((bm, padded_width), lambda i: (i, 0)),
                  pl.BlockSpec((padded_width,), lambda i: (0,)),
                  pl.BlockSpec((padded_width,), lambda i: (0,))],
        out_specs=pl.BlockSpec((bm, padded_width), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((padded_rows, padded_width), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name=f"stream1_ln_experimental_{arithmetic}_{mask_mode}",
    )
    return call(xp, sp, bp)[:rows, :width]


def _minimal_predicate_select_kernel(values_ref, output_ref, *, logical_width, predicate_layout):
    values = values_ref[...]
    columns = _column_indices(values.shape, direct_2d=predicate_layout == "direct_2d")
    valid = ((columns & 1) == 0) & (columns < logical_width)
    if predicate_layout != "direct_2d":
        valid = valid[None, :]
    with jax.named_scope("mask_minimal_select"):
        output_ref[...] = jnp.where(valid, values, 0.0)


def minimal_predicate_select(
    values, *, operand_dtype: str = "bf16", predicate_layout: str = "broadcast",
    bm: int = 128, interpret: bool = False,
):
    """Keep even columns, zero odd columns; return logical shape/operand dtype.

    Unlike a full-width prefix mask, the alternating mask remains nonconstant
    at production width 1024. BF16 and FP32 arms select operands in that dtype.
    Storage width is aligned to 128, and padded columns are zeroed in-kernel.
    ``direct_2d`` compares integer 2D iota before forming its predicate; a backend
    may still canonicalize this, so it is not a verified layout workaround.
    """
    _validate_values(values)
    if operand_dtype not in ("bf16", "fp32"):
        raise ValueError("operand_dtype must be 'bf16' or 'fp32'")
    if predicate_layout not in ("broadcast", "direct_2d"):
        raise ValueError("predicate_layout must be 'broadcast' or 'direct_2d'")
    if not isinstance(bm, int) or bm <= 0:
        raise ValueError("bm must be a positive integer")
    dtype = jnp.bfloat16 if operand_dtype == "bf16" else jnp.float32
    rows, width = values.shape
    padded_rows = pad_to_multiple(rows, bm)
    padded_width = pad_to_multiple(width, 128)
    xp = jnp.pad(values.astype(dtype), ((0, padded_rows - rows), (0, padded_width - width)))
    spec = pl.BlockSpec((bm, padded_width), lambda i: (i, 0))
    call = pl.pallas_call(
        functools.partial(_minimal_predicate_select_kernel, logical_width=width,
                          predicate_layout=predicate_layout),
        grid=(padded_rows // bm,), in_specs=[spec], out_specs=spec,
        out_shape=jax.ShapeDtypeStruct((padded_rows, padded_width), dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret, name=f"stream1_minimal_select_{operand_dtype}_{predicate_layout}",
    )
    return call(xp)[:rows, :width]
