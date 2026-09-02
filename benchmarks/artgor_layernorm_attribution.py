"""Causal LayerNorm arithmetic attribution for the all-Pallas Artgor engine.

This module deliberately keeps the candidate matrix sequential: every arm
changes exactly one observable rounding boundary relative to ``hlo_mixed``.
The TPU runner uses the same checkpoint names for materialized JAX controls
and Pallas custom calls, so the first divergent value is attributable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np


CHECKPOINT_NAMES = (
    "mean",
    "centered",
    "variance",
    "invstd",
    "affine_fp32",
    "affine_bf16",
    "relu",
)


@dataclass(frozen=True)
class LayerNormArithmetic:
    """Explicit rounding boundaries in one LayerNorm hypothesis."""

    mean_bf16: bool = True
    variance_bf16: bool = True
    epsilon_bf16: bool = True
    invstd_bf16: bool = True
    affine_fp32: bool = True


def attribution_variants() -> dict[str, LayerNormArithmetic]:
    """Return a fixed one-factor ladder, never a Cartesian sweep."""

    baseline = LayerNormArithmetic()
    return {
        "hlo_mixed_control": baseline,
        "fp32_mean": replace(baseline, mean_bf16=False),
        "fp32_variance": replace(baseline, variance_bf16=False),
        "fp32_epsilon": replace(baseline, epsilon_bf16=False),
        "fp32_invstd": replace(baseline, invstd_bf16=False),
        "bf16_affine": replace(baseline, affine_fp32=False),
    }


def jax_layernorm_checkpoints(
    values,
    scale,
    bias,
    *,
    epsilon: float = 1e-5,
    arithmetic: LayerNormArithmetic = LayerNormArithmetic(),
) -> dict[str, jax.Array]:
    """Expose each arithmetic boundary without changing their evaluation order."""

    values_fp32 = values.astype(jnp.float32)
    width = values.shape[-1]
    mean_fp32 = jnp.sum(values_fp32, axis=1, keepdims=True) / width
    mean = mean_fp32.astype(jnp.bfloat16) if arithmetic.mean_bf16 else mean_fp32
    centered = values_fp32 - mean.astype(jnp.float32)
    variance_fp32 = jnp.sum(centered * centered, axis=1, keepdims=True) / width
    variance = (
        variance_fp32.astype(jnp.bfloat16)
        if arithmetic.variance_bf16
        else variance_fp32
    )
    epsilon_value = jnp.asarray(
        epsilon, jnp.bfloat16 if arithmetic.epsilon_bf16 else jnp.float32
    ).astype(jnp.float32)
    invstd_fp32 = jax.lax.rsqrt(variance.astype(jnp.float32) + epsilon_value)
    invstd = (
        invstd_fp32.astype(jnp.bfloat16)
        if arithmetic.invstd_bf16
        else invstd_fp32
    )
    normalized = centered * invstd.astype(jnp.float32)
    if arithmetic.affine_fp32:
        affine_fp32 = (
            normalized * scale.astype(jnp.float32)[None, :]
            + bias.astype(jnp.float32)[None, :]
        )
    else:
        affine_fp32 = (
            normalized.astype(jnp.bfloat16) * scale.astype(jnp.bfloat16)[None, :]
            + bias.astype(jnp.bfloat16)[None, :]
        ).astype(jnp.float32)
    affine_bf16 = affine_fp32.astype(jnp.bfloat16)
    relu = jnp.maximum(affine_bf16, jnp.bfloat16(0)).astype(jnp.bfloat16)
    return {
        "mean": mean,
        "centered": centered,
        "variance": variance,
        "invstd": invstd,
        "affine_fp32": affine_fp32,
        "affine_bf16": affine_bf16,
        "relu": relu,
    }


def _checkpoint_metrics(reference, candidate) -> dict[str, object]:
    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        return {
            "shape_equal": False,
            "mismatch_count": int(max(reference_array.size, candidate_array.size)),
            "max_abs": None,
            "mean_abs": None,
        }
    equal = np.equal(reference_array, candidate_array)
    finite = np.isfinite(reference_array) & np.isfinite(candidate_array)
    equal = equal | (np.isnan(reference_array) & np.isnan(candidate_array))
    difference = np.abs(
        reference_array.astype(np.float64) - candidate_array.astype(np.float64)
    )
    return {
        "shape_equal": True,
        "finite": bool(np.all(finite)),
        "mismatch_count": int(equal.size - np.count_nonzero(equal)),
        "max_abs": float(np.nanmax(difference)) if difference.size else 0.0,
        "mean_abs": float(np.nanmean(difference)) if difference.size else 0.0,
    }


def compare_checkpoint_sequences(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """Compare in reference order and report the first divergent boundary."""

    if tuple(reference) != tuple(candidate):
        raise ValueError("reference and candidate checkpoint order must match")
    metrics = {
        name: _checkpoint_metrics(reference[name], candidate[name])
        for name in reference
    }
    first_mismatch = next(
        (name for name, value in metrics.items() if value["mismatch_count"]),
        None,
    )
    return {"first_mismatch": first_mismatch, "checkpoints": metrics}


def decide_attribution(corpora: Mapping[str, Mapping[str, bool]]) -> dict[str, object]:
    """Promotion gate: exact both in probe and production-shaped dispatches."""

    if not corpora:
        return {"promote": False, "reason": "no_corpora"}
    if not all(result.get("small_exact", False) for result in corpora.values()):
        return {"promote": False, "reason": "small_probe_not_exact"}
    if not all(result.get("production_exact", False) for result in corpora.values()):
        return {"promote": False, "reason": "production_shape_not_exact"}
    return {
        "promote": True,
        "reason": "all_materialized_boundaries_exact",
        "arithmetic": asdict(LayerNormArithmetic()),
    }
