import numpy as np

from benchmarks.artgor_layernorm_attribution import (
    CASE_DEFINITIONS,
    RESULT_NAME,
    attribution_variants,
    compare_checkpoint_sequences,
    decide_attribution,
    jax_layernorm_checkpoints,
)
from tpu_beam_search.stream1_layernorm_pallas_attribution import (
    PallasLayerNormArithmetic,
    _logical_width_requires_mask,
    pallas_layernorm_probe,
)


def test_attribution_variants_are_ordered_one_factor_changes():
    variants = attribution_variants()
    assert tuple(variants) == (
        "hlo_mixed_control",
        "fp32_mean",
        "fp32_variance",
        "fp32_epsilon",
        "fp32_invstd",
        "bf16_affine",
    )
    baseline = variants["hlo_mixed_control"]
    for name, candidate in tuple(variants.items())[1:]:
        differences = sum(
            getattr(candidate, field) != getattr(baseline, field)
            for field in candidate.__dataclass_fields__
        )
        assert differences == 1, name


def test_attribution_protocol_covers_six_frozen_corpora():
    assert RESULT_NAME == "artgor_layernorm_attribution.json"
    assert tuple(name for name, _, _ in CASE_DEFINITIONS) == (
        "legal_seed_42",
        "legal_seed_142",
        "legal_seed_242",
        "stress_seed_43",
        "stress_seed_143",
        "stress_seed_243",
    )


def test_jax_checkpoint_contract_has_every_observable_boundary():
    import jax.numpy as jnp

    values = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.bfloat16)
    scale = jnp.ones((4,), dtype=jnp.bfloat16)
    bias = jnp.zeros((4,), dtype=jnp.bfloat16)
    checkpoints = jax_layernorm_checkpoints(values, scale, bias)
    assert tuple(checkpoints) == (
        "mean",
        "centered",
        "variance",
        "invstd",
        "affine_fp32",
        "affine_bf16",
        "relu",
    )
    assert checkpoints["mean"].shape == (1, 1)
    assert checkpoints["centered"].shape == values.shape
    assert checkpoints["relu"].dtype == jnp.bfloat16


def test_comparison_reports_first_mismatching_boundary():
    reference = {
        "mean": np.asarray([[1]], dtype=np.float32),
        "variance": np.asarray([[2]], dtype=np.float32),
        "relu": np.asarray([[3]], dtype=np.float32),
    }
    candidate = dict(reference)
    candidate["variance"] = np.asarray([[2.5]], dtype=np.float32)
    result = compare_checkpoint_sequences(reference, candidate)
    assert result["first_mismatch"] == "variance"
    assert result["checkpoints"]["mean"]["mismatch_count"] == 0
    assert result["checkpoints"]["variance"]["mismatch_count"] == 1


def test_decision_requires_all_corpora_and_production_shape_exact():
    passing = {
        "legal42": {"small_exact": True, "production_exact": True},
        "stress43": {"small_exact": True, "production_exact": True},
    }
    assert decide_attribution(passing)["promote"] is True
    passing["stress43"]["production_exact"] = False
    decision = decide_attribution(passing)
    assert decision["promote"] is False
    assert decision["reason"] == "production_shape_not_exact"


def test_interpreted_pallas_probe_exposes_each_checkpoint():
    import jax.numpy as jnp

    values = jnp.asarray([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.bfloat16)
    scale = jnp.asarray([1, 2, 1, 2], dtype=jnp.bfloat16)
    bias = jnp.asarray([0, 1, 0, 1], dtype=jnp.bfloat16)
    reference = jax_layernorm_checkpoints(values, scale, bias)
    for name, expected in reference.items():
        actual = pallas_layernorm_probe(
            values,
            scale,
            bias,
            checkpoint=name,
            bm=2,
            width_alignment=128,
            interpret=True,
        )
        if expected.shape[1] == 1:
            expected = jnp.broadcast_to(expected, values.shape)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_interpreted_pallas_probe_matches_each_one_factor_jax_arm():
    import jax.numpy as jnp

    values = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.bfloat16)
    scale = jnp.asarray([1, 2, 1, 2], dtype=jnp.bfloat16)
    bias = jnp.asarray([0, 1, 0, 1], dtype=jnp.bfloat16)
    for arithmetic in attribution_variants().values():
        expected = jax_layernorm_checkpoints(
            values, scale, bias, arithmetic=arithmetic,
        )["relu"]
        actual = pallas_layernorm_probe(
            values,
            scale,
            bias,
            checkpoint="relu",
            arithmetic=PallasLayerNormArithmetic(**arithmetic.__dict__),
            bm=1,
            interpret=True,
        )
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_aligned_production_width_elides_compiler_sensitive_predicate():
    assert _logical_width_requires_mask(1024, 1024) is False
    assert _logical_width_requires_mask(130, 256) is True
