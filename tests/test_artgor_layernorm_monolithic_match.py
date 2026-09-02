from benchmarks.artgor_layernorm_monolithic_match import (
    RESULT_NAME,
    decide_match,
    variant_names,
)


def test_monolithic_match_uses_the_frozen_one_factor_ladder():
    assert RESULT_NAME == "artgor_layernorm_monolithic_match.json"
    assert variant_names() == (
        "hlo_mixed_control", "fp32_mean", "fp32_variance",
        "fp32_epsilon", "fp32_invstd", "bf16_affine",
    )


def test_monolithic_match_promotes_only_hash_exact_jax_and_pallas_variant():
    cases = {
        "a": {
            "x": {"jax": {"exact": True}, "pallas": {"exact": True}},
            "y": {"jax": {"exact": True}, "pallas": {"exact": False}},
        },
        "b": {
            "x": {"jax": {"exact": True}, "pallas": {"exact": True}},
            "y": {"jax": {"exact": True}, "pallas": {"exact": True}},
        },
    }
    assert decide_match(cases) == {
        "exact_jax_variants": ["x", "y"],
        "exact_pallas_variants": ["x"],
        "selected": "x",
    }
