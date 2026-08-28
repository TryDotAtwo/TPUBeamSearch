from benchmarks.stream1_layernorm_block_diagnostic import (
    diagnostic_levels,
    residual_block_candidates,
)


def test_block_diagnostic_is_incremental_and_uses_same_input():
    assert diagnostic_levels() == (
        "dense1",
        "dense1_layernorm_relu",
        "residual_block",
    )


def test_residual_block_ab_includes_real_one_kernel_candidate():
    assert residual_block_candidates() == ("two_kernel", "one_kernel")
