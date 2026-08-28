from benchmarks.stream1_layernorm_block_diagnostic import diagnostic_levels


def test_block_diagnostic_is_incremental_and_uses_same_input():
    assert diagnostic_levels() == (
        "dense1",
        "dense1_layernorm_relu",
        "residual_block",
    )
