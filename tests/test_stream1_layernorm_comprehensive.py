from benchmarks.stream1_layernorm_comprehensive import (
    screening_configs,
    select_fastest_valid,
)


def test_screening_matrix_covers_tiling_statistics_and_kernel_boundaries():
    configs = screening_configs()
    assert len(configs) == 32
    assert {config["bm"] for config in configs} == {128, 256}
    assert {config["bk"] for config in configs} == {128, 256}
    assert {config["bn"] for config in configs} == {256, 512}
    assert {config["fp32_statistics"] for config in configs} == {False, True}
    assert {config["fusion"] for config in configs} == {"per_layer", "per_block"}


def test_promotion_keeps_only_fastest_correct_candidates():
    entries = [
        {"id": "slow", "status": "valid", "states_per_second": 10.0},
        {"id": "wrong", "status": "correctness_failed", "states_per_second": 99.0},
        {"id": "fast", "status": "valid", "states_per_second": 30.0},
        {"id": "middle", "status": "valid", "states_per_second": 20.0},
    ]
    assert [entry["id"] for entry in select_fastest_valid(entries, 2)] == [
        "fast",
        "middle",
    ]
