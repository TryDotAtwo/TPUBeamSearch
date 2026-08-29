from benchmarks.stream1_layernorm_depth_diagnostic import (
    begin_config,
    depth_configs,
    diagnostic_depths,
)


def test_depth_diagnostic_covers_all_residual_depths():
    assert diagnostic_depths(10) == tuple(range(1, 11))


def test_depth_diagnostic_separates_boundary_and_statistics():
    configs = depth_configs()
    assert len(configs) == 4
    assert {config["fusion"] for config in configs} == {"per_layer", "per_block"}
    assert {config["fp32_statistics"] for config in configs} == {False, True}
    assert {
        (config["fusion"], config["bm"], config["bk"], config["bn"])
        for config in configs
    } == {
        ("per_block", 128, 256, 512),
        ("per_layer", 256, 256, 512),
    }


def test_active_config_is_visible_to_incremental_checkpoints():
    result = {"configs": []}
    config = {"id": "candidate"}
    active = begin_config(result, config)
    active["depths"].append({"depth": 1})
    assert result["configs"] == [
        {"id": "candidate", "depths": [{"depth": 1}]}
    ]
