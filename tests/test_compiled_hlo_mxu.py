"""Stable parsing of the compact TPU MXU schedule fields used in reports."""

from benchmarks.compiled_hlo_mxu import extract_mxu_schedules


def test_extracts_only_convolution_schedule_lines_and_preserves_order():
    text = """
%loop = bf16[] fusion(), backend_config={"window_config":{"iteration_bounds":["9"]}}
%dense = bf16[16,1024] fusion(), metadata={op_name="jit(call)/shard_map/dot_general" stack_frame_id=53}, backend_config={"convolution_algorithm_config":{"emitter":"EmitInputBatchInLanes"},"used_scoped_memory_configs":[{"size":"12025856"}],"window_config":{"input_window_bounds":["128","6"],"iteration_bounds":["1","22","1"],"kernel_window_bounds":["128","8"],"output_window_bounds":["128","6"]}}
%head = bf16[16,30] fusion(), metadata={op_name="jit(call)/shard_map/dot_general"}, backend_config={"convolution_algorithm_config":{"emitter":"EmitOutputBatchInLanes"},"used_scoped_memory_configs":[],"window_config":{"input_window_bounds":["32","64"],"iteration_bounds":["1","2","4"],"kernel_window_bounds":["4","2"],"output_window_bounds":["4","64"]}}
"""
    rows = extract_mxu_schedules(text)
    assert [row["index"] for row in rows] == [0, 1]
    assert rows[0] == {
        "index": 0,
        "line": 3,
        "op_name": "jit(call)/shard_map/dot_general",
        "emitter": "EmitInputBatchInLanes",
        "iteration_bounds": [1, 22, 1],
        "input_window_bounds": [128, 6],
        "kernel_window_bounds": [128, 8],
        "output_window_bounds": [128, 6],
        "scoped_memory_bytes": [12_025_856],
    }
    assert rows[1]["iteration_bounds"] == [1, 2, 4]
    assert rows[1]["scoped_memory_bytes"] == []
