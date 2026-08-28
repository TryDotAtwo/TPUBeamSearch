import numpy as np

from benchmarks.stream1_layernorm_full_mlp import candidate_configs, make_valid_states


def test_full_layernorm_mlp_compares_separate_and_selected_fusion():
    configs = candidate_configs()
    assert {fusion for fusion, _, _, _, _ in configs} == {"separate", "per_layer"}
    assert all(not fp32_statistics for _, fp32_statistics, _, _, _ in configs)
    assert all((bm, bk, bn) == (256, 256, 512) for _, _, bm, bk, bn in configs)


def test_full_layernorm_benchmark_states_are_diverse_and_in_domain():
    states = np.asarray(make_valid_states(64, 150, 150))
    assert states.shape == (64, 150)
    assert states.dtype == np.uint8
    assert int(states.min()) >= 0
    assert int(states.max()) < 150
    assert np.unique(states, axis=0).shape[0] > 32
