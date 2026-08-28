from benchmarks.stream1_layernorm_full_mlp import candidate_configs


def test_full_layernorm_mlp_compares_separate_and_selected_fusion():
    configs = candidate_configs()
    assert {fusion for fusion, _, _, _ in configs} == {"separate", "per_layer"}
    assert all((bm, bk, bn) == (256, 256, 512) for _, bm, bk, bn in configs)
