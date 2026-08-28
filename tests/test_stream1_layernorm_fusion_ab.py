from benchmarks.stream1_layernorm_fusion_ab import candidate_configs


def test_layernorm_fusion_ab_pairs_each_tile_plan():
    configs = candidate_configs()
    plans = {(bm, bk, bn) for _, bm, bk, bn in configs}
    for plan in plans:
        modes = {mode for mode, bm, bk, bn in configs if (bm, bk, bn) == plan}
        assert modes == {"separate", "fused"}
    assert all(bm % 128 == bk % 128 == bn % 128 == 0 for _, bm, bk, bn in configs)
