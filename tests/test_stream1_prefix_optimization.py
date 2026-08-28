from benchmarks.stream1_prefix_optimization import candidate_configs


def test_prefix_optimization_candidates_include_baseline_and_pipeline_controls():
    candidates = candidate_configs()

    assert (256, 128, 512, 0, False) in candidates
    assert (256, 128, 512, 1, False) in candidates
    assert (256, 128, 512, 2, True) in candidates
    assert all(bm % 128 == 0 for bm, _, _, _, _ in candidates)
    assert all(bk % 128 == 0 for _, bk, _, _, _ in candidates)
    assert all(bn % 128 == 0 for _, _, bn, _, _ in candidates)
    assert len(candidates) == len(set(candidates))
