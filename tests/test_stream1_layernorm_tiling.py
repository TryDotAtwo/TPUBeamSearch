from benchmarks.stream1_layernorm_tiling import candidate_bms


def test_layernorm_tiling_candidates_are_unique_and_aligned():
    candidates = candidate_bms()
    assert candidates == (128, 256, 512, 1024)
    assert len(candidates) == len(set(candidates))
    assert all(bm % 128 == 0 for bm in candidates)
