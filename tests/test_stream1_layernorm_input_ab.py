from benchmarks.stream1_layernorm_input_ab import candidate_configs
from tpu_beam_search.stream1_architecture import InputEncodingKind


def test_input_ab_covers_each_encoding_with_aligned_unique_configs():
    configs = candidate_configs()
    assert {config[0] for config in configs} == set(InputEncodingKind)
    assert len(configs) == len(set(configs))
    for _, bm, bk, bn in configs:
        assert bm % 128 == 0
        assert bk % 128 == 0
        assert bn % 128 == 0
