import numpy as np
import pytest

from tpu_beam_search.beam_types import BeamStorage, pack_candidates, unpack_candidates


@pytest.mark.parametrize('length,moves,storage', [(120, 24, 128), (150, 30, 160)])
def test_logical_state_and_candidate_storage_are_independent(length, moves, storage):
    layout = BeamStorage(length, moves, 129)
    assert layout.STATE_STORAGE_LEN == storage
    assert layout.candidate_capacity == 256
    assert layout.metadata_bytes == 8192
    assert layout.response_index_offset == length


def test_lossless_words_and_neutral_invalid_tail():
    packed = pack_candidates([2**127 + 7], [2**63 + 9], [11], [0x10002], capacity=129)
    np.testing.assert_array_equal(packed[:, 0], [7, 0, 0, 2**31, 9, 2**31, 11, 0x10002])
    assert packed.shape == (8, 256)
    assert packed.dtype == np.uint32
    assert np.all(packed[6, 1:] == 0xffffffff)
    assert np.all(packed[[0, 1, 2, 3, 4, 5, 7], 1:] == 0)
    decoded = unpack_candidates(packed, count=1)
    assert decoded == ([2**127 + 7], [2**63 + 9], [11], [0x10002])


@pytest.mark.parametrize('args', [(0, 24, 128), (120, 257, 128), (120, 24, 0)])
def test_invalid_storage_rejected(args):
    with pytest.raises(ValueError):
        BeamStorage(*args)


@pytest.mark.parametrize('field,value', [(0, -1), (0, 2**128), (1, 2**64), (2, 2**32), (3, -1), (1, 1.5)])
def test_invalid_word_value_is_not_silently_wrapped(field, value):
    fields = [[1], [2], [3], [4]]
    fields[field] = [value]
    with pytest.raises(ValueError):
        pack_candidates(*fields, capacity=128)


def test_count_capacity_and_field_lengths_are_checked():
    with pytest.raises(ValueError):
        pack_candidates([1, 2], [1, 2], [1, 2], [1, 2], capacity=1)
    with pytest.raises(ValueError):
        pack_candidates([1], [], [1], [1], capacity=128)
    with pytest.raises(ValueError):
        unpack_candidates(np.zeros((8, 128), np.uint32), count=129)
    with pytest.raises(ValueError):
        BeamStorage(120, 24, 128, WORLD_SIZE=257)
