import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from test_beam_k1_neighborhood import inputs
from tpu_beam_search.beam_k1_neighborhood import prepare_k1_neighborhood
from tpu_beam_search.beam_suffix_table import prepare_k2_suffix_table


@pytest.mark.parametrize('mode', ['immediate', 'k1', 'k2'])
def test_batch_keeps_solved_channel_and_immediate_candidate_hash_separate(mode):
    from tpu_beam_search.beam_stream2_batch import pallas_stream2_batch
    central, generators, table = inputs()
    parents = np.tile(central, (4, 1))
    parents[0, :3] = [1, 0, 2]  # Move zero reaches central exactly.
    parents[1, :3], parents[2, :3] = [0, 2, 1], [2, 2, 2]
    neighborhood = prepare_k1_neighborhood(central, generators, table,
        state_len=3, radius=1, max_entries=3, bucket_count=4)
    suffixes = prepare_k2_suffix_table(move_count=2, radius=2)
    options = {} if mode == 'immediate' else {
        'k1_table': jnp.asarray(neighborhood.table.words), 'bucket_count': 4}
    if mode == 'k2':
        options.update(suffix_words=jnp.asarray(suffixes.words), suffix_count=suffixes.count)
    result = pallas_stream2_batch(*map(jnp.asarray, (parents, generators, central, table)),
        jnp.array([3], jnp.uint32), jnp.full((1, 128), 0xffffffff, jnp.uint32),
        jnp.array([0xffffffff, 2], jnp.uint32), jnp.array([4096], jnp.uint32),
        interpret=pltpu.InterpretParams(detect_races=True), **options)
    assert int(result.count[0]) == 6 and int(result.error[0, 0]) == 0
    expected_found = np.zeros((1, 128), np.uint32)
    expected_ids = np.zeros_like(expected_found)
    expected_solution = np.zeros((4, 128), np.uint32)
    def hash_state(state):
        return np.bitwise_xor.reduce(table[:, np.arange(128)*3+state], axis=1)
    for parent in range(3):
        for move in range(2):
            lane = parent*2+move
            child = parents[parent, generators[move]]
            immediate = hash_state(child)
            np.testing.assert_array_equal(result.words[:4, lane], immediate)
            expected_solution[:, lane] = immediate
            for suffix in range(suffixes.count if mode == 'k2' else 1):
                projected = child.copy()
                packed = int(suffixes.words[0, suffix])
                for step in range(int(suffixes.words[2, suffix])):
                    projected = projected[generators[(packed >> (5*step)) & 31]]
                hashed = hash_state(projected)
                hit = (np.array_equal(projected, central) if mode == 'immediate'
                       else tuple(map(int, hashed)) in neighborhood.suffix_by_hash)
                if hit:
                    expected_found[0, lane], expected_ids[0, lane] = 1, suffix
                    expected_solution[:, lane] = hashed
                    break
    np.testing.assert_array_equal(result.found, expected_found)
    np.testing.assert_array_equal(result.suffix_ids, expected_ids)
    np.testing.assert_array_equal(result.solution_hashes, expected_solution)
    assert expected_found.any()  # UINT_MAX scores do not suppress solutions.
    if mode == 'k2':
        assert expected_ids.any()


@pytest.mark.parametrize('fault', ['parent_overflow', 'count'])
def test_invalid_batch_cannot_publish_candidates_or_solutions(fault):
    from tpu_beam_search.beam_stream2_batch import pallas_stream2_batch
    central, generators, table = inputs()
    parents = np.tile(central, (2, 1))
    parents[:, :3] = [1, 0, 2]  # Both rows would yield a solution without the gate.
    result = pallas_stream2_batch(jnp.asarray(parents),
        *map(jnp.asarray, (generators, central, table)),
        jnp.array([3 if fault == 'count' else 2], jnp.uint32),
        jnp.zeros((1, 128), jnp.uint32),
        jnp.array([0xffffffff, 0xffffffff] if fault == 'parent_overflow' else [0, 0], jnp.uint32),
        jnp.array([0], jnp.uint32), interpret=True)
    assert int(result.count[0]) == 0 and int(result.error[0, 0]) > 0
    assert not np.asarray(result.found).any()
