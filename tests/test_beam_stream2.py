import jax.numpy as jnp
import numpy as np
import pytest

from tpu_beam_search.beam_stream2 import pallas_hash_goal


@pytest.mark.parametrize('count', [0, 1, 3])
def test_candidate_identity_hash_and_exact_goal_with_tail(count):
    # The goal matches parent 0 / move 1, but not parent 0 / move 0.
    parents = np.array([[0, 1, 2, 0], [2, 0, 1, 0], [1, 2, 0, 0]], np.uint8)
    generators = np.array([[1, 0, 2, 3], [2, 0, 1, 3]], np.int32)
    central = np.array([2, 0, 1, 0], np.uint8)
    rng = np.random.default_rng(981)
    table = rng.integers(0, 2**32, (4, 4, 3), dtype=np.uint32)
    table[:, 3] = 0
    out = pallas_hash_goal(jnp.array(parents), jnp.array(generators), jnp.array(central),
                          jnp.array(table.reshape(4, -1)), jnp.array([count], jnp.uint32),
                          interpret=True)
    hashes, goals, valid = map(np.asarray, out)
    expected = np.zeros((4, 128), np.uint32)
    goal = np.zeros((1, 128), np.uint32)
    for parent in range(count):
        for move in range(2):
            child = parents[parent, generators[move]]
            index = parent * 2 + move
            expected[:, index] = np.bitwise_xor.reduce(table[:, np.arange(4), child], axis=1)
            goal[0, index] = np.array_equal(child, central)
    np.testing.assert_array_equal(hashes, expected)
    np.testing.assert_array_equal(goals, goal)
    np.testing.assert_array_equal(valid, (np.arange(128) < count * 2)[None, :])


def test_hash_collision_does_not_report_goal_and_padding_is_compared():
    parents = jnp.array([[0, 1, 2, 9]], jnp.uint8)
    generators = jnp.array([[0, 1, 2, 3]], jnp.int32)
    central = jnp.array([0, 1, 2, 0], jnp.uint8)
    table = jnp.zeros((4, 4 * 10), jnp.uint32)
    h, goal, valid = pallas_hash_goal(parents, generators, central, table,
                                     jnp.array([1], jnp.uint32), interpret=True)
    assert np.all(h == 0)
    assert np.all(goal == 0)
    assert int(valid[0, 0]) == 1


def test_stream2_rejects_mismatched_tables():
    with pytest.raises(ValueError):
        pallas_hash_goal(jnp.zeros((1, 4), jnp.uint8), jnp.zeros((2, 5), jnp.int32),
                         jnp.zeros(4, jnp.uint8), jnp.zeros((4, 12), jnp.uint32),
                         jnp.array([1], jnp.uint32), interpret=True)
