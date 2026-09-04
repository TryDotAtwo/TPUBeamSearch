import jax.numpy as jnp
import numpy as np
import pytest

from tpu_beam_search.beam_external_sort import (
    pallas_compare_exchange_pass,
    pallas_external_bitonic_sort,
)


@pytest.mark.parametrize('size,stride', [(2, 1), (256, 128), (256, 64)])
def test_external_compare_exchange_pass_matches_global_bitonic_network(size, stride):
    n = 256
    rng = np.random.default_rng(9000 + size + stride)
    data = rng.integers(0, 2**32, (3, n), dtype=np.uint32)
    data[1] = np.arange(n, dtype=np.uint32)
    indices = np.arange(n)
    partner_indices = indices ^ stride
    expected = data.copy()
    for index, partner_index in enumerate(partner_indices):
        a = (int(data[0, index]), int(data[1, index]))
        b = (int(data[0, partner_index]), int(data[1, partner_index]))
        less = a < b
        equal = a == b
        want_min = ((index & size) == 0) == ((index & stride) == 0)
        swap = (want_min and not less and not equal) or (not want_min and less)
        expected[:, index] = data[:, partner_index] if swap else data[:, index]

    actual = pallas_compare_exchange_pass(
        jnp.asarray(data), key_planes=(0, 1), size=size, stride=stride,
        tile_candidates=128, interpret=True)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_external_compare_exchange_pass_rejects_unaligned_or_invalid_network():
    data = jnp.zeros((3, 256), jnp.uint32)
    for kwargs in (
        dict(size=3, stride=1),
        dict(size=128, stride=256),
        dict(size=256, stride=128, tile_candidates=64),
    ):
        with pytest.raises(ValueError):
            pallas_compare_exchange_pass(
                data, key_planes=(0,), interpret=True, **kwargs)


def test_external_bitonic_sort_orders_256_columns_with_valid_records_first():
    rng = np.random.default_rng(4242)
    data = np.empty((4, 256), np.uint32)
    data[0] = rng.integers(0, 17, 256, dtype=np.uint32)
    data[1] = np.arange(256, dtype=np.uint32)
    data[2] = (np.arange(256) % 5 != 0).astype(np.uint32)
    data[3] = rng.integers(0, 2**32, 256, dtype=np.uint32)
    order = sorted(range(256), key=lambda i: (
        1 - int(data[2, i]), int(data[0, i]), int(data[1, i])))

    actual = pallas_external_bitonic_sort(
        jnp.asarray(data), key_planes=(2, 0, 1), validity_plane=2,
        tile_candidates=128, interpret=True)
    np.testing.assert_array_equal(np.asarray(actual), data[:, order])
