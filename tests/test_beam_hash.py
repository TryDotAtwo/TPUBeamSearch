import jax.numpy as jnp
import numpy as np
import pytest

from tpu_beam_search.beam_hash import pallas_route_hashes


def oracle(lo, hi, salt):
    mask = 2**64 - 1
    def mix(x):
        x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & mask
        x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & mask
        return x ^ (x >> 31)
    rotated = ((hi << 32) | (hi >> 32)) & mask
    return mix(lo ^ rotated ^ salt ^ mix((hi + 0x9e3779b97f4a7c15) & mask))


@pytest.mark.parametrize('world,shards', [(8, 7), (3, 2**32 - 1), (256, 1024), (1, 1)])
def test_word_arithmetic_matches_unsigned64_modulo(world, shards):
    rng = np.random.default_rng(74)
    words = rng.integers(0, 2**32, (4, 256), dtype=np.uint32)
    words[:, 0], words[:, 1] = 0, 0xffffffff
    expected = np.zeros((2, 256), np.uint32)
    for i in range(256):
        lo, hi = int(words[0, i]) | int(words[1, i]) << 32, int(words[2, i]) | int(words[3, i]) << 32
        expected[:, i] = [oracle(lo, hi, 0x243f6a8885a308d3) % world,
                          oracle(lo, hi, 0x13198a2e03707344) % shards]
    actual = pallas_route_hashes(jnp.array(words), world_size=world,
                                 shard_count=shards, interpret=True)
    np.testing.assert_array_equal(actual, expected)
