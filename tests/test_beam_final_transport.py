import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize('width', [128, 256])
def test_response_transport_preserves_every_byte(width):
    from tpu_beam_search.beam_final_transport import pallas_wire_to_planes, pallas_planes_to_wire
    wire = np.random.default_rng(611).integers(0, 256, (256, width), dtype=np.uint8)
    planes = pallas_wire_to_planes(jnp.asarray(wire), interpret=True)
    expected = wire.copy().view('<u4').T.copy()
    np.testing.assert_array_equal(planes, expected)
    actual = pallas_planes_to_wire(planes, interpret=True)
    np.testing.assert_array_equal(actual, wire)


def test_response_target_survives_transport():
    from tpu_beam_search.beam_final_transport import pallas_wire_to_planes, pallas_planes_to_wire
    from tpu_beam_search.beam_final_response import pallas_pack_response, pallas_unpack_response
    states = jnp.full((128, 128), 217, jnp.uint8)
    targets = jnp.arange(128, dtype=jnp.uint32)[None, :] + jnp.uint32(0xffeedd00)
    wire = pallas_pack_response(states, targets, state_len=120, interpret=True)
    encoded = pallas_wire_to_planes(wire, interpret=True)
    decoded = pallas_planes_to_wire(encoded, interpret=True)
    clean, actual_targets = pallas_unpack_response(decoded, state_len=120, interpret=True)
    np.testing.assert_array_equal(actual_targets, targets)
    np.testing.assert_array_equal(clean[:, :120], states[:, :120])
    assert not np.asarray(clean[:, 120:]).any()
