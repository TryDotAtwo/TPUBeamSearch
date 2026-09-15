import numpy as np
import jax.numpy as jnp
import pytest


def test_byte_output_stores_are_full_rectangles_not_rank_one_columns():
    import jax
    from tpu_beam_search.beam_final_transport import pallas_planes_to_wire
    traced = jax.make_jaxpr(lambda x: pallas_planes_to_wire(x, interpret=True))(
        jax.ShapeDtypeStruct((32,128),jnp.uint32))
    stores = []
    def visit(obj):
        if hasattr(obj, 'jaxpr'):
            visit(obj.jaxpr)
        elif hasattr(obj, 'eqns'):
            for eq in obj.eqns:
                if eq.primitive.name == 'swap':
                    value = eq.invars[1].aval
                    if value.dtype == np.dtype('uint8'):
                        stores.append(value.shape)
                for value in eq.params.values():
                    visit(value)
        elif isinstance(obj, (tuple,list)):
            for value in obj:
                visit(value)
    visit(traced)
    assert stores, 'must inspect actual byte stores'
    assert all(len(s)==2 and s[0]%8==0 and s[1]%128==0 for s in stores), stores


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


@pytest.mark.parametrize('words',[32,64,96])
def test_planes_to_bytes_preserves_word_lane_and_tile_identity(words):
    from tpu_beam_search.beam_final_transport import pallas_planes_to_wire
    # Independent Python integer encoding, not the inverse production adapter.
    # Word and row affect different bytes; asymmetric constants expose reversal.
    planes = np.empty((words,384),np.uint32)
    expected = np.empty((384,words*4),np.uint8)
    patterns = [0,0xffffffff,0x01020304,0x80000000]
    for word in range(words):
        for row in range(384):
            value = patterns[row] if row<4 else ((word<<24)|(row<<8)|0x5a)
            planes[word,row] = value
            expected[row,word*4:word*4+4] = list(value.to_bytes(4,'little'))
    actual = pallas_planes_to_wire(jnp.asarray(planes),interpret=True)
    np.testing.assert_array_equal(actual,expected)
