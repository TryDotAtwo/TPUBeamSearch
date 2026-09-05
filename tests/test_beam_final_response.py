import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize('state_len,width',[(120,128),(150,256)])
def test_response_round_trip_preserves_logical_state_and_clears_padding(state_len,width):
    from tpu_beam_search.beam_final_response import pallas_pack_response,pallas_unpack_response
    states = np.full((128,width),239,np.uint8)
    states[:,:state_len] = np.arange(state_len,dtype=np.uint8)
    targets = np.arange(128,dtype=np.uint32)[None,:]+0xffeedd00
    wire = pallas_pack_response(jnp.asarray(states),jnp.asarray(targets),state_len=state_len,interpret=True)
    expected = states.copy()
    expected[:,state_len:] = 0
    for byte in range(4):
        expected[:,state_len+byte] = ((targets[0]>>(byte*8))&255).astype(np.uint8)
    np.testing.assert_array_equal(wire,expected)
    clean,decoded = pallas_unpack_response(wire,state_len=state_len,interpret=True)
    expected[:,state_len:] = 0
    np.testing.assert_array_equal(clean,expected)
    np.testing.assert_array_equal(decoded,targets)
