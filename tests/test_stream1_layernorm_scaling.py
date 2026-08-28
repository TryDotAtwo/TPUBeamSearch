import numpy as np

from benchmarks.stream1_layernorm_scaling import (
    NUM_CLASSES,
    STATE_SIZE,
    valid_state_tile,
)


def test_scaling_state_tile_is_valid_and_diverse():
    assert NUM_CLASSES == 150
    states = valid_state_tile(256)
    assert states.shape == (256, STATE_SIZE)
    assert states.dtype == np.uint8
    assert int(states.min()) >= 0
    assert int(states.max()) < NUM_CLASSES
    assert np.unique(states, axis=0).shape[0] == 256
