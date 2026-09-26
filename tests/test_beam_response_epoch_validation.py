import numpy as np


def _reference():
    wire = np.zeros((8, 1024, 128), np.uint8)
    control = np.zeros((8, 2, 128), np.uint32)
    wire[3, 5, 120] = 255
    control[3, 0, 0] = 1
    return wire, control


def test_response_epoch_validator_accepts_exact_byte_and_control_abi():
    from benchmarks.beam_response_epoch_probe import validate_epoch

    reference = _reference()
    result = validate_epoch(reference, reference)
    assert result["exact"]
    assert result["shape_ok"] and result["dtype_ok"] and result["hash_ok"]
    assert result["mismatches"] == [[0] * 8, [0] * 8]


def test_response_epoch_validator_rejects_equal_values_with_wrong_dtype():
    from benchmarks.beam_response_epoch_probe import validate_epoch

    wire, control = _reference()
    result = validate_epoch((wire.astype(np.float32), control), (wire, control))
    assert not result["exact"]
    assert not result["dtype_ok"]


def test_response_epoch_validator_rejects_shape_and_bit_corruption():
    from benchmarks.beam_response_epoch_probe import validate_epoch

    wire, control = _reference()
    assert not validate_epoch((wire[:, :1], control), (wire, control))["exact"]
    bad_wire = wire.copy()
    bad_wire[3, 5, 120] ^= np.uint8(0x80)
    assert not validate_epoch((bad_wire, control), (wire, control))["exact"]
    bad_control = control.copy()
    bad_control[3, 0, 0] ^= np.uint32(0x80000000)
    result = validate_epoch((wire, bad_control), (wire, control))
    assert not result["exact"]
    assert not result["hash_ok"]
