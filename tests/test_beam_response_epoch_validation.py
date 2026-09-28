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


def test_self_epoch_diagnostic_separates_preparation_from_output_corruption():
    from benchmarks.beam_response_epoch_probe import diagnose_self_epoch
    wire=np.zeros((8,2048,128),np.uint8)
    grouped=np.zeros((8,35,2048),np.uint32)
    intervals=np.zeros((8,3,128),np.uint32)
    error=np.zeros((8,1,128),np.uint32)
    for rank in range(8):
        intervals[rank,1,rank]=129
        intervals[rank,0,rank+1:8]=129
    want_wire=np.zeros((8,1024,128),np.uint8)
    got_wire=want_wire.copy()
    got_wire[2,0,7]=9
    control=np.zeros((8,2,128),np.uint32)
    report=diagnose_self_epoch(wire,(grouped,intervals,error),
        (got_wire,control),(want_wire,control))
    assert report['grouped_live_word_mismatches']==[0]*8
    assert report['interval_mismatches']==[0]*8
    assert report['wire_mismatches_by_byte_column'][7]==1
    assert report['first_wire_mismatches']==[[2,0,7,9,0]]


def test_self_packet_diagnostic_reports_corrupted_word_and_control():
    from benchmarks.beam_response_epoch_probe import diagnose_self_packet
    wire=np.zeros((8,2048,128),np.uint8)
    packet=np.zeros((8,8,32,128),np.uint32)
    control=np.zeros((8,8,2,128),np.uint32)
    for rank in range(8):
        control[rank,rank,0,0]=128
    packet[3,3,1,4]=1
    control[5,5,0,0]=127
    result=diagnose_self_packet(packet,control,wire)
    assert result['packet_word_mismatches']==[0,0,0,1,0,0,0,0]
    assert result['packet_control_mismatches']==[0,0,0,0,0,1,0,0]
