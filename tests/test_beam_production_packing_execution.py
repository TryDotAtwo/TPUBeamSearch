import numpy as np


def test_physical_gate_fixtures_cover_cross_tile_chunk_and_error():
    from benchmarks.beam_production_packing_execution import make_inputs, expected

    payload, intervals, chunks, prior = make_inputs()
    wire, control = expected(payload, intervals, chunks, prior)
    assert payload.shape == (8, 32, 2048)
    assert wire.shape == (8, 8, 32, 128)
    assert control.shape == (8, 8, 2, 128)
    np.testing.assert_array_equal(wire[0, 1, 0, :2], payload[0, 0, 127:129])
    np.testing.assert_array_equal(wire[1, 0, 0], payload[1, 0, 129:257])
    assert not wire[2].any()
    assert not wire[3:6].any()
    np.testing.assert_array_equal(control[3:6, :, 1, 0], np.ones((3, 8), np.uint32))
    assert not wire[7].any()


def test_physical_gate_uses_independent_payloads_on_each_core():
    from benchmarks.beam_production_packing_execution import make_inputs

    payload, _, _, _ = make_inputs()
    assert len({payload[rank].tobytes() for rank in range(8)}) == 8


def test_physical_gate_covers_35_plane_response_caller():
    from benchmarks.beam_production_packing_execution import make_inputs, expected

    arrays = make_inputs(planes=35)
    wire, control = expected(*arrays)
    assert arrays[0].shape == (8, 35, 2048)
    assert wire.shape == (8, 8, 35, 128)
    assert control.shape == (8, 8, 2, 128)
    np.testing.assert_array_equal(wire[6, 0, 34], arrays[0][6, 34, 1920:2048])


def test_all_live_fixture_exercises_every_device_and_peer():
    from benchmarks.beam_production_packing_execution import make_inputs, expected

    arrays = make_inputs(scenario="all_live")
    wire, control = expected(*arrays)
    assert np.all(control[:, :, 0, 0] == 128)
    assert not control[:, :, 1, :].any()
    for device in range(8):
        for peer in range(8):
            assert np.any(wire[device, peer])


def test_gate_rejects_dtype_mismatch_and_high_bit_corruption():
    from benchmarks.beam_production_packing_execution import make_inputs, expected, validate_outputs

    arrays = make_inputs()
    wire, control = expected(*arrays)
    assert np.uint32(0xffffffff) in arrays[0]
    assert np.uint32(0x80000000) in arrays[0]
    assert np.uint32(0x01000001) in arrays[0]
    assert validate_outputs(wire, control, wire, control)["exact"]
    assert not validate_outputs(wire.astype(np.float64), control, wire, control)["exact"]
    broken = wire.copy()
    broken[0, 0, 0, 1] ^= np.uint32(0x80000000)
    assert not validate_outputs(broken, control, wire, control)["exact"]


def test_shard_map_wrapper_keeps_one_payload_per_device():
    import jax
    from benchmarks.beam_production_packing_execution import make_inputs, expected, make_sharded_call

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("core",))
    call = make_sharded_call(mesh, interpret=True)
    arrays = make_inputs()
    got = call(*(jax.numpy.asarray(array[:1]) for array in arrays))
    want = expected(*arrays)
    for actual, reference in zip(got, want):
        np.testing.assert_array_equal(np.asarray(actual), reference[:1])
