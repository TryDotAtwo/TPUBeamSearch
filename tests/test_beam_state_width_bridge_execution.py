"""Physical-gate fixture contracts; local checks do not prove TPU execution."""
import numpy as np


def test_width_gate_oracle_covers_all_eight_distinct_devices():
    from benchmarks.beam_state_width_bridge_execution import make_inputs, expected

    parents, generators, requests, counts, targets, frontier = make_inputs()
    wire, published, errors, scatter_errors = expected(
        parents, generators, requests, counts, targets, frontier,
    )
    assert parents.shape == (8, 128, 160)
    assert generators.shape == (8, 30, 160)
    assert wire.shape == (8, 128, 160)
    assert published.shape == (8, 128, 160)
    assert errors.shape == (8, 2, 128)
    assert scatter_errors.shape == (8, 2, 128)
    assert len({parents[device].tobytes() for device in range(8)}) == 8
    assert all(np.any(wire[device, 0]) for device in range(8))
    assert not wire[:, 1:].any()
    assert not errors[:, 0].any()
    assert np.all(errors[:, 1, 0] == np.uint32(0xffffffff))
    assert not scatter_errors[:, 0].any()
    for device in range(8):
        parent = int(requests[device, 0, 0])
        move = int(requests[device, 3, 0] >> np.uint32(16))
        for position in (0, 71, 149):
            selected = generators[device, move, position]
            assert wire[device, 0, position] == parents[device, parent, selected]
        assert wire[device, 0, 150] == requests[device, 2, 0]
        assert not wire[device, 0, 154:].any()
        target = int(requests[device, 2, 0])
        np.testing.assert_array_equal(published[device, target, :150], wire[device, 0, :150])
        assert not published[device, target, 150:].any()
        np.testing.assert_array_equal(published[device, target + 1], frontier[device, target + 1])


def test_width_gate_validator_rejects_high_byte_corruption():
    from benchmarks.beam_state_width_bridge_execution import make_inputs, expected, validate

    arrays = make_inputs()
    wire, published, errors, scatter_errors = expected(*arrays)
    assert validate(wire, published, errors, scatter_errors,
                    wire, published, errors, scatter_errors)["exact"]
    broken = wire.copy()
    broken[7, 0, 149] ^= np.uint8(0x80)
    assert not validate(broken, published, errors, scatter_errors,
                        wire, published, errors, scatter_errors)["exact"]


def test_width_gate_sharded_wrapper_runs_real_pallas_interpreter():
    import jax
    from benchmarks.beam_state_width_bridge_execution import make_inputs, expected, make_sharded_call

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("core",))
    arrays = make_inputs()
    call = make_sharded_call(mesh, interpret=True)
    actual = call(*(jax.numpy.asarray(value[:1]) for value in arrays))
    want = expected(*arrays)
    for result, reference in zip(actual, want):
        np.testing.assert_array_equal(np.asarray(result), reference[:1])
