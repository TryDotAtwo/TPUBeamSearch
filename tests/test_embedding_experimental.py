"""Exact lookup contracts; interpreter success is not TPU compilation evidence."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def module():
    return importlib.import_module("tpu_beam_search.stream1_embedding_experimental")


@pytest.mark.parametrize("implementation", ["jax_flat", "jax_tiled", "pallas_banked"])
def test_position_major_lookup_and_runtime_table(implementation):
    table = jnp.asarray([[1., 2.], [3., 4.], [5., 6.]], jnp.float32)
    states = jnp.array([[2, 0], [1, 2], [0, 1]], jnp.uint8)
    call = jax.jit(lambda s, e: module().flat_embedding(
        s, e, implementation=implementation, bm=8, interpret=True))
    got = call(states, table)
    assert got.dtype == jnp.bfloat16
    np.testing.assert_array_equal(got, [[5., 6., 1., 2.], [3., 4., 5., 6.], [1., 2., 3., 4.]])
    np.testing.assert_array_equal(call(states, table + 8), np.asarray(got, np.float32) + 8)


@pytest.mark.parametrize("storage_dtype", [jnp.bfloat16, jnp.float32])
def test_prepacked_banks_preserve_bf16_conversion_order_and_hand_layout(storage_dtype):
    # Non-BF16 values make the conversion boundary observable.  The expected
    # bank formula is independent of the implementation under test.
    table = (np.arange(150 * 24, dtype=np.float32).reshape(150, 24) / 17) + 0.0013
    banks = module().prepare_banked_embedding(jnp.asarray(table), storage_dtype=storage_dtype)
    rounded = np.asarray(jnp.asarray(table, jnp.bfloat16), np.float32)

    assert banks.low.shape == banks.high.shape == (3, 128, 128)
    assert banks.low.dtype == banks.high.dtype == storage_dtype
    for phase, lane, category in ((0, 0, 0), (1, 17, 127), (2, 127, 149)):
        feature = (phase * 128 + lane) % 24
        bank, local_category = (banks.low, category) if category < 128 else (banks.high, category - 128)
        assert float(bank[phase, lane, local_category]) == float(
            jnp.asarray(rounded[category, feature], storage_dtype))
    # Padding is semantic zero, not an uninitialized or repeated category.
    assert float(banks.high[0, 0, 127]) == 0.0


@pytest.mark.parametrize("storage_dtype", [jnp.bfloat16, jnp.float32])
def test_prepacked_lookup_is_exact_and_does_not_receive_runtime_embedding(storage_dtype):
    states = np.tile(np.arange(150, dtype=np.uint8), (9, 1))
    table = (np.arange(150 * 24).reshape(150, 24) / 17).astype(np.float32)
    banks = module().prepare_banked_embedding(jnp.asarray(table), storage_dtype=storage_dtype)
    call = jax.jit(lambda s, b: module().flat_embedding_prepacked(
        s, b, embed_dim=24, bm=8, interpret=True))
    got = call(jnp.asarray(states), banks)
    expected = np.asarray(jnp.asarray(table, jnp.bfloat16))[states].reshape(9, 3600)

    assert got.shape == (9, 3600)
    np.testing.assert_array_equal(got, expected)


def test_runtime_and_prepacked_banked_paths_are_exactly_equivalent():
    table = jnp.arange(150 * 24, dtype=jnp.float32).reshape(150, 24) / 13
    states = jnp.asarray(np.random.default_rng(5).integers(0, 150, (11, 150), dtype=np.uint8))
    runtime = module().flat_embedding(states, table, implementation="pallas_banked", bm=8, interpret=True)
    banks = module().prepare_banked_embedding(table, storage_dtype=jnp.float32)
    prepacked = module().flat_embedding_prepacked(states, banks, embed_dim=24, bm=8, interpret=True)
    np.testing.assert_array_equal(prepacked, runtime)


@pytest.mark.parametrize("change", ["dtype", "shape", "phases", "embed_dim", "bank_type"])
def test_invalid_prepacked_contract_is_rejected(change):
    states = jnp.zeros((2, 150), jnp.uint8)
    banks = module().prepare_banked_embedding(jnp.zeros((150, 24), jnp.float32))
    embed_dim = 24
    if change == "dtype":
        banks = type(banks)(banks.low, banks.high.astype(jnp.int32))
    elif change == "shape":
        banks = type(banks)(banks.low, banks.high[:, :, :-1])
    elif change == "phases":
        banks = type(banks)(banks.low[:2], banks.high[:2])
    elif change == "embed_dim":
        embed_dim = 0
    else:
        banks = (banks.low, banks.high)
    with pytest.raises((TypeError, ValueError)):
        module().flat_embedding_prepacked(states, banks, embed_dim=embed_dim, bm=8, interpret=True)


@pytest.mark.parametrize("implementation", ["jax_flat", "jax_tiled", "pallas_banked"])
def test_checkpoint_shape_class149_state_bank_and_tail(implementation):
    states = np.tile(np.arange(150, dtype=np.uint8), (9, 1))
    table = (np.arange(150 * 24).reshape(150, 24) / 17).astype(np.float32)
    expected = np.asarray(jnp.asarray(table, jnp.bfloat16))[states].reshape(9, 3600)
    got = module().flat_embedding(jnp.asarray(states), jnp.asarray(table),
        implementation=implementation, bm=8, interpret=True)
    assert got.shape == (9, 3600)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("change", ["signed", "empty", "unknown", "zero_tile", "rank", "integer_table"])
def test_invalid_static_contract_is_rejected(change):
    states, table = jnp.zeros((2, 150), jnp.uint8), jnp.zeros((150, 24), jnp.float32)
    kwargs = dict(implementation="jax_flat", bm=8)
    if change == "signed": states = states.astype(jnp.int8)
    if change == "empty": states = states[:0]
    if change == "unknown": kwargs["implementation"] = "unknown"
    if change == "zero_tile": kwargs["bm"] = 0
    if change == "rank": table = table.reshape(-1)
    if change == "integer_table": table = table.astype(jnp.int32)
    with pytest.raises(ValueError):
        module().flat_embedding(states, table, **kwargs)


def test_kernel_gathers_use_supported_mosaic_contract_not_general_take():
    # Characterize our lowering boundary against pinned JAX0.10.2's documented
    # source restriction. This is not a TPU compile or layout-safety proof.
    closed = jax.make_jaxpr(lambda s, e: module().flat_embedding(
        s, e, implementation="pallas_banked"))(
            jnp.zeros((128, 150), jnp.uint8), jnp.zeros((150, 24), jnp.float32))
    kernel = next(e.params["jaxpr"] for e in closed.jaxpr.eqns if e.primitive.name == "pallas_call")
    def gathers(graph):
        for eqn in graph.eqns:
            if eqn.primitive.name == "gather":
                yield eqn
            if "jaxpr" in eqn.params:
                child = eqn.params["jaxpr"]
                yield from gathers(getattr(child, "jaxpr", child))
    found = list(gathers(kernel))
    assert len(found) == 3
    for eqn in found:
        p = eqn.params
        assert p["mode"] == jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS
        assert p["slice_sizes"] == (1, 1)
        assert p["dimension_numbers"].offset_dims == ()
        assert eqn.invars[1].aval.shape[:-1] == eqn.outvars[0].aval.shape
