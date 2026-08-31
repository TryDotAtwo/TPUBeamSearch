"""Real CPU Pallas interpreter witnesses, not TPU compilation/accuracy proof."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import tpu_beam_search.stream1_layernorm_experimental as experimental_kernels


@pytest.fixture
def kernels():
    return experimental_kernels


def _inputs(width, rows=3):
    rng = np.random.default_rng(71)
    return tuple(jnp.asarray(rng.normal(size=shape), jnp.bfloat16)
                 for shape in ((rows, width), (width,), (width,)))


def _cpu_expression(values, scale, bias, epsilon, arithmetic):
    """Independent unpadded expression; no production helper or Pallas kernel."""
    x, scale, bias = (v.astype(jnp.bfloat16) for v in (values, scale, bias))
    width = x.shape[1]
    if arithmetic == "legacy_bf16":
        mean = jnp.sum(x, axis=1, keepdims=True) / width
        centered = x - mean
        variance = jnp.sum(centered * centered, axis=1, keepdims=True) / width
        return (centered * jax.lax.rsqrt(variance + epsilon) * scale + bias)
    xf = x.astype(jnp.float32)
    mean = (jnp.sum(xf, axis=1, keepdims=True) / width).astype(jnp.bfloat16)
    centered = xf - mean.astype(jnp.float32)
    variance = (jnp.sum(centered * centered, axis=1, keepdims=True) / width
                ).astype(jnp.bfloat16)
    eps = jnp.asarray(epsilon, jnp.bfloat16).astype(jnp.float32)
    inv = jax.lax.rsqrt(variance.astype(jnp.float32) + eps).astype(jnp.bfloat16)
    return ((centered * inv.astype(jnp.float32)) * scale.astype(jnp.float32)
            + bias.astype(jnp.float32)).astype(jnp.bfloat16)


@pytest.mark.parametrize("arithmetic", ["legacy_bf16", "hlo_mixed"])
@pytest.mark.parametrize("width", [130, 1024])
@pytest.mark.parametrize("mask_mode", ["all", "fp32_where", "direct_2d"])
def test_masked_ln_matches_unpadded_expression(kernels, arithmetic, width, mask_mode):
    # Catches population padding, missing BF16 stage casts, or changed affine order.
    x, scale, bias = _inputs(width)
    expected = _cpu_expression(x, scale, bias, 0.03719, arithmetic)
    actual = kernels.experimental_layer_norm(
        x, scale, bias, bm=2, arithmetic=arithmetic, mask_mode=mask_mode,
        epsilon=0.03719, interpret=True,
    )
    assert actual.shape == x.shape
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("arithmetic", ["legacy_bf16", "hlo_mixed"])
@pytest.mark.parametrize("mask_mode", ["none", "input", "center", "output"])
def test_single_mask_arms_preserve_unpadded_operator(kernels, arithmetic, mask_mode):
    # Catches a site-specific arm accidentally omitting computation, not just mask.
    x, scale, bias = _inputs(128)
    actual = kernels.experimental_layer_norm(
        x, scale, bias, bm=2, arithmetic=arithmetic, mask_mode=mask_mode,
        interpret=True,
    )
    np.testing.assert_array_equal(actual, _cpu_expression(x, scale, bias, 1e-5, arithmetic))


@pytest.mark.parametrize("mask_mode", ["none", "input", "center", "output"])
def test_partial_mask_arms_reject_padded_population(kernels, mask_mode):
    with pytest.raises(ValueError, match="unpadded|logical.*padded"):
        kernels.experimental_layer_norm(*_inputs(130), mask_mode=mask_mode, interpret=True)


@pytest.mark.parametrize("arithmetic", ["legacy_bf16", "hlo_mixed"])
@pytest.mark.parametrize("mask_mode", ["all", "fp32_where", "direct_2d"])
def test_raw_kernel_excludes_poisoned_tail_and_zeros_affine_tail(kernels, arithmetic, mask_mode):
    # Output slicing must not hide polluted reductions or nonzero invalid lanes.
    x, scale, bias = _inputs(130, rows=2)
    xp = jnp.pad(x, ((0, 0), (0, 126)), constant_values=jnp.nan)
    sp = jnp.pad(scale, (0, 126), constant_values=2)
    bp = jnp.pad(bias, (0, 126), constant_values=7)
    call = pl.pallas_call(
        functools.partial(kernels._experimental_layer_norm_kernel, logical_width=130,
                          epsilon=0.03719, arithmetic=arithmetic, mask_mode=mask_mode),
        out_shape=jax.ShapeDtypeStruct((2, 256), jnp.bfloat16), interpret=True,
    )
    actual = call(xp, sp, bp)
    np.testing.assert_array_equal(actual[:, :130],
                                  _cpu_expression(x, scale, bias, 0.03719, arithmetic))
    np.testing.assert_array_equal(actual[:, 130:], 0)


def test_mixed_schedule_is_not_all_fp32_or_legacy_bf16(kernels):
    # The fixture discriminates the hypothesis from both tempting shortcuts.
    x, scale, bias = _inputs(130)
    mixed = kernels.experimental_layer_norm(*[x, scale, bias], bm=2,
                                            arithmetic="hlo_mixed", interpret=True)
    xf = x.astype(jnp.float32)
    centered = xf - jnp.mean(xf, axis=1, keepdims=True)
    all_fp32 = (centered * jax.lax.rsqrt(jnp.mean(centered**2, axis=1, keepdims=True)
                                       + 1e-5) * scale.astype(jnp.float32)
                + bias.astype(jnp.float32)).astype(jnp.bfloat16)
    assert np.any(np.asarray(mixed) != np.asarray(all_fp32))
    assert np.any(np.asarray(mixed) != np.asarray(
        _cpu_expression(x, scale, bias, 1e-5, "legacy_bf16")))


def test_float32_inputs_are_quantized_at_bf16_input_boundary(kernels):
    x, scale, bias = (v.astype(jnp.float32) + 0.0013 for v in _inputs(130))
    actual = kernels.experimental_layer_norm(x, scale, bias, bm=2, interpret=True)
    np.testing.assert_array_equal(actual, _cpu_expression(x, scale, bias, 1e-5, "legacy_bf16"))


@pytest.mark.parametrize("kwargs,match", [
    ({"arithmetic": "bad"}, "arithmetic"), ({"mask_mode": "bad"}, "mask_mode"),
    ({"bm": 0}, "bm"), ({"alignment": 0}, "alignment"),
    ({"epsilon": -1}, "epsilon"), ({"epsilon": float("nan")}, "epsilon"),
])
def test_ln_rejects_invalid_options(kernels, kwargs, match):
    with pytest.raises(ValueError, match=match):
        kernels.experimental_layer_norm(*_inputs(128), interpret=True, **kwargs)


@pytest.mark.parametrize("shapes", [((130,), (130,), (130,)),
                                    ((2, 130), (129,), (130,)),
                                    ((2, 130), (130,), (1, 130)),
                                    ((0, 130), (130,), (130,)),
                                    ((2, 0), (0,), (0,))])
def test_ln_rejects_invalid_shapes(kernels, shapes):
    with pytest.raises(ValueError):
        kernels.experimental_layer_norm(*(jnp.zeros(s) for s in shapes), interpret=True)


@pytest.mark.parametrize("operand_dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("predicate_layout", ["broadcast", "direct_2d"])
@pytest.mark.parametrize("width", [130, 1024])
def test_minimal_select_has_nonconstant_column_predicate(kernels, operand_dtype, predicate_layout, width):
    # Fails if an all-valid predicate, wrong axis, or BF16 rounding contaminates FP32 arm.
    values = jnp.arange(3 * width, dtype=jnp.float32).reshape(3, width) + 0.013
    dtype = jnp.bfloat16 if operand_dtype == "bf16" else jnp.float32
    expected = np.asarray(values.astype(dtype)).copy()
    expected[:, 1::2] = 0
    actual = kernels.minimal_predicate_select(
        values, operand_dtype=operand_dtype, predicate_layout=predicate_layout,
        bm=2, interpret=True,
    )
    assert actual.shape == values.shape
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kwargs", [{"operand_dtype": "bad"},
                                    {"predicate_layout": "bad"}, {"bm": 0}])
def test_minimal_select_rejects_invalid_options(kernels, kwargs):
    with pytest.raises(ValueError):
        kernels.minimal_predicate_select(jnp.ones((2, 128)), interpret=True, **kwargs)


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.complex64, jnp.bool_])
def test_ln_rejects_nonreal_input_dtype(kernels, dtype):
    with pytest.raises(ValueError, match="floating"):
        kernels.experimental_layer_norm(jnp.ones((2, 128), dtype),
                                         jnp.ones(128), jnp.zeros(128), interpret=True)


@pytest.mark.parametrize("shape", [(128,), (0, 128), (2, 0)])
def test_minimal_select_rejects_nonmatrix_or_empty_shape(kernels, shape):
    with pytest.raises(ValueError):
        kernels.minimal_predicate_select(jnp.ones(shape), interpret=True)


@pytest.mark.parametrize("probe", ["ln", "minimal"])
@pytest.mark.parametrize("direct", [False, True])
def test_control_and_direct_probes_emit_distinct_predicate_ranks(kernels, probe, direct):
    # These arms isolate a layout failure: equal values do not prove the required
    # rank1-xi1 control versus rank2-integer comparison was actually constructed.
    x, scale, bias = _inputs(130, rows=2)
    if probe == "ln":
        fn = lambda x: kernels.experimental_layer_norm(
            x, scale, bias, bm=2, mask_mode="direct_2d" if direct else "all", interpret=True)
    else:
        fn = lambda x: kernels.minimal_predicate_select(
            x, bm=2, predicate_layout="direct_2d" if direct else "broadcast", interpret=True)
    traced = jax.make_jaxpr(fn)(x)
    call = next(eqn for eqn in traced.jaxpr.eqns if eqn.primitive.name == "pallas_call")
    compare = next(eqn for eqn in call.params["jaxpr"].eqns if eqn.primitive.name == "lt")
    assert compare.outvars[0].aval.shape == ((2, 256) if direct else (256,))
