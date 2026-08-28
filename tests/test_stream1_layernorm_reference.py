from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_beam_search.artgor_reference import artgor_reference_apply
from tpu_beam_search.stream1_architecture import (
    InputEncodingKind,
    Stream1Architecture,
)
from tpu_beam_search.stream1_layernorm_reference import (
    layernorm_stream1_weights_from_artgor_params,
    stream1_layernorm_reference_inference,
)
from tpu_beam_search.stream1_inference import make_jitted_stream1_inference


def _params():
    key_values = jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 7
    return {
        "encoding": "embedding",
        "state_size": 2,
        "num_classes": 5,
        "d_model": 8,
        "output_dim": 4,
        "embed": key_values,
        "input_stack": [
            {
                "lin_w": jnp.arange(48, dtype=jnp.float32).reshape(6, 8) / 31,
                "lin_b": jnp.linspace(-0.3, 0.3, 8),
                "ln_gamma": jnp.linspace(0.7, 1.3, 8),
                "ln_beta": jnp.linspace(-0.2, 0.2, 8),
            }
        ],
        "res_blocks": [
            {
                "lin1_w": jnp.arange(64, dtype=jnp.float32).reshape(8, 8) / 53,
                "lin1_b": jnp.linspace(-0.2, 0.2, 8),
                "ln1_gamma": jnp.linspace(0.8, 1.2, 8),
                "ln1_beta": jnp.linspace(-0.1, 0.1, 8),
                "lin2_w": jnp.flip(
                    jnp.arange(64, dtype=jnp.float32).reshape(8, 8), axis=1
                ) / 47,
                "lin2_b": jnp.linspace(0.2, -0.2, 8),
                "ln2_gamma": jnp.linspace(1.2, 0.8, 8),
                "ln2_beta": jnp.linspace(0.1, -0.1, 8),
            }
        ],
        "head_w": jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 19,
        "head_b": jnp.linspace(-0.4, 0.4, 4),
    }


@pytest.mark.parametrize("encoding", list(InputEncodingKind))
def test_layernorm_reference_encodings_match_artgor_oracle(encoding):
    params = _params()
    architecture = Stream1Architecture.from_artgor_params(
        params, STATE_STORAGE_LEN=4
    )
    architecture = replace(architecture, INPUT_ENCODING=encoding)
    weights = layernorm_stream1_weights_from_artgor_params(params, architecture)
    states = jnp.asarray(
        [[0, 1, 99, 88], [4, 3, 77, 66], [2, 2, 55, 44]], dtype=jnp.uint8
    )

    expected = artgor_reference_apply(
        params, states[:, : architecture.STATE_LEN], dtype=jnp.bfloat16
    )
    actual = stream1_layernorm_reference_inference(states, weights, architecture)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_layernorm_reference_validates_weight_shapes():
    params = _params()
    architecture = Stream1Architecture.from_artgor_params(
        params, STATE_STORAGE_LEN=2
    )
    params["head_w"] = jnp.zeros((7, 4), dtype=jnp.float32)
    with pytest.raises(ValueError, match="head"):
        layernorm_stream1_weights_from_artgor_params(params, architecture)


def test_universal_jitted_reference_dispatches_layernorm_statically():
    params = _params()
    architecture = Stream1Architecture.from_artgor_params(
        params, STATE_STORAGE_LEN=2
    )
    weights = layernorm_stream1_weights_from_artgor_params(params, architecture)
    states = jnp.asarray([[0, 1], [4, 3]], dtype=jnp.uint8)
    inference = make_jitted_stream1_inference(architecture, backend="reference")

    actual = inference(states, weights)
    original_jit = jax.jit(
        lambda values: artgor_reference_apply(
            params, values, dtype=jnp.bfloat16
        )
    )
    expected = original_jit(states)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
