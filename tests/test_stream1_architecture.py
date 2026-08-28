from __future__ import annotations

import jax.numpy as jnp
import pytest

from tpu_beam_search.stream1_architecture import (
    InputEncodingKind,
    NormalizationKind,
    Stream1Architecture,
)


def _artgor_params(*, residual_count: int = 2):
    hidden = 16
    params = {
        "encoding": "embedding",
        "state_size": 6,
        "num_classes": 7,
        "d_model": hidden,
        "output_dim": 4,
        "embed": jnp.zeros((7, 3), dtype=jnp.float32),
        "input_stack": [
            {
                "lin_w": jnp.zeros((18, hidden), dtype=jnp.float32),
                "lin_b": jnp.zeros((hidden,), dtype=jnp.float32),
                "ln_gamma": jnp.ones((hidden,), dtype=jnp.float32),
                "ln_beta": jnp.zeros((hidden,), dtype=jnp.float32),
            }
        ],
        "res_blocks": [],
        "head_w": jnp.zeros((hidden, 4), dtype=jnp.float32),
        "head_b": jnp.zeros((4,), dtype=jnp.float32),
    }
    for _ in range(residual_count):
        params["res_blocks"].append(
            {
                "lin1_w": jnp.zeros((hidden, hidden), dtype=jnp.float32),
                "lin1_b": jnp.zeros((hidden,), dtype=jnp.float32),
                "ln1_gamma": jnp.ones((hidden,), dtype=jnp.float32),
                "ln1_beta": jnp.zeros((hidden,), dtype=jnp.float32),
                "lin2_w": jnp.zeros((hidden, hidden), dtype=jnp.float32),
                "lin2_b": jnp.zeros((hidden,), dtype=jnp.float32),
                "ln2_gamma": jnp.ones((hidden,), dtype=jnp.float32),
                "ln2_beta": jnp.zeros((hidden,), dtype=jnp.float32),
            }
        )
    return params


def test_existing_bn_constructor_remains_source_compatible():
    architecture = Stream1Architecture(
        STATE_LEN=120,
        STATE_STORAGE_LEN=128,
        NUM_CLASSES=120,
        HIDDEN1=1536,
        HIDDEN2=512,
        RESIDUAL_COUNT=2,
        MOVE_COUNT=24,
    )
    assert architecture.NORMALIZATION is NormalizationKind.FOLDED_BATCH_NORM
    assert architecture.INPUT_ENCODING is InputEncodingKind.VIRTUAL_ONE_HOT_MXU
    assert architecture.EMBED_DIM == 0


def test_artgor_architecture_is_derived_from_parameter_shapes():
    architecture = Stream1Architecture.from_artgor_params(
        _artgor_params(), STATE_STORAGE_LEN=8
    )
    assert architecture.STATE_LEN == 6
    assert architecture.STATE_STORAGE_LEN == 8
    assert architecture.NUM_CLASSES == 7
    assert architecture.EMBED_DIM == 3
    assert architecture.HIDDEN1 == architecture.HIDDEN2 == 16
    assert architecture.RESIDUAL_COUNT == 2
    assert architecture.MOVE_COUNT == 4
    assert architecture.NORMALIZATION is NormalizationKind.LAYER_NORM
    assert architecture.INPUT_ENCODING is InputEncodingKind.EMBEDDING_GATHER


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda p: p.__setitem__("embed", p["embed"].reshape(1, 21)), "embedding"),
        (lambda p: p["input_stack"].append(p["input_stack"][0]), "input_stack"),
        (lambda p: p["res_blocks"][0].update(lin2_w=jnp.zeros((15, 16))), "residual"),
        (lambda p: p.update(head_w=jnp.zeros((15, 4))), "head"),
    ],
)
def test_artgor_architecture_rejects_inconsistent_shapes(mutation, message):
    params = _artgor_params()
    mutation(params)
    with pytest.raises(ValueError, match=message):
        Stream1Architecture.from_artgor_params(params, STATE_STORAGE_LEN=8)
