"""Opt-in operator/control builders; no production defaults are changed."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tpu_beam_search.stream1_layernorm_reference import layer_norm_reference


def dense_configs():
    configs = [dict(id=f"jax-{boundary}", dense="jax", boundary=boundary, control=True,
                    bm=128, bk=256, bn=512) for boundary in ("none", "pre", "post", "both")]
    for bm, bk, bn in ((128, 256, 512), (256, 256, 512), (512, 256, 512),
                       (128, 256, 1024), (128, 1024, 512)):
        configs.append(dict(id=f"late-m{bm}-k{bk}-n{bn}", dense="late", boundary="none",
                            control=False, bm=bm, bk=bk, bn=bn,
                            changes_arithmetic_schedule=bk != 256))
    return configs


def full_configs():
    base = dict(dense="jax", boundary="none", norm="jax", embedding="reference",
                bm=128, bk=256, bn=512, control=False)
    configs = [{**base, **c} for c in dense_configs()]
    for arithmetic, mode in (("legacy_bf16", "none"), ("hlo_mixed", "none"),
                              ("hlo_mixed", "direct_2d")):
        configs.append(dict(base, id=f"ln-{arithmetic}-{mode}", norm="experimental",
                            arithmetic=arithmetic, mask_mode=mode))
    for embedding in ("jax_flat", "jax_tiled", "pallas_banked"):
        configs.append({**base, "id": f"embedding-{embedding}", "embedding": embedding})
    return configs


def candidate_dense(x, w, b, config, *, interpret=False):
    boundary = config.get("boundary", "none")
    if boundary not in ("none", "pre", "post", "both"):
        raise ValueError("unknown Dense boundary")
    if boundary in ("pre", "both"):
        x = jax.lax.optimization_barrier(x)
    if config["dense"] == "jax":
        result = x @ w + b
    elif config["dense"] == "late":
        from tpu_beam_search.stream1_layernorm_pallas import pallas_layernorm_dense
        result = pallas_layernorm_dense(x, w, b, bm=config["bm"], bk=config["bk"], bn=config["bn"],
                                        dense_rounding="late", interpret=interpret)
    else:
        raise ValueError("unknown Dense implementation")
    if boundary in ("post", "both"):
        result = jax.lax.optimization_barrier(result)
    return result


def candidate_full(config, architecture, *, interpret=False):
    def call(states, weights):
        from tpu_beam_search.stream1_embedding_experimental import (
            flat_embedding, flat_embedding_prepacked,
        )
        from tpu_beam_search.stream1_layernorm_experimental import experimental_layer_norm
        epsilon = architecture.LAYER_NORM_EPSILON
        states = states[:, :architecture.STATE_LEN]
        encoding = config.get("embedding", "reference")
        if encoding == "reference":
            encoded = weights.embedding.astype(jnp.bfloat16)[states.astype(jnp.int32)].reshape(states.shape[0], -1)
        elif encoding == "pallas_banked_prepacked":
            encoded = flat_embedding_prepacked(
                states, weights.embedding, embed_dim=architecture.EMBED_DIM,
                bm=config.get("bm", 128), interpret=interpret,
            )
        else:
            encoded = flat_embedding(states, weights.embedding, implementation=encoding,
                                      bm=config.get("bm", 128), interpret=interpret)
        input_dense = encoded @ weights.input.dense.weight + weights.input.dense.bias
        hidden = jax.nn.relu(layer_norm_reference(input_dense, weights.input.normalization, epsilon=epsilon))

        def layer(x, params):
            dense = candidate_dense(x, params.dense.weight, params.dense.bias, config, interpret=interpret)
            if config.get("norm", "jax") == "jax":
                return layer_norm_reference(dense, params.normalization, epsilon=epsilon)
            if config["norm"] != "experimental":
                raise ValueError("unknown LN implementation")
            return experimental_layer_norm(dense, params.normalization.scale, params.normalization.bias,
                epsilon=epsilon, bm=config.get("bm", 128), arithmetic=config["arithmetic"],
                mask_mode=config["mask_mode"], interpret=interpret)

        for block in weights.residuals:
            branch = jax.nn.relu(layer(hidden, block.first))
            branch = layer(branch, block.second)
            hidden = jax.nn.relu(hidden + branch)
        return hidden @ weights.output.weight + weights.output.bias
    return call


def jax_ln_observe(values, normalization, epsilon):
    """Same source expression with observed nodes; final TPU fusion may change."""
    mean = jnp.mean(values, axis=-1, keepdims=True)
    centered = values - mean
    variance = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    inv = jax.lax.rsqrt(variance + epsilon)
    output = centered * inv * normalization.scale + normalization.bias
    return dict(mean=mean, variance=variance, invstd=inv, output=output)


def mismatch_witnesses(reference, candidate, limit=16):
    """Bounded direct witnesses, preserving original dtype/bit patterns in JSON."""
    ref, cand = np.asarray(reference), np.asarray(candidate)
    if ref.shape != cand.shape or not isinstance(limit, int) or limit < 0:
        raise ValueError("witness shapes must match and limit must be nonnegative")
    r32, c32 = ref.astype(np.float32), cand.astype(np.float32)
    indices = np.flatnonzero(r32.ravel() != c32.ravel())
    def scalar(value):
        return float(value) if np.isfinite(float(value)) else None
    examples = []
    for index in indices[:limit]:
        coord = tuple(int(x) for x in np.unravel_index(index, ref.shape))
        examples.append(dict(index=list(coord), reference=scalar(ref[coord]), candidate=scalar(cand[coord]),
            reference_bytes_hex=ref[coord].tobytes().hex(), candidate_bytes_hex=cand[coord].tobytes().hex()))
    return dict(shape=list(ref.shape), reference_dtype=str(ref.dtype), candidate_dtype=str(cand.dtype),
                mismatch_count=int(indices.size), nonfinite_reference=int(np.count_nonzero(~np.isfinite(r32))),
                nonfinite_candidate=int(np.count_nonzero(~np.isfinite(c32))), examples=examples)


def node_summaries(nodes):
    result = {}
    for key, node in nodes.items():
        values = np.asarray(node, np.float32)
        result[key] = dict(shape=list(node.shape), dtype=str(node.dtype),
            finite=bool(np.isfinite(values).all()),
            sample=[float(x) if np.isfinite(x) else None for x in values.reshape(-1)[:16]])
    return result


def validate_provenance(context, previous):
    for key in ("checkpoint_sha256", "original_source_sha256", "puzzle_sha256", "input_sha256"):
        if key not in context or key not in previous or context[key] != previous[key]:
            raise ValueError(f"provenance differs from follow-up v1: {key}")
