from __future__ import annotations

import jax
from jax.sharding import PartitionSpec as P


def make_sharded_inference(local_infer, *, mesh, weights_example):
    """Map independent state shards while replicating model weights."""

    weight_specs = jax.tree.map(lambda _: P(), weights_example)
    return jax.jit(
        jax.shard_map(
            local_infer,
            mesh=mesh,
            in_specs=(P("core", None), weight_specs),
            out_specs=P("core", None),
            check_vma=False,
        )
    )
