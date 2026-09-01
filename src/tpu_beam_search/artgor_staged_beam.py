"""Staged exact inference integration for Artgor's Cube555 TPU beam.

Inference is executed as explicit prefix/head dispatch pairs.  Their BF16 Q
outputs are assembled on device and consumed by a search-only depth executable,
so the original 131072-parent streaming selection order can remain unchanged.

The search and host walkback are an attributed derivative of Andrey
Lukyanenko/Artgor's immutable Kaggle script version 344319112, frozen under
``third_party/artgor_cube555_v344319112``.  The one-depth test compares the
derived body with that source tensor for tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import os
import tempfile
import time
from typing import Any, Callable, Hashable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from .artgor_exact_inference import (
    ArtgorExactConfig,
    ArtgorExactInference,
    prepare_artgor_exact_inference,
)
from .stream1_layernorm_exact import stream1_layernorm_exact_prefix
from .stream1_layernorm_pallas import pallas_layernorm_dense


PACK_SIZE = 160
STATE_DTYPE = jnp.uint8
BPTR_PL_BITS = 24
BPTR_RANK_BITS = 3
BPTR_MOVE_BITS = 5
BPTR_RANK_SHIFT = BPTR_PL_BITS
BPTR_MOVE_SHIFT = BPTR_PL_BITS + BPTR_RANK_BITS
BPTR_PL_MASK = (1 << BPTR_PL_BITS) - 1
BPTR_RANK_MASK = (1 << BPTR_RANK_BITS) - 1
BPTR_MOVE_MASK = (1 << BPTR_MOVE_BITS) - 1
STATE_DTYPE_NP = np.uint8


def _sorted_contains(sorted_row, keys):
    n = sorted_row.shape[0]
    steps = int(np.ceil(np.log2(max(n, 2)))) + 1
    lo = jnp.zeros(keys.shape, dtype=jnp.int32)
    hi = jnp.full(keys.shape, n, dtype=jnp.int32)
    for _ in range(steps):
        mid = (lo + hi) >> 1
        midc = jnp.clip(mid, 0, n - 1)
        go_right = sorted_row[midc] < keys
        lo = jnp.where(go_right, mid + 1, lo)
        hi = jnp.where(go_right, hi, mid)
    pos = jnp.clip(lo, 0, n - 1)
    return sorted_row[pos] == keys


def _topk_smallest(values, k):
    neg_v, idx = jax.lax.top_k(-values, k)
    return -neg_v, idx


@dataclass(frozen=True)
class StagedDepthConfig:
    world_size: int
    b_local: int
    inference_chunk: int
    parent_chunk: int
    n_gen: int
    state_size: int

    def validate(self, *, require_published_geometry: bool = True) -> None:
        values = (
            self.world_size,
            self.b_local,
            self.inference_chunk,
            self.parent_chunk,
            self.n_gen,
            self.state_size,
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in values
        ):
            raise ValueError("staged depth geometry must use positive integers")
        if self.b_local % self.parent_chunk:
            raise ValueError("parent_chunk must divide B_local")
        if self.parent_chunk % self.inference_chunk:
            raise ValueError("inference_chunk must divide parent_chunk")
        if self.b_local > 1 << 24:
            raise ValueError("B_local exceeds the packed uint32 parent budget")
        if self.n_gen > 1 << 5:
            raise ValueError("n_gen exceeds the packed uint32 move budget")
        if require_published_geometry and (
            self.world_size != 8
            or self.n_gen != 30
            or self.state_size != 150
        ):
            raise ValueError(
                "published Artgor exact geometry is 8 cores, 30 moves, "
                "150 state bytes"
            )


@dataclass(frozen=True)
class StagedDepthExecutables:
    """Four explicit boundaries used by one staged beam depth."""

    config: StagedDepthConfig
    prefix_from_beam: Callable
    head: Callable
    assemble_q: Callable
    search_depth: Callable


@dataclass(frozen=True)
class ArtgorExactBeamRuntime:
    """Prepared replicated weights and the two exact inference dispatches."""

    inference: ArtgorExactInference
    weights: Any
    hidden_size: int
    prefix_local: Callable | None = None
    head_local: Callable | None = None


def prepare_artgor_exact_beam_runtime(
    params,
    *,
    mesh,
    exact_config: ArtgorExactConfig = ArtgorExactConfig(),
    state_storage_len: int = 150,
    interpret: bool = False,
) -> ArtgorExactBeamRuntime:
    inference, weights, architecture = prepare_artgor_exact_inference(
        params,
        mesh=mesh,
        config=exact_config,
        state_storage_len=state_storage_len,
        interpret=interpret,
    )

    def prefix_local(states, runtime_weights):
        return stream1_layernorm_exact_prefix(
            states,
            runtime_weights,
            architecture,
            bm=exact_config.prefix_bm,
            interpret=interpret,
        )

    def head_local(hidden, runtime_weights):
        return pallas_layernorm_dense(
            hidden,
            runtime_weights.output.weight,
            runtime_weights.output.bias,
            bm=exact_config.head_bm,
            bk=exact_config.head_bk,
            bn=exact_config.head_bn,
            dense_rounding=exact_config.dense_rounding,
            interpret=interpret,
        )

    return ArtgorExactBeamRuntime(
        inference=inference,
        weights=weights,
        hidden_size=architecture.HIDDEN2,
        prefix_local=prefix_local,
        head_local=head_local,
    )


def inference_chunk_starts(config: StagedDepthConfig) -> tuple[int, ...]:
    config.validate(require_published_geometry=False)
    return tuple(range(0, config.b_local, config.inference_chunk))


def parent_window_starts(config: StagedDepthConfig) -> tuple[int, ...]:
    config.validate(require_published_geometry=False)
    return tuple(range(0, config.b_local, config.parent_chunk))


def concatenate_q_chunks(chunks: Sequence) -> jnp.ndarray:
    if not chunks:
        raise ValueError("at least one Q chunk is required")
    arrays = tuple(jnp.asarray(chunk) for chunk in chunks)
    first = arrays[0]
    if first.ndim != 3 or min(first.shape) <= 0:
        raise ValueError("Q chunks must have shape [world, parents, moves]")
    if any(
        array.ndim != 3
        or array.shape[0] != first.shape[0]
        or array.shape[2] != first.shape[2]
        or array.dtype != first.dtype
        for array in arrays[1:]
    ):
        raise ValueError("Q chunks must agree on world, move, and dtype axes")
    return jnp.concatenate(arrays, axis=1)


def run_staged_depth(
    executables: StagedDepthExecutables,
    states,
    weights,
    *search_carry,
):
    """Score all parent chunks, assemble Q on device, then search once."""

    config = executables.config
    config.validate(require_published_geometry=False)
    q_chunks = []
    for start in inference_chunk_starts(config):
        hidden = executables.prefix_from_beam(
            states, weights, jnp.int32(start)
        )
        q_chunks.append(executables.head(hidden, weights))
    q_values = executables.assemble_q(tuple(q_chunks))
    return executables.search_depth(states, q_values, *search_carry)


def build_staged_depth_executables(
    exact_inference,
    weights_example,
    *,
    mesh,
    config: StagedDepthConfig,
    all_moves,
    V0,
    hash_vec,
    V0_hash,
    K_per_peer: int,
    owner_hash_vec=None,
    trace_prefix_hashes=None,
    history_depth: int = 0,
    history_bits: int = 0,
    history_exact: bool = True,
    eg_hashes=None,
    eg_ztab=None,
    inv_move_tbl=None,
    prefix_local: Callable | None = None,
    head_local: Callable | None = None,
    donate_search_carry: bool = True,
    require_published_geometry: bool = True,
) -> StagedDepthExecutables:
    """Compile the explicit prefix, head, Q-assembly, and search boundaries."""

    config.validate(
        require_published_geometry=require_published_geometry
    )
    axis_names = tuple(mesh.axis_names)
    if len(axis_names) != 1:
        raise ValueError("the staged Artgor engine requires a one-axis mesh")
    axis_name = axis_names[0]
    if int(mesh.size) != config.world_size:
        raise ValueError("mesh size must equal staged world_size")

    if (prefix_local is None) != (head_local is None):
        raise ValueError("prefix_local and head_local must be supplied together")
    if prefix_local is not None:
        weight_specs = jax.tree.map(lambda _: P(), weights_example)

        def prefix_from_beam_local(states, weights, parent_start):
            local_states = states[0]
            chunk = jax.lax.dynamic_slice(
                local_states,
                (parent_start, jnp.int32(0)),
                (config.inference_chunk, config.state_size),
            )
            return prefix_local(chunk, weights)[None, :, :]

        def head_local_explicit(hidden, weights):
            return head_local(hidden[0], weights)[None, :, :]

        prefix_from_beam = jax.jit(
            jax.shard_map(
                prefix_from_beam_local,
                mesh=mesh,
                in_specs=(P(axis_name), weight_specs, P()),
                out_specs=P(axis_name),
                check_vma=False,
            )
        )
        head = jax.jit(
            jax.shard_map(
                head_local_explicit,
                mesh=mesh,
                in_specs=(P(axis_name), weight_specs),
                out_specs=P(axis_name),
                check_vma=False,
            )
        )
    else:
        # Interpreter fixtures may inject a global explicit-axis callable.
        @jax.jit
        def prefix_from_beam(states, weights, parent_start):
            chunk = jax.lax.dynamic_slice(
                states,
                (jnp.int32(0), parent_start, jnp.int32(0)),
                (
                    config.world_size,
                    config.inference_chunk,
                    config.state_size,
                ),
            )
            return exact_inference.prefix(chunk, weights)

        head = exact_inference.head

    @jax.jit
    def assemble_q(chunks):
        return concatenate_q_chunks(chunks)

    search_body = build_precomputed_q_search_body(
        all_moves,
        V0,
        hash_vec,
        V0_hash,
        config.b_local,
        config.world_size,
        K_per_peer,
        config.n_gen,
        config.state_size,
        parent_chunk=config.parent_chunk,
        owner_hash_vec=owner_hash_vec,
        trace_prefix_hashes=trace_prefix_hashes,
        history_depth=history_depth,
        history_bits=history_bits,
        history_exact=history_exact,
        eg_hashes=eg_hashes,
        eg_ztab=eg_ztab,
        inv_move_tbl=inv_move_tbl,
        axis_name=axis_name,
    )
    mapped_search = jax.shard_map(
        search_body,
        mesh=mesh,
        in_specs=(P(axis_name),) * 9 + (P(),),
        out_specs=(P(axis_name),) * 13,
        check_vma=False,
    )
    if donate_search_carry:
        search_depth = jax.jit(
            mapped_search,
            donate_argnums=(0, 2, 3, 4, 5, 6, 7, 8),
        )
    else:
        search_depth = jax.jit(mapped_search)

    return StagedDepthExecutables(
        config=config,
        prefix_from_beam=prefix_from_beam,
        head=head,
        assemble_q=assemble_q,
        search_depth=search_depth,
    )


def _compile_staged_depth_executables(
    executables: StagedDepthExecutables,
    *,
    states,
    weights,
    hidden_size: int,
    search_carry: tuple,
    mesh,
) -> tuple[StagedDepthExecutables, float, float]:
    """Lower and compile every boundary without running a beam depth."""

    if not hasattr(executables.head, "lower"):
        return executables, 0.0, 0.0
    config = executables.config
    axis_name = tuple(mesh.axis_names)[0]
    sharding = NamedSharding(mesh, P(axis_name))
    hidden_aval = jax.ShapeDtypeStruct(
        (config.world_size, config.inference_chunk, hidden_size),
        jnp.bfloat16,
        sharding=sharding,
    )
    q_chunk_aval = jax.ShapeDtypeStruct(
        (config.world_size, config.inference_chunk, config.n_gen),
        jnp.bfloat16,
        sharding=sharding,
    )
    q_full_aval = jax.ShapeDtypeStruct(
        (config.world_size, config.b_local, config.n_gen),
        jnp.bfloat16,
        sharding=sharding,
    )

    lower_start = time.perf_counter()
    lowered_prefix = executables.prefix_from_beam.lower(
        states, weights, jnp.int32(0)
    )
    lowered_head = executables.head.lower(hidden_aval, weights)
    lowered_assemble = executables.assemble_q.lower(
        tuple(
            q_chunk_aval for _ in inference_chunk_starts(config)
        )
    )
    lowered_search = executables.search_depth.lower(
        states,
        q_full_aval,
        *search_carry,
    )
    lower_s = time.perf_counter() - lower_start

    compile_start = time.perf_counter()
    compiled = StagedDepthExecutables(
        config=config,
        prefix_from_beam=lowered_prefix.compile(),
        head=lowered_head.compile(),
        assemble_q=lowered_assemble.compile(),
        search_depth=lowered_search.compile(),
    )
    compile_s = time.perf_counter() - compile_start
    return compiled, lower_s, compile_s


def build_precomputed_q_search_body(
    all_moves,
    V0,
    hash_vec,
    V0_hash,
    B_local: int,
    world_size: int,
    K_per_peer: int,
    n_gen: int,
    state_size: int,
    *,
    parent_chunk: int,
    owner_hash_vec=None,
    trace_prefix_hashes=None,
    history_depth: int = 0,
    history_bits: int = 0,
    history_exact: bool = True,
    eg_hashes=None,
    eg_ztab=None,
    inv_move_tbl=None,
    axis_name: Hashable = "core",
):
    """Build Artgor's streaming search body with Q supplied as an input.

    The body is intentionally the original Phase-C Q-mode body after the model
    call has been replaced by a dynamic slice of ``q_values``.  Child order,
    routing, per-destination top-K, score packing, all-to-all, deduplication,
    history, solved checks, and the thirteen returned tensors stay unchanged.
    """

    BIG_F32 = jnp.float32(1e9)
    aB_local = K_per_peer * world_size
    trace_enabled = trace_prefix_hashes is not None
    if trace_enabled:
        trace_prefix_hashes = jnp.asarray(trace_prefix_hashes, dtype=jnp.int64)
    if B_local % parent_chunk:
        raise ValueError("parent_chunk must divide B_local")
    n_chunks = B_local // parent_chunk
    chunk_n = parent_chunk * n_gen

    arange_aB = jnp.arange(aB_local, dtype=jnp.int32)
    sender_rank_per_recv = (arange_aB // K_per_peer).astype(jnp.int8)

    hist_bits_n = history_bits or (
        int(np.ceil(np.log2(max(B_local, 2)))) + 5
    )
    hist_size = 1 << hist_bits_n
    hist_mask = hist_size - 1

    def beam_step_local(
        states,
        q_values,
        min_v_log,
        found_step,
        found_pos_local,
        found_pos_rank,
        verify_state,
        hist,
        in_move,
        j,
    ):
        states = states[0]
        q_values = q_values[0]
        min_v_log = min_v_log[0]
        found_step = found_step[0]
        found_pos_local = found_pos_local[0]
        found_pos_rank = found_pos_rank[0]
        verify_state = verify_state[0]
        hist = hist[0]
        in_move = in_move[0]
        rank_int = jax.lax.axis_index(axis_name).astype(jnp.int32)

        init_top_scores = jax.lax.pcast(
            jnp.full(
                (world_size, K_per_peer), BIG_F32, dtype=jnp.float32
            ),
            (axis_name,),
            to="varying",
        )
        init_top_pack = jax.lax.pcast(
            jnp.zeros(
                (world_size, K_per_peer, PACK_SIZE), dtype=jnp.uint8
            ),
            (axis_name,),
            to="varying",
        )

        def chunk_body(carry, chunk_i):
            top_scores, top_pack = carry
            parent_start = chunk_i * jnp.int32(parent_chunk)
            states_chunk = jax.lax.dynamic_slice(
                states,
                (parent_start, jnp.int32(0)),
                (parent_chunk, state_size),
            )
            children = states_chunk[:, all_moves].reshape(-1, state_size)
            child_v = jax.lax.dynamic_slice(
                q_values,
                (parent_start, jnp.int32(0)),
                (parent_chunk, n_gen),
            ).astype(jnp.float32).reshape(-1)

            if inv_move_tbl is not None:
                in_move_chunk = jax.lax.dynamic_slice(
                    in_move, (parent_start,), (parent_chunk,)
                )
                move_of_child = jnp.tile(
                    jnp.arange(n_gen, dtype=jnp.int32), parent_chunk
                )
                parent_move = jnp.repeat(
                    in_move_chunk.astype(jnp.int32), n_gen
                )
                banned = (parent_move >= 0) & (
                    move_of_child
                    == inv_move_tbl[
                        jnp.clip(parent_move, 0, n_gen - 1)
                    ]
                )
                child_v = jnp.where(banned, BIG_F32, child_v)

            if owner_hash_vec is not None:
                h_owner = jnp.sum(
                    children.astype(jnp.uint32) * owner_hash_vec, axis=1
                )
                owner = (
                    h_owner & jnp.uint32(world_size - 1)
                ).astype(jnp.int32)
            else:
                h = jnp.sum(
                    children.astype(jnp.int64) * hash_vec, axis=1
                )
                owner = (h % jnp.int64(world_size)).astype(jnp.int32)

            new_top_scores = top_scores
            new_top_pack = top_pack
            for dest in range(world_size):
                masked_scores = jnp.where(owner == dest, child_v, BIG_F32)
                merged_scores = jnp.concatenate(
                    [new_top_scores[dest], masked_scores], axis=0
                )
                top_v_new, keep = _topk_smallest(
                    merged_scores, K_per_peer
                )

                from_old = keep < K_per_peer
                old_idx = jnp.clip(keep, 0, K_per_peer - 1)
                new_idx = jnp.clip(
                    keep - K_per_peer, 0, chunk_n - 1
                )
                old_pack = new_top_pack[dest][old_idx]

                sel_states = children[new_idx]
                sel_parent_local = (
                    parent_start + (new_idx // n_gen)
                ).astype(jnp.int32)
                sel_move = (new_idx % n_gen).astype(jnp.int8)

                is_pad = top_v_new >= (BIG_F32 * 0.5)
                zero_state = jnp.zeros(
                    (K_per_peer, state_size), dtype=jnp.uint8
                )
                zero_int32 = jnp.zeros(K_per_peer, dtype=jnp.int32)
                zero_int8 = jnp.zeros(K_per_peer, dtype=jnp.int8)
                sel_states_u8 = jnp.where(
                    is_pad[:, None], zero_state, sel_states.astype(jnp.uint8)
                )
                sel_parent_local_z = jnp.where(
                    is_pad, zero_int32, sel_parent_local
                )
                sel_move_z = jnp.where(is_pad, zero_int8, sel_move)

                new_pack = jnp.zeros(
                    (K_per_peer, PACK_SIZE), dtype=jnp.uint8
                )
                new_pack = new_pack.at[:, 0:state_size].set(sel_states_u8)
                new_pack = new_pack.at[:, state_size + 0].set(
                    (sel_parent_local_z & 0xFF).astype(jnp.uint8)
                )
                new_pack = new_pack.at[:, state_size + 1].set(
                    ((sel_parent_local_z >> 8) & 0xFF).astype(jnp.uint8)
                )
                new_pack = new_pack.at[:, state_size + 2].set(
                    ((sel_parent_local_z >> 16) & 0xFF).astype(jnp.uint8)
                )
                new_pack = new_pack.at[:, state_size + 3].set(
                    ((sel_parent_local_z >> 24) & 0xFF).astype(jnp.uint8)
                )
                new_pack = new_pack.at[:, state_size + 4].set(
                    sel_move_z.astype(jnp.uint8)
                )

                merged_pack = jnp.where(
                    from_old[:, None], old_pack, new_pack
                )
                top_v_bf16 = top_v_new.astype(jnp.bfloat16)
                top_v_u16 = jax.lax.bitcast_convert_type(
                    top_v_bf16, jnp.uint16
                )
                merged_pack = merged_pack.at[:, state_size + 5].set(
                    (top_v_u16 & 0xFF).astype(jnp.uint8)
                )
                merged_pack = merged_pack.at[:, state_size + 6].set(
                    ((top_v_u16 >> 8) & 0xFF).astype(jnp.uint8)
                )

                new_top_scores = new_top_scores.at[dest].set(top_v_new)
                new_top_pack = new_top_pack.at[dest].set(merged_pack)

            return (new_top_scores, new_top_pack), None

        (_, final_top_pack), _ = jax.lax.scan(
            chunk_body,
            (init_top_scores, init_top_pack),
            jnp.arange(n_chunks, dtype=jnp.int32),
        )

        recv_buckets = jax.lax.all_to_all(
            final_top_pack,
            axis_name=axis_name,
            split_axis=0,
            concat_axis=0,
            tiled=True,
        )

        recv_flat = recv_buckets.reshape(-1, PACK_SIZE)
        recv_states_u8 = recv_flat[:, 0:state_size]
        recv_states = recv_states_u8.astype(STATE_DTYPE)
        recv_parent_local = (
            recv_flat[:, state_size + 0].astype(jnp.int32)
            | (recv_flat[:, state_size + 1].astype(jnp.int32) << 8)
            | (recv_flat[:, state_size + 2].astype(jnp.int32) << 16)
            | (recv_flat[:, state_size + 3].astype(jnp.int32) << 24)
        )
        recv_move = recv_flat[:, state_size + 4].astype(jnp.int8)
        recv_sender_rank = sender_rank_per_recv

        recv_state_sum = jnp.sum(recv_states.astype(jnp.int32), axis=1)
        is_padding = recv_state_sum == 0

        recv_h = jnp.sum(
            recv_states.astype(jnp.int64) * hash_vec, axis=1
        )
        sort_idx = jnp.argsort(recv_h)
        sorted_h = recv_h[sort_idx]
        sorted_states = recv_states[sort_idx]
        sorted_parent_local = recv_parent_local[sort_idx]
        sorted_move = recv_move[sort_idx]
        sorted_sender_rank = recv_sender_rank[sort_idx]
        sorted_is_padding = is_padding[sort_idx]
        is_dup_sorted = jnp.concatenate(
            [
                jnp.zeros(1, dtype=jnp.bool_),
                sorted_h[1:] == sorted_h[:-1],
            ]
        )

        recv_v_u16 = (
            recv_flat[:, state_size + 5].astype(jnp.uint16)
            | (
                recv_flat[:, state_size + 6].astype(jnp.uint16)
                << jnp.uint16(8)
            )
        )
        recv_v_bf16 = jax.lax.bitcast_convert_type(
            recv_v_u16, jnp.bfloat16
        )
        recv_v = recv_v_bf16.astype(jnp.float32)
        sorted_v = recv_v[sort_idx]
        drop_mask = is_dup_sorted | sorted_is_padding

        if history_depth > 0:
            in_hist = jnp.zeros(sorted_h.shape, dtype=jnp.bool_)
            if history_exact:
                for hi in range(history_depth):
                    in_hist = in_hist | _sorted_contains(
                        hist[hi], sorted_h
                    )
            else:
                idx = (sorted_h & jnp.int64(hist_mask)).astype(jnp.int32)
                for hi in range(history_depth):
                    in_hist = in_hist | hist[hi][idx]
            drop_mask = drop_mask | in_hist
        sorted_v_masked = jnp.where(drop_mask, BIG_F32, sorted_v)

        top_v_keep, keep_sorted_idx = _topk_smallest(
            sorted_v_masked, B_local
        )
        chosen_states = sorted_states[keep_sorted_idx]
        chosen_parent_local = sorted_parent_local[keep_sorted_idx]
        chosen_parent_rank = sorted_sender_rank[keep_sorted_idx]
        chosen_move = sorted_move[keep_sorted_idx]
        chosen_h = sorted_h[keep_sorted_idx]
        chosen_state_sum = jnp.sum(
            chosen_states.astype(jnp.int32), axis=1
        )
        chosen_is_real = chosen_state_sum != 0

        if history_depth > 0:
            if history_exact:
                row = jnp.sort(
                    jnp.where(chosen_is_real, chosen_h, jnp.int64(-1))
                )
            else:
                set_idx = (
                    chosen_h & jnp.int64(hist_mask)
                ).astype(jnp.int32)
                row = jnp.zeros((hist_size,), dtype=jnp.bool_).at[
                    set_idx
                ].set(True)
            new_hist = jnp.concatenate([hist[1:], row[None, :]], axis=0)
        else:
            new_hist = hist

        packed_backptr = (
            chosen_parent_local.astype(jnp.uint32)
            & jnp.uint32(BPTR_PL_MASK)
        ) | (
            chosen_parent_rank.astype(jnp.uint32)
            << jnp.uint32(BPTR_RANK_SHIFT)
        ) | (
            chosen_move.astype(jnp.uint32)
            << jnp.uint32(BPTR_MOVE_SHIFT)
        )

        new_min_v_log = min_v_log.at[j].set(top_v_keep[0])

        if trace_enabled:
            trace_hash = trace_prefix_hashes[j]
            eq_trace = (chosen_h == trace_hash) & chosen_is_real
            trace_any = jnp.any(eq_trace)
            trace_pos = jnp.argmax(eq_trace.astype(jnp.int32)).astype(
                jnp.int32
            )
            trace_hit = trace_any.astype(jnp.int32)
            trace_v = jnp.where(
                trace_any, top_v_keep[trace_pos], BIG_F32
            )
            trace_cutoff = top_v_keep[-1]
        else:
            trace_hit = jnp.int32(0)
            trace_pos = jnp.int32(-1)
            trace_v = BIG_F32
            trace_cutoff = BIG_F32

        eq_v0 = (chosen_h == V0_hash) & chosen_is_real
        if eg_hashes is not None:
            eg_h = jnp.zeros(chosen_states.shape[0], dtype=jnp.int64)
            cs = chosen_states.astype(jnp.int32)
            for pos in range(state_size):
                eg_h = eg_h ^ eg_ztab[pos][cs[:, pos]]
            eq_v0 = eq_v0 | (
                _sorted_contains(eg_hashes, eg_h) & chosen_is_real
            )
        any_hit = jnp.any(eq_v0)
        pos_hit = jnp.argmax(eq_v0.astype(jnp.int32)).astype(jnp.int32)
        is_first_hit = (found_step == -1) & any_hit
        new_found_step = jnp.where(is_first_hit, j, found_step)
        new_found_pos_local = jnp.where(
            is_first_hit, pos_hit, found_pos_local
        )
        new_found_pos_rank = jnp.where(
            is_first_hit, rank_int, found_pos_rank
        )
        candidate_state = chosen_states[pos_hit]
        new_verify_state = jnp.where(
            is_first_hit, candidate_state, verify_state
        )

        return (
            chosen_states[None, :, :],
            packed_backptr[None, :],
            new_min_v_log[None, :],
            new_found_step[None],
            new_found_pos_local[None],
            new_found_pos_rank[None],
            new_verify_state[None, :],
            new_hist[None],
            chosen_move[None, :],
            trace_hit[None],
            trace_pos[None],
            trace_v[None],
            trace_cutoff[None],
        )

    return beam_step_local


def beam_solve_v_only_spmd_packed_exact(
    init_state_list: list[int],
    v_params,
    all_moves: jax.Array,
    V0: jax.Array,
    hash_vec: jax.Array,
    mesh,
    B_local: int,
    K_per_peer: int,
    n_gen: int = 30,
    state_size: int = 150,
    num_steps: int = 300,
    dtype=jnp.bfloat16,
    internal_bs: int = 32768,
    tree_path: str | None = None,
    parent_chunk: int | None = 131072,
    pack_v_score: bool = True,
    progress_every: int = 0,
    owner_hash_vec: jax.Array | None = None,
    trace_prefix_hashes: np.ndarray | None = None,
    stop_on_trace_drop: bool = False,
    history_depth: int = 0,
    history_bits: int = 0,
    history_exact: bool = True,
    eg_hashes=None,
    eg_ztab=None,
    inv_move_tbl=None,
    q_mode: bool = True,
    qv_consistency: float = 0.0,
    pre_topk_mult: int = 0,
    *,
    exact_config: ArtgorExactConfig = ArtgorExactConfig(),
    exact_runtime: ArtgorExactBeamRuntime | None = None,
    _require_published_geometry: bool = True,
) -> dict[str, Any]:
    """Artgor's packed SPMD solver with staged bitwise-exact Q inference.

    Only the Q evaluation boundary changes.  The search body is the original
    streamed Q-mode body with its model call replaced by an input Q slice.
    """

    exact_config.validate()
    if dtype != jnp.bfloat16:
        raise ValueError("the published exact engine requires BF16")
    if not q_mode:
        raise ValueError("the exact staged solver supports Q mode only")
    if not pack_v_score:
        raise ValueError("Q mode requires pack_v_score=True")
    if float(qv_consistency) != 0.0:
        raise ValueError(
            "nonzero qv_consistency requires the original JAX QV path"
        )
    if pre_topk_mult:
        raise ValueError("pre_topk_mult is not part of the exact staged path")
    if isinstance(v_params, dict) and "members" in v_params:
        raise ValueError("checkpoint blends require the original JAX path")
    if not jax.config.jax_enable_x64:
        raise ValueError("JAX_ENABLE_X64 must be enabled before importing JAX")

    devices = np.asarray(mesh.devices).reshape(-1)
    world_size = int(devices.size)
    axis_names = tuple(mesh.axis_names)
    if axis_names != ("core",):
        raise ValueError("exact inference and search require mesh axis 'core'")
    parent_chunk = (
        exact_config.parent_chunk if parent_chunk is None else parent_chunk
    )
    if parent_chunk != exact_config.parent_chunk:
        raise ValueError("parent_chunk must match exact_config.parent_chunk")
    if internal_bs != exact_config.inference_chunk:
        raise ValueError("internal_bs must match exact inference_chunk")
    if K_per_peer * world_size < B_local:
        raise ValueError("receive buckets cannot supply B_local candidates")
    if B_local > 1 << BPTR_PL_BITS:
        raise ValueError("B_local exceeds the packed uint32 parent budget")
    if world_size > 1 << BPTR_RANK_BITS or n_gen > 1 << BPTR_MOVE_BITS:
        raise ValueError("rank or move count exceeds packed uint32 budget")
    if PACK_SIZE < state_size + 7:
        raise ValueError("PACK_SIZE is too small for state and BF16 score")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    depth_config = StagedDepthConfig(
        world_size=world_size,
        b_local=B_local,
        inference_chunk=exact_config.inference_chunk,
        parent_chunk=parent_chunk,
        n_gen=n_gen,
        state_size=state_size,
    )
    depth_config.validate(
        require_published_geometry=_require_published_geometry
    )

    init_state = np.asarray(init_state_list, dtype=STATE_DTYPE_NP)
    if init_state.shape != (state_size,):
        raise ValueError(f"initial state must have shape ({state_size},)")
    if trace_prefix_hashes is not None:
        trace_prefix_hashes = np.asarray(
            trace_prefix_hashes, dtype=np.int64
        )
        if (
            trace_prefix_hashes.ndim != 1
            or len(trace_prefix_hashes) < num_steps
        ):
            raise ValueError(
                "trace_prefix_hashes must cover every requested step"
            )
    if np.array_equal(init_state, np.asarray(V0)):
        return {
            "found": True,
            "path_len": 0,
            "path_idx": [],
            "found_step": -1,
            "wall_s": 0.0,
            "first_iter_s": None,
            "last_completed_step": -1,
            "lower_s": 0.0,
            "compile_s": 0.0,
            "min_v_trajectory_rank0": [],
            "timing_breakdown": {},
        }

    if exact_runtime is None:
        exact_runtime = prepare_artgor_exact_beam_runtime(
            v_params,
            mesh=mesh,
            exact_config=exact_config,
            state_storage_len=state_size,
        )
    replicated = NamedSharding(mesh, P())
    weights_d = jax.tree.map(
        lambda value: jax.device_put(value, replicated),
        exact_runtime.weights,
    )
    sharding = NamedSharding(mesh, P("core"))
    flat_state_sharding = NamedSharding(mesh, P("core", None))

    # The seed uses the same exact prefix/head executables and their production
    # 32K shape.  Only row zero is observed; the remaining rows are padding.
    seed_local = np.zeros(
        (exact_config.inference_chunk, state_size), dtype=STATE_DTYPE_NP
    )
    seed_local[0] = init_state
    seed_states = jax.make_array_from_callback(
        (world_size * exact_config.inference_chunk, state_size),
        flat_state_sharding,
        lambda _index: seed_local,
    )
    seed_start = time.perf_counter()
    seed_q = exact_runtime.inference(seed_states, weights_d)
    values0 = np.asarray(seed_q[0], dtype=np.float32)
    seed_inference_s = time.perf_counter() - seed_start

    all_moves_np = np.asarray(all_moves, dtype=np.int32)
    neighbors0 = init_state[all_moves_np]
    k0 = min(B_local, n_gen)
    _, top_idx0 = _topk_smallest(jnp.asarray(values0), k0)
    top_idx0_np = np.asarray(top_idx0, dtype=np.int32)
    chosen0_np = neighbors0[top_idx0_np].astype(STATE_DTYPE_NP)

    states0_np = np.zeros((B_local, state_size), dtype=STATE_DTYPE_NP)
    states0_np[:k0] = chosen0_np
    move0_full_np = np.zeros((B_local,), dtype=np.int8)
    move0_full_np[:k0] = top_idx0_np.astype(np.int8)

    v0_np = np.asarray(V0, dtype=STATE_DTYPE_NP)
    eq0 = np.all(chosen0_np == v0_np[None, :], axis=1)
    if bool(eq0.any()):
        pos0 = int(np.argmax(eq0))
        return {
            "found": True,
            "path_len": 1,
            "path_idx": [int(top_idx0_np[pos0])],
            "found_step": 0,
            "wall_s": seed_inference_s,
            "first_iter_s": None,
            "last_completed_step": 0,
            "lower_s": 0.0,
            "compile_s": 0.0,
            "min_v_trajectory_rank0": [],
            "timing_breakdown": {"seed_inference_s": seed_inference_s},
        }

    try:
        seed_q.delete()
        seed_states.delete()
    except Exception:
        pass
    del seed_q, seed_states, seed_local
    gc.collect()

    min_v_log = jnp.full(
        (world_size, num_steps), 1e6, dtype=jnp.float32
    )
    found_step = jnp.full((world_size,), -1, dtype=jnp.int32)
    found_pos_local = jnp.full((world_size,), -1, dtype=jnp.int32)
    found_pos_rank = jnp.full((world_size,), -1, dtype=jnp.int32)
    verify_state = jnp.zeros(
        (world_size, state_size), dtype=STATE_DTYPE
    )
    hist_rows = max(1, history_depth)
    hist_bits_n = history_bits or (
        int(np.ceil(np.log2(max(B_local, 2)))) + 5
    )
    hist_size = 1 << hist_bits_n
    if history_exact:
        hist = jnp.full(
            (world_size, hist_rows, B_local), -1, dtype=jnp.int64
        )
    else:
        hist = jnp.zeros(
            (world_size, hist_rows, hist_size), dtype=jnp.bool_
        )
    in_move = jnp.full((world_size, B_local), -1, dtype=jnp.int8)

    states_d = jax.make_array_from_callback(
        (world_size, B_local, state_size),
        sharding,
        lambda _index: states0_np[None],
    )
    mv_log_d = jax.device_put(min_v_log, sharding)
    fs_d = jax.device_put(found_step, sharding)
    fpl_d = jax.device_put(found_pos_local, sharding)
    fpr_d = jax.device_put(found_pos_rank, sharding)
    vstate_d = jax.device_put(verify_state, sharding)
    hist_d = jax.device_put(hist, sharding)
    inmv_d = jax.device_put(in_move, sharding)

    v0_hash_host = int(
        np.sum(np.asarray(V0).astype(np.int64) * np.asarray(hash_vec))
    )
    executables = build_staged_depth_executables(
        exact_runtime.inference,
        exact_runtime.weights,
        mesh=mesh,
        config=depth_config,
        all_moves=all_moves,
        V0=V0,
        hash_vec=hash_vec,
        V0_hash=jnp.int64(v0_hash_host),
        K_per_peer=K_per_peer,
        owner_hash_vec=owner_hash_vec,
        trace_prefix_hashes=trace_prefix_hashes,
        history_depth=history_depth,
        history_bits=hist_bits_n,
        history_exact=history_exact,
        eg_hashes=eg_hashes,
        eg_ztab=eg_ztab,
        inv_move_tbl=inv_move_tbl,
        prefix_local=exact_runtime.prefix_local,
        head_local=exact_runtime.head_local,
        donate_search_carry=True,
        require_published_geometry=_require_published_geometry,
    )

    if progress_every:
        print("[lower] staged exact depth...", flush=True)
    executables, lower_s, compile_s = _compile_staged_depth_executables(
        executables,
        states=states_d,
        weights=weights_d,
        hidden_size=exact_runtime.hidden_size,
        search_carry=(
            mv_log_d,
            fs_d,
            fpl_d,
            fpr_d,
            vstate_d,
            hist_d,
            inmv_d,
            jnp.int32(1),
        ),
        mesh=mesh,
    )
    if progress_every:
        print(
            f"[compile] lower={lower_s:.1f}s compile={compile_s:.1f}s",
            flush=True,
        )

    if tree_path is None:
        work_dir = "/kaggle/working"
        if not os.path.isdir(work_dir):
            work_dir = tempfile.gettempdir()
        fd, tree_path = tempfile.mkstemp(
            prefix="tree_exact_", suffix=".u32", dir=work_dir
        )
        os.close(fd)
    tree_mm = np.memmap(
        tree_path,
        mode="w+",
        dtype=np.uint32,
        shape=(num_steps, world_size, B_local),
    )
    seed_packed = move0_full_np.astype(np.uint32) << BPTR_MOVE_SHIFT
    tree_mm[0] = np.broadcast_to(
        seed_packed[None, :], (world_size, B_local)
    )

    t_start = time.perf_counter()
    first_iter_s = None
    last_completed_step = 0
    fs_per_rank = np.asarray(fs_d)
    trace_rows = [] if trace_prefix_hashes is not None else None
    depth_times = []
    copy_times = []
    write_times = []

    try:
        for step in range(1, num_steps):
            iter_start = time.perf_counter()
            (
                states_d,
                packed_d,
                mv_log_d,
                fs_d,
                fpl_d,
                fpr_d,
                vstate_d,
                hist_d,
                inmv_d,
                trace_hit_d,
                trace_pos_d,
                trace_v_d,
                trace_cutoff_d,
            ) = run_staged_depth(
                executables,
                states_d,
                weights_d,
                mv_log_d,
                fs_d,
                fpl_d,
                fpr_d,
                vstate_d,
                hist_d,
                inmv_d,
                jnp.int32(step),
            )
            jax.block_until_ready(fs_d)
            depth_s = time.perf_counter() - iter_start
            depth_times.append(depth_s)

            copy_start = time.perf_counter()
            packed_h = np.asarray(packed_d)
            copy_s = time.perf_counter() - copy_start
            copy_times.append(copy_s)

            write_start = time.perf_counter()
            tree_mm[step] = packed_h
            write_s = time.perf_counter() - write_start
            write_times.append(write_s)
            try:
                packed_d.delete()
            except Exception:
                pass
            del packed_d, packed_h

            fs_per_rank = np.asarray(fs_d)
            last_completed_step = step
            trace_msg = ""
            if trace_rows is not None:
                trace_hit_h = np.asarray(trace_hit_d)
                trace_pos_h = np.asarray(trace_pos_d)
                trace_v_h = np.asarray(trace_v_d)
                trace_cutoff_h = np.asarray(trace_cutoff_d)
                hit_ranks = np.where(trace_hit_h > 0)[0]
                row = {
                    "step": int(step),
                    "hit": bool(len(hit_ranks)),
                    "hit_ranks": [int(x) for x in hit_ranks.tolist()],
                    "best_cutoff_v": float(np.min(trace_cutoff_h)),
                }
                if len(hit_ranks):
                    rank0 = int(hit_ranks[0])
                    row.update(
                        {
                            "rank": rank0,
                            "pos": int(trace_pos_h[rank0]),
                            "target_v": float(trace_v_h[rank0]),
                            "rank_cutoff_v": float(
                                trace_cutoff_h[rank0]
                            ),
                        }
                    )
                    trace_msg = f" trace=hit r{rank0}"
                else:
                    trace_msg = " trace=drop"
                trace_rows.append(row)

            if first_iter_s is None:
                first_iter_s = time.perf_counter() - iter_start
            if progress_every and (
                step == 1
                or step % progress_every == 0
                or np.any(fs_per_rank >= 0)
            ):
                print(
                    f"[step {step:03d}/{num_steps - 1}] "
                    f"depth={depth_s:.2f}s copy={copy_s:.2f}s "
                    f"write={write_s:.2f}s fs={fs_per_rank.tolist()}"
                    f"{trace_msg}",
                    flush=True,
                )
            if (
                trace_rows is not None
                and stop_on_trace_drop
                and not trace_rows[-1]["hit"]
            ):
                break
            if np.any(fs_per_rank >= 0):
                break

        tree_mm.flush()
        fpl_per_rank = np.asarray(fpl_d)
        mv_per_rank = np.asarray(mv_log_d)
        wall_s = time.perf_counter() - t_start
        timing_breakdown = {
            "seed_inference_s": seed_inference_s,
            "depth_device_s": depth_times,
            "backpointer_copy_s": copy_times,
            "backpointer_write_s": write_times,
        }

        int_max = 2**30
        fs_signed = np.where(fs_per_rank >= 0, fs_per_rank, int_max)
        global_min_step = int(fs_signed.min())
        if global_min_step >= int_max:
            result = {
                "found": False,
                "path_len": 0,
                "path_idx": [],
                "found_step": -1,
                "wall_s": wall_s,
                "first_iter_s": first_iter_s,
                "last_completed_step": last_completed_step,
                "lower_s": lower_s,
                "compile_s": compile_s,
                "min_v_trajectory_rank0": mv_per_rank[0].tolist(),
                "timing_breakdown": timing_breakdown,
            }
            if trace_rows is not None:
                result["trace_prefix"] = trace_rows
            return result

        winner_rank = int(np.where(fs_signed == global_min_step)[0][0])
        found_at = int(fs_per_rank[winner_rank])
        found_pos = int(fpl_per_rank[winner_rank])
        path_idx = []
        current_rank = winner_rank
        current_pos = found_pos
        for back_step in range(found_at, -1, -1):
            record = int(tree_mm[back_step, current_rank, current_pos])
            parent_local = record & BPTR_PL_MASK
            parent_rank = (record >> BPTR_RANK_SHIFT) & BPTR_RANK_MASK
            move = (record >> BPTR_MOVE_SHIFT) & BPTR_MOVE_MASK
            path_idx.append(int(move))
            if back_step > 0:
                current_rank = int(parent_rank)
                current_pos = int(parent_local)
        path_idx.reverse()

        result = {
            "found": True,
            "path_len": len(path_idx),
            "path_idx": path_idx,
            "found_step": found_at,
            "found_pos_local": found_pos,
            "found_pos_rank": winner_rank,
            "wall_s": wall_s,
            "first_iter_s": first_iter_s,
            "last_completed_step": last_completed_step,
            "lower_s": lower_s,
            "compile_s": compile_s,
            "min_v_trajectory_rank0": mv_per_rank[0].tolist(),
            "timing_breakdown": timing_breakdown,
        }
        if trace_rows is not None:
            result["trace_prefix"] = trace_rows
        return result
    finally:
        try:
            del tree_mm
        except Exception:
            pass
        gc.collect()
        try:
            if tree_path and os.path.exists(tree_path):
                os.unlink(tree_path)
        except OSError:
            pass
