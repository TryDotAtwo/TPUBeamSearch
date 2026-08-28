from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.stream1_first_layer_ab import (
    find_generator_file,
    find_move_count_checkpoint,
    make_reachable_states,
)
from tpu_beam_search.config import BeamConfig, load_generators
from tpu_beam_search.stream1_inference import (
    Stream1Architecture,
    make_jitted_stream1_inference,
    stream1_weights_from_pytorch_state_dict,
)
from tpu_beam_search.stream1_pallas import pallas_fused_folded_hidden


SCREEN_BATCH = 4096
STARTUP_BATCHES = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
FINAL_BATCHES = (32768, 65536)
WARMUPS = 5
REPEATS = 15


def candidate_configs():
    """Aligned prefix candidates: (BM, BK_input, BN_input, buffers, lookahead)."""
    configs = {
        (bm, bk, bn, 2, False)
        for bm in (256, 512, 1024)
        for bk in (128, 256, 512)
        for bn in (256, 512, 768, 1536)
    }
    configs.update(
        {
            (256, 128, 512, 1, False),
            (256, 128, 512, 2, True),
        }
    )
    return tuple(sorted(configs))


def measure(call):
    started = time.perf_counter()
    output = call()
    output.block_until_ready()
    first = time.perf_counter() - started
    for _ in range(WARMUPS):
        call().block_until_ready()
    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        call().block_until_ready()
        samples.append(time.perf_counter() - started)
    return output, first, statistics.median(samples), samples


def prefix_call(states, weights, architecture, config):
    bm, bk, bn, buffers, lookahead = config
    return pallas_fused_folded_hidden(
        states,
        weights.input.weight,
        weights.input.bias,
        weights.hidden.weight,
        weights.hidden.bias,
        STATE_LEN=architecture.STATE_LEN,
        NUM_CLASSES=architecture.NUM_CLASSES,
        bm=bm,
        bk_input=bk,
        bn_input=bn,
        bk_hidden=256,
        bn_hidden=512,
        pipeline_buffer_count=buffers,
        pipeline_lookahead=lookahead,
    )


def key(config):
    bm, bk, bn, buffers, lookahead = config
    return f"bm{bm}_bk{bk}_bn{bn}_buf{buffers}_look{int(lookahead)}"


def main():
    generator_path = find_generator_file()
    beam = BeamConfig.from_generators(generator_path)
    generators = np.asarray(load_generators(generator_path))
    checkpoint_path, state_dict, inspected = find_move_count_checkpoint(beam.MOVE_COUNT)
    architecture = Stream1Architecture.from_pytorch_state_dict(
        state_dict,
        STATE_LEN=beam.STATE_LEN,
        STATE_STORAGE_LEN=beam.STATE_STORAGE_LEN,
        NUM_CLASSES=beam.NUM_CLASSES,
    )
    weights = stream1_weights_from_pytorch_state_dict(state_dict, architecture)
    del state_dict

    result = {
        "environment": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "contract": {
            "checkpoint": str(checkpoint_path),
            "inspected_checkpoints": inspected,
            "architecture": architecture.__dict__,
            "screen_batch": SCREEN_BATCH,
            "startup_batches": list(STARTUP_BATCHES),
            "final_batches": list(FINAL_BATCHES),
            "warmups": WARMUPS,
            "repeats": REPEATS,
        },
        "screen": {},
        "startup": {},
        "full_model": {},
        "decision": {},
    }

    screen_states = make_reachable_states(
        generators, SCREEN_BATCH, beam.STATE_LEN, beam.STATE_STORAGE_LEN
    )
    baseline_config = (256, 128, 512, 2, False)
    baseline_output = jax.jit(
        lambda states: prefix_call(states, weights, architecture, baseline_config)
    )(screen_states)
    baseline_output.block_until_ready()

    accepted = []
    for config in candidate_configs():
        name = key(config)
        compiled = jax.jit(
            lambda states, config=config: prefix_call(states, weights, architecture, config)
        )
        try:
            output, first, steady, samples = measure(lambda: compiled(screen_states))
            max_error = float(jnp.max(jnp.abs(
                output.astype(jnp.float32) - baseline_output.astype(jnp.float32)
            )))
            if not np.isfinite(max_error) or max_error > 0.25:
                raise AssertionError(f"BF16 error gate failed: {max_error}")
            entry = {
                "config": config,
                "compile_and_first_seconds": first,
                "steady_seconds_median": steady,
                "states_per_second": SCREEN_BATCH / steady,
                "max_baseline_error": max_error,
                "samples": samples,
            }
            result["screen"][name] = entry
            accepted.append((steady, config))
            print("SCREEN", name, json.dumps(entry), flush=True)
        except Exception as error:
            result["screen"][name] = {
                "config": config,
                "rejected": type(error).__name__,
                "message": str(error),
            }
            print("SCREEN_REJECTED", name, repr(error), flush=True)

    if not accepted:
        raise RuntimeError("all prefix configurations were rejected")
    accepted.sort()
    winner = accepted[0][1]

    # Host/kernel launch startup is amortized by batch; per-row-block pipeline bubbles
    # are exposed by comparing BM and buffer modes over the same batch frontier.
    diagnostic_configs = []
    for config in (baseline_config, winner, (256, 128, 512, 1, False), (256, 128, 512, 2, True)):
        if config not in diagnostic_configs and key(config) in result["screen"] and "rejected" not in result["screen"][key(config)]:
            diagnostic_configs.append(config)
    for config in diagnostic_configs:
        name = key(config)
        result["startup"][name] = {}
        for batch in STARTUP_BATCHES:
            states = make_reachable_states(
                generators, batch, beam.STATE_LEN, beam.STATE_STORAGE_LEN
            )
            compiled = jax.jit(
                lambda states, config=config: prefix_call(states, weights, architecture, config)
            )
            output, first, steady, samples = measure(lambda: compiled(states))
            entry = {
                "batch": batch,
                "compile_and_first_seconds": first,
                "steady_seconds_median": steady,
                "states_per_second": batch / steady,
                "nanoseconds_per_state": steady * 1e9 / batch,
                "samples": samples,
            }
            result["startup"][name][str(batch)] = entry
            print("STARTUP", name, batch, json.dumps(entry), flush=True)

    finalists = [config for _, config in accepted[:3]]
    if baseline_config not in finalists:
        finalists.append(baseline_config)
    for config in finalists:
        name = key(config)
        result["full_model"][name] = {}
        bm, bk, bn, buffers, lookahead = config
        inference = make_jitted_stream1_inference(
            architecture,
            backend="pallas",
            residual_fusion="separate",
            bm=bm,
            bk_input=bk,
            bn_input=bn,
            bk_hidden=256,
            bn_hidden=512,
            prefix_pipeline_buffer_count=buffers,
            prefix_pipeline_lookahead=lookahead,
            bk_residual=256,
            bn_residual=512,
            bk_output=512,
            bn_output=256,
        )
        for batch in FINAL_BATCHES:
            states = make_reachable_states(
                generators, batch, beam.STATE_LEN, beam.STATE_STORAGE_LEN
            )
            try:
                output, first, steady, samples = measure(lambda: inference(states, weights))
                if not bool(jnp.all(jnp.isfinite(output.astype(jnp.float32)))):
                    raise AssertionError("non-finite logits")
                entry = {
                    "batch": batch,
                    "compile_and_first_seconds": first,
                    "steady_seconds_median": steady,
                    "states_per_second": batch / steady,
                    "nanoseconds_per_state": steady * 1e9 / batch,
                    "samples": samples,
                }
                result["full_model"][name][str(batch)] = entry
                print("FULL", name, batch, json.dumps(entry), flush=True)
            except Exception as error:
                result["full_model"][name][str(batch)] = {
                    "rejected": type(error).__name__, "message": str(error)
                }
                print("FULL_REJECTED", name, batch, repr(error), flush=True)

    result["decision"] = {
        "prefix_winner": key(winner),
        "prefix_winner_config": winner,
        "prefix_finalists": [key(config) for config in finalists],
    }
    path = Path("/kaggle/working/stream1_prefix_optimization.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("DECISION", json.dumps(result["decision"]), flush=True)
    print("RESULT_PATH", path, flush=True)


if __name__ == "__main__":
    main()
