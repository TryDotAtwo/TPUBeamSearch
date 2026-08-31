"""Diagnostic timing of already compiled JAX executables, never compilation.

These measurements neither establish numerical acceptance nor change a Q gate.
Queued repeated inputs are not a compiled scan or a real chunked beam workload.
All failures propagate so the caller can record them against the affected case.
"""

from __future__ import annotations

import math
from numbers import Integral
from pathlib import Path
from statistics import median
from time import perf_counter

import jax


def _positive_count(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _compiled_only(call):
    # A lazy jax.jit object can compile on invocation after a shape change.
    # Requiring the public AOT type makes the same-executable scope explicit.
    if not isinstance(call, jax.stages.Compiled):
        raise TypeError("call must be an already compiled jax.stages.Compiled executable")


def _elapsed_ms(start):
    elapsed = (perf_counter() - start) * 1000.0
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed time must be finite and nonnegative")
    return elapsed


def _summary(samples):
    return {
        "samples_ms": samples,
        "median_ms": float(median(samples)),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def paired_interleaved_measure(cases, *, warmups=5, repeats=12):
    """Measure synchronized calls in alternating forward/reverse rounds.

    ``cases`` maps names to ``(compiled_executable, positional_args_tuple)``.
    The insertion order defines the first round. Two cases therefore run ABBA
    across two rounds. ``execution_order`` contains timed rounds only; warmup
    rounds use the same alternating policy and reset before timed rounds.
    Input synchronization and warmups are outside the measured intervals.
    """
    _positive_count("warmups", warmups)
    _positive_count("repeats", repeats)
    if not cases:
        raise ValueError("cases must not be empty")
    for call, args in cases.values():
        _compiled_only(call)
        if not isinstance(args, tuple):
            raise TypeError("case arguments must be a tuple")
    for _, args in cases.values():
        jax.block_until_ready(args)

    names = list(cases)
    for repeat in range(warmups):
        for name in names if repeat % 2 == 0 else reversed(names):
            call, args = cases[name]
            jax.block_until_ready(call(*args))

    samples = {name: [] for name in names}
    execution_order = []
    for repeat in range(repeats):
        order = names[:] if repeat % 2 == 0 else list(reversed(names))
        execution_order.append(order)
        for name in order:
            call, args = cases[name]
            start = perf_counter()
            jax.block_until_ready(call(*args))
            samples[name].append(_elapsed_ms(start))
    return {
        "label": "paired_interleaved_synchronous_same_executable",
        "warmups": int(warmups),
        "repeats": int(repeats),
        "execution_order": execution_order,
        "cases": {name: _summary(values) for name, values in samples.items()},
    }


def queued_measure(call, *args, queue_depth=8, warmups=5, repeats=5):
    """Queue identical calls; report batch wall time and amortized per-call cost.

    Every output pytree is retained and synchronized before the timer stops.
    ``samples_ms`` and its median/min/max are batch durations divided by
    ``batch_call_count``: queued throughput cost, not single-call latency.
    There is no JIT, scan, new executable, or changed argument dependency graph.
    Warmups are individual synchronized calls, not queued batches.
    """
    _positive_count("queue_depth", queue_depth)
    _positive_count("warmups", warmups)
    _positive_count("repeats", repeats)
    _compiled_only(call)
    jax.block_until_ready(args)
    for _ in range(warmups):
        jax.block_until_ready(call(*args))

    batch_samples = []
    for _ in range(repeats):
        start = perf_counter()
        outputs = [call(*args) for _ in range(queue_depth)]
        jax.block_until_ready(outputs)
        batch_samples.append(_elapsed_ms(start))
        # Do not keep the previous batch live during the next allocation wave.
        del outputs
    return {
        "label": "queued_same_executable_not_real_scan",
        "warmups": int(warmups),
        "repeats": int(repeats),
        "queue_depth": int(queue_depth),
        "batch_call_count": int(queue_depth),
        "timed_call_count": int(queue_depth * repeats),
        "batch_samples_ms": batch_samples,
        **_summary([elapsed / queue_depth for elapsed in batch_samples]),
    }


def diagnostic_profile(call, *args, directory, iterations=3):
    """Trace an existing executable, independent of optimization acceptance.

    The caller chooses whether/where to profile and records per-case failures.
    No numerical gate, global JAX configuration, or executable is modified.
    """
    _positive_count("iterations", iterations)
    _compiled_only(call)
    trace_directory = str(Path(directory))
    jax.block_until_ready(args)
    with jax.profiler.trace(trace_directory, create_perfetto_link=False):
        for _ in range(iterations):
            jax.block_until_ready(call(*args))
    return {
        "label": "diagnostic_only",
        "directory": trace_directory,
        "iterations": int(iterations),
    }
