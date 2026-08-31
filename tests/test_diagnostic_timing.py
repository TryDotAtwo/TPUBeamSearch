"""CPU measurement-contract checks; these do not establish TPU performance."""

from contextlib import contextmanager
import importlib
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def timing():
    return importlib.import_module("benchmarks.diagnostic_timing")


def compiled_witness(name, calls, traces):
    """Compile once, with a real execution callback and two output leaves."""
    with jax.default_device(jax.devices("cpu")[0]):
        value = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

        def work(x):
            traces.append(name)
            jax.debug.callback(lambda _: calls.append(name), x, ordered=True)
            return {"sum": x + 1, "nested": (x * 2,)}

        compiled = jax.jit(work).lower(value).compile()
    return compiled, value


def assert_samples(summary, count):
    samples = summary["samples_ms"]
    assert len(samples) == count
    assert all(math.isfinite(sample) and sample >= 0 for sample in samples)
    assert summary["min_ms"] == min(samples)
    assert summary["max_ms"] == max(samples)
    assert summary["median_ms"] == float(np.median(samples))


def test_paired_runs_forward_reverse_rounds_without_retracing(timing):
    calls, traces = [], []
    first, x = compiled_witness("A", calls, traces)
    second, y = compiled_witness("B", calls, traces)
    result = timing.paired_interleaved_measure(
        {"A": (first, (x,)), "B": (second, (y,))}, warmups=2, repeats=3,
    )

    assert calls == ["A", "B", "B", "A", "A", "B", "B", "A", "A", "B"]
    assert traces == ["A", "B"]
    assert result["execution_order"] == [["A", "B"], ["B", "A"], ["A", "B"]]
    assert result["warmups"] == 2
    assert result["repeats"] == 3
    assert_samples(result["cases"]["A"], 3)
    assert_samples(result["cases"]["B"], 3)


def test_queued_keeps_same_arguments_and_reports_per_call_not_scan_latency(timing):
    calls, traces = [], []
    compiled, value = compiled_witness("queued", calls, traces)
    result = timing.queued_measure(compiled, value, queue_depth=3, warmups=2, repeats=4)

    assert calls == ["queued"] * 14  # two warmups plus four batches of three
    assert traces == ["queued"]
    assert result["label"] == "queued_same_executable_not_real_scan"
    assert result["queue_depth"] == result["batch_call_count"] == 3
    assert result["timed_call_count"] == 12
    assert result["warmups"] == 2
    assert result["repeats"] == 4
    assert_samples(result, 4)
    assert len(result["batch_samples_ms"]) == 4
    np.testing.assert_allclose(result["samples_ms"], np.asarray(result["batch_samples_ms"]) / 3)


def test_queued_synchronizes_every_retained_output_leaf(timing, monkeypatch):
    calls, traces, batches = [], [], []
    compiled, value = compiled_witness("queued", calls, traces)
    real_block = jax.block_until_ready

    def observe_block(tree):
        # Forward real synchronization; only observe the completion boundary.
        if isinstance(tree, list):
            batches.append(tree)
        return real_block(tree)

    monkeypatch.setattr(jax, "block_until_ready", observe_block)
    timing.queued_measure(compiled, value, queue_depth=3, warmups=1, repeats=2)

    assert len(batches) == 2
    assert all(len(batch) == 3 for batch in batches)
    assert all(len(jax.tree.leaves(batch)) == 6 for batch in batches)
    for batch in batches:
        assert len({id(output) for output in batch}) == 3
        for output in batch:
            np.testing.assert_array_equal(output["sum"], [2.0, 3.0])
            np.testing.assert_array_equal(output["nested"][0], [2.0, 4.0])


@pytest.mark.parametrize("helper,option", [
    ("paired", "warmups"), ("paired", "repeats"),
    ("queued", "warmups"), ("queued", "repeats"), ("queued", "queue_depth"),
    ("profile", "iterations"),
])
@pytest.mark.parametrize("invalid", [0, -1, 1.5, True])
def test_invalid_counts_fail_before_executing(timing, tmp_path, helper, option, invalid):
    calls, traces = [], []
    compiled, value = compiled_witness("not-run", calls, traces)
    options = {option: invalid}
    with pytest.raises(ValueError, match=option):
        if helper == "paired":
            timing.paired_interleaved_measure({"A": (compiled, (value,))}, **options)
        elif helper == "queued":
            timing.queued_measure(compiled, value, **options)
        else:
            timing.diagnostic_profile(compiled, value, directory=tmp_path, **options)
    assert calls == []


@pytest.mark.parametrize("helper", ["paired", "queued", "profile"])
def test_lazy_jit_cannot_silently_compile_inside_measurement(timing, tmp_path, helper):
    traces = []

    def uncompiled(value):
        traces.append("traced")
        return value + 1

    lazy = jax.jit(uncompiled)
    with pytest.raises(TypeError, match="[Cc]ompiled"):
        if helper == "paired":
            timing.paired_interleaved_measure({"A": (lazy, (jnp.ones(2),))})
        elif helper == "queued":
            timing.queued_measure(lazy, jnp.ones(2))
        else:
            timing.diagnostic_profile(lazy, jnp.ones(2), directory=tmp_path)
    assert traces == []


def test_empty_paired_cases_are_rejected(timing):
    with pytest.raises(ValueError, match="cases"):
        timing.paired_interleaved_measure({})


def test_paired_requires_tuple_arguments(timing):
    compiled, value = compiled_witness("not-run", [], [])
    with pytest.raises(TypeError, match="tuple"):
        timing.paired_interleaved_measure({"A": (compiled, [value])})


def test_executable_argument_errors_are_not_dropped(timing):
    compiled, _ = compiled_witness("bad-shape", [], [])
    with pytest.raises(TypeError):
        timing.queued_measure(compiled, jnp.ones(3), warmups=1, repeats=1)


def test_profile_runs_existing_executable_inside_diagnostic_scope(timing, tmp_path, monkeypatch):
    calls, traces, contexts = [], [], []
    compiled, value = compiled_witness("profile", calls, traces)

    @contextmanager
    def trace(directory, *, create_perfetto_link):
        contexts.append((directory, create_perfetto_link, "entered"))
        assert calls == []
        yield
        assert calls == ["profile"] * 3
        contexts.append("exited")

    # The actual profiler writes platform-specific external artifacts. Keep the
    # CPU executable real; replace only the trace context's recording boundary.
    monkeypatch.setattr(jax.profiler, "trace", trace)
    result = timing.diagnostic_profile(compiled, value, directory=tmp_path, iterations=3)

    assert result == {"label": "diagnostic_only", "directory": str(tmp_path), "iterations": 3}
    assert contexts == [(str(tmp_path), False, "entered"), "exited"]
    assert traces == ["profile"]


def test_profile_failure_is_left_to_caller(timing, tmp_path, monkeypatch):
    compiled, value = compiled_witness("profile", [], [])

    @contextmanager
    def trace(*args, **kwargs):
        raise RuntimeError("profile backend unavailable")
        yield

    monkeypatch.setattr(jax.profiler, "trace", trace)
    with pytest.raises(RuntimeError, match="profile backend unavailable"):
        timing.diagnostic_profile(compiled, value, directory=tmp_path)
