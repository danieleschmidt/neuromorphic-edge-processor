"""Tests for NeuromorphicProcessor."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from neuromorphic.processor import NeuromorphicProcessor
from neuromorphic.event_stream import EventStream


def make_simple_processor(n_cores=2, neurons=32, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    proc = NeuromorphicProcessor(
        n_cores=n_cores,
        neurons_per_core=neurons,
        dt_us=1000.0,
        v_threshold=0.5,
        rng=rng,
    )
    proc.router.connect_pipeline(list(range(n_cores)))
    for i in range(n_cores):
        proc.core(i).init_weights(n_inputs=neurons, connectivity=0.3)
    return proc


def make_input_stream(n_neurons=32, n_events=50, duration_us=50_000):
    rng = np.random.default_rng(0)
    stream = EventStream(capacity=n_events)
    ts  = np.sort(rng.integers(0, int(duration_us), n_events)).astype(np.uint64)
    ids = rng.integers(0, n_neurons, n_events).astype(np.uint32)
    pol = rng.choice([-1, 1], n_events).astype(np.int8)
    stream.append_batch(ts, ids, pol)
    return stream


def test_processor_init():
    proc = make_simple_processor(n_cores=3, neurons=64)
    assert proc.n_cores == 3
    assert len(proc.cores) == 3
    assert proc.core(0).n_neurons == 64


def test_run_returns_dict():
    proc = make_simple_processor()
    stream = make_input_stream()
    results = proc.run(stream, duration_us=10_000)
    assert "total_spikes" in results
    assert "total_sops" in results
    assert "energy_nj" in results
    assert "n_steps" in results
    assert results["n_steps"] == 10


def test_run_duration():
    proc = make_simple_processor()
    stream = make_input_stream()
    results = proc.run(stream, duration_us=50_000)
    assert results["n_steps"] == 50
    assert abs(proc.t_us - 50_000) < 1  # should advance time


def test_energy_positive():
    proc = make_simple_processor()
    stream = make_input_stream(n_events=100)
    results = proc.run(stream, duration_us=20_000)
    assert results["energy_nj"] >= 0


def test_total_sops_consistent():
    proc = make_simple_processor()
    stream = make_input_stream(n_events=100)
    results = proc.run(stream, duration_us=20_000)
    # total_sops from results should match sum of core stats
    core_sops = sum(cs["total_sops"] for cs in results["core_stats"])
    assert results["total_sops"] == core_sops


def test_reset_clears_state():
    proc = make_simple_processor()
    stream = make_input_stream(n_events=100)
    proc.run(stream, duration_us=20_000)
    proc.reset()
    assert proc.t_us == 0.0
    assert proc.total_sops() == 0
    assert proc.core(0).total_spikes_out == 0


def test_output_streams_populated():
    proc = make_simple_processor(n_cores=2, neurons=32)
    # Set low threshold so spikes actually propagate
    for c in proc.cores:
        c.v_threshold = 0.01
    stream = make_input_stream(n_events=200)
    proc.run(stream, duration_us=20_000, output_core_ids=[1])
    # Output streams exist for all cores
    assert 1 in proc.output_streams


def test_on_step_callback():
    proc = make_simple_processor()
    stream = make_input_stream()
    steps_seen = []
    proc.run(stream, duration_us=5_000, on_step=lambda t, s: steps_seen.append(t))
    assert len(steps_seen) == 5


def test_core_stats_in_results():
    proc = make_simple_processor(n_cores=3)
    stream = make_input_stream()
    results = proc.run(stream, duration_us=10_000)
    assert len(results["core_stats"]) == 3
    for cs in results["core_stats"]:
        assert "core_id" in cs
        assert "total_sops" in cs


def test_single_core():
    proc = NeuromorphicProcessor(n_cores=1, neurons_per_core=32, dt_us=1000.0)
    proc.core(0).init_weights(n_inputs=32, connectivity=0.5)
    stream = make_input_stream(n_neurons=32, n_events=50)
    results = proc.run(stream, duration_us=10_000)
    assert results["n_steps"] == 10
