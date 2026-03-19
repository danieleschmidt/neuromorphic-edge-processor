"""Tests for NeuromorphicCore and LIFNeuron."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from neuromorphic.core import NeuromorphicCore, LIFNeuron
from neuromorphic.event_stream import EventStream


def test_lif_single_spike():
    n = LIFNeuron(tau_m=20.0, v_threshold=1.0, dt_ms=1.0)
    # Drive with strong current
    spiked = False
    for t in range(100):
        if n.step(2.0, t):
            spiked = True
            break
    assert spiked, "LIF neuron should spike under strong drive"


def test_lif_no_spike_below_threshold():
    n = LIFNeuron(tau_m=20.0, v_threshold=1.0, dt_ms=1.0)
    for t in range(200):
        spiked = n.step(0.001, t)
        assert not spiked, "LIF should not spike with minimal current"


def test_lif_refractory():
    n = LIFNeuron(tau_m=20.0, v_threshold=1.0, refractory_ms=5.0, dt_ms=1.0)
    # Force a spike
    n.v = 1.5
    n.step(0.0, 0)
    # During refractory, should not spike even with strong drive
    for t in range(1, 6):
        spiked = n.step(100.0, t)
        assert not spiked, f"Should not spike during refractory (step {t})"


def test_core_tick_no_input():
    core = NeuromorphicCore(core_id=0, n_neurons=16, dt_us=1000.0)
    core.init_weights(n_inputs=8, connectivity=0.5)
    fired, pols = core.tick()
    assert len(fired) == 0, "No spikes without input"


def test_core_receive_events_and_spike():
    rng = np.random.default_rng(0)
    core = NeuromorphicCore(
        core_id=0, n_neurons=16, dt_us=1000.0,
        v_threshold=0.05,   # very low threshold
        rng=rng,
    )
    core.init_weights(n_inputs=16, connectivity=1.0)  # full connectivity
    # Override weights to be large
    W = np.ones((16, 16)) * 0.1
    core.set_weights(W)

    # Inject strong events
    ids  = np.arange(16, dtype=np.uint32)
    pols = np.ones(16, dtype=np.int8)
    sops = core.receive_events_batch(ids, pols)
    assert sops > 0, "Should have SOPs from batch"

    fired, _ = core.tick()
    assert len(fired) > 0, "Should produce output spikes with strong input"


def test_core_sop_count():
    core = NeuromorphicCore(core_id=0, n_neurons=10, dt_us=1000.0)
    # Set full connectivity weight matrix (10 input → 10 output neurons)
    W = np.ones((10, 10)) * 0.1
    core.set_weights(W)

    ids  = np.array([0, 1, 2], dtype=np.uint32)
    pols = np.ones(3, dtype=np.int8)
    sops = core.receive_events_batch(ids, pols)
    # Each spike fans out to 10 neurons → 3 × 10 = 30 SOPs
    assert sops == 30
    assert core.total_synaptic_ops == 30


def test_core_receive_events_time_window():
    core = NeuromorphicCore(core_id=0, n_neurons=8)
    W = np.ones((8, 8)) * 0.1
    core.set_weights(W)

    stream = EventStream()
    stream.append(500,  0, 1)
    stream.append(1500, 1, 1)   # outside window
    stream.append(2500, 2, 1)   # outside window

    core.receive_events(stream, t_start_us=0, t_end_us=1000)
    # Only 1 event should have been processed
    assert core.total_synaptic_ops == 8  # 1 event × 8 connections


def test_core_reset():
    core = NeuromorphicCore(core_id=0, n_neurons=8)
    W = np.ones((8, 8)) * 0.1
    core.set_weights(W)
    ids  = np.array([0, 1], dtype=np.uint32)
    pols = np.ones(2, dtype=np.int8)
    core.receive_events_batch(ids, pols)
    core.tick()
    core.reset()
    assert core.total_spikes_out == 0
    assert np.all(core.v == 0)


def test_core_stats():
    core = NeuromorphicCore(core_id=7, n_neurons=32)
    stats = core.stats()
    assert stats["core_id"] == 7
    assert stats["n_neurons"] == 32
    assert "total_sops" in stats


def test_core_no_weights_error():
    core = NeuromorphicCore(core_id=0, n_neurons=8)
    stream = EventStream()
    stream.append(0, 0, 1)
    with pytest.raises(RuntimeError):
        core.receive_events(stream, 0, 1000)
