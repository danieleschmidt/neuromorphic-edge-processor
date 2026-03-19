"""Tests for EventStream."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from neuromorphic.event_stream import EventStream, SpikeEvent


def test_append_and_len():
    s = EventStream(capacity=10)
    assert len(s) == 0
    s.append(100, 5, 1)
    assert len(s) == 1


def test_append_batch():
    s = EventStream(capacity=100)
    ts  = np.array([0, 1000, 2000], dtype=np.uint64)
    ids = np.array([0, 1, 2], dtype=np.uint32)
    pol = np.array([1, -1, 1], dtype=np.int8)
    s.append_batch(ts, ids, pol)
    assert len(s) == 3


def test_data_view():
    s = EventStream()
    s.append(500, 3, 1)
    s.append(600, 7, -1)
    d = s.data
    assert d[0]["timestamp_us"] == 500
    assert d[1]["neuron_id"] == 7
    assert d[1]["polarity"] == -1


def test_window():
    s = EventStream()
    for t in range(0, 5000, 1000):
        s.append(t, 0, 1)
    w = s.window(1000, 3000)
    assert len(w) == 2  # t=1000 and t=2000


def test_window_empty():
    s = EventStream()
    s.append(500, 0, 1)
    w = s.window(1000, 2000)
    assert len(w) == 0


def test_neuron_events():
    s = EventStream()
    s.append(100, 2, 1)
    s.append(200, 3, 1)
    s.append(300, 2, 1)
    evs = s.neuron_events(2)
    assert list(evs) == [100, 300]


def test_firing_rate():
    s = EventStream()
    # 10 spikes for neuron 0 over 1 second = 10 Hz
    for i in range(10):
        s.append(i * 100_000, 0, 1)
    rate = s.firing_rate(0, 1_000_000)
    assert abs(rate - 10.0) < 1e-6


def test_sparsity_high():
    s = EventStream()
    # Only 1 neuron fires once in a 10-neuron, 10-bin setup → very sparse
    s.append(0, 0, 1)
    sp = s.sparsity(n_neurons=10, duration_us=10_000, bin_us=1000.0)
    assert sp > 0.9


def test_grow_capacity():
    s = EventStream(capacity=4)
    for i in range(20):
        s.append(i * 100, i, 1)
    assert len(s) == 20


def test_rate_coding():
    rng = np.random.default_rng(42)
    rates = np.array([100.0, 0.0, 50.0])  # Hz
    s = EventStream.from_rate_coding(rates, duration_us=100_000, dt_us=1000.0, rng=rng)
    d = s.data
    # Neuron 1 (rate=0) should have 0 spikes
    assert np.sum(d["neuron_id"] == 1) == 0
    # Neuron 0 (rate=100Hz) should have more spikes than neuron 2 (50Hz)
    assert np.sum(d["neuron_id"] == 0) > np.sum(d["neuron_id"] == 2)


def test_temporal_contrast_encoding():
    rng = np.random.default_rng(42)
    # Simple increasing signal → should trigger ON events
    signal = np.ones((10, 4))
    # Make one neuron rapidly increase
    signal[3:, 0] = 5.0
    s = EventStream.from_temporal_contrast(signal, threshold=0.3, dt_us=1000.0)
    d = s.data
    on_events  = d[d["polarity"] ==  1]
    # Should have at least some ON events for neuron 0
    assert len(on_events) > 0


def test_iterate():
    s = EventStream()
    s.append(10, 1, 1)
    s.append(20, 2, -1)
    evs = list(s)
    assert len(evs) == 2
    assert isinstance(evs[0], SpikeEvent)
    assert evs[0].timestamp_us == 10
    assert evs[1].polarity == -1


def test_spike_event_repr():
    ev = SpikeEvent(1000, 5, 1)
    assert "1000" in repr(ev)
    assert "+" in repr(ev)
