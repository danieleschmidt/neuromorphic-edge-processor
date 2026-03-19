"""Tests for SynapticRouter."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from neuromorphic.router import SynapticRouter


def test_connect_and_topology():
    r = SynapticRouter(n_cores=4)
    r.connect(0, 1)
    r.connect(0, 2)
    topo = r.topology()
    assert 1 in topo[0]
    assert 2 in topo[0]
    assert len(topo[1]) == 0


def test_connect_pipeline():
    r = SynapticRouter(n_cores=4)
    r.connect_pipeline([0, 1, 2, 3])
    topo = r.topology()
    assert 1 in topo[0]
    assert 2 in topo[1]
    assert 3 in topo[2]
    assert len(topo[3]) == 0


def test_route_enqueues():
    r = SynapticRouter(n_cores=3, neurons_per_core=16)
    r.connect(0, 1)
    r.connect(0, 2)
    fired = np.array([0, 5], dtype=np.uint32)
    pols  = np.ones(2, dtype=np.int8)
    n_dispatched = r.route(0, fired, pols, t_us=0.0)
    assert n_dispatched == 4  # 2 spikes × 2 destinations

    assert r.peek_queue_size(1) == 2
    assert r.peek_queue_size(2) == 2


def test_route_no_destinations():
    r = SynapticRouter(n_cores=2, neurons_per_core=16)
    fired = np.array([0, 1], dtype=np.uint32)
    pols  = np.ones(2, dtype=np.int8)
    n = r.route(0, fired, pols, t_us=0.0)
    assert n == 0


def test_route_empty_fires():
    r = SynapticRouter(n_cores=2)
    r.connect(0, 1)
    n = r.route(0, np.array([], dtype=np.uint32), np.array([], dtype=np.int8), 0.0)
    assert n == 0


def test_flush_queue():
    r = SynapticRouter(n_cores=2, neurons_per_core=16)
    r.connect(0, 1)
    fired = np.array([3], dtype=np.uint32)
    pols  = np.ones(1, dtype=np.int8)
    r.route(0, fired, pols, t_us=1000.0)

    queue = r.flush_queue(1)
    assert len(queue) == 1
    # After flush, queue should be empty
    assert r.peek_queue_size(1) == 0


def test_global_id_in_queue():
    """Verify global neuron IDs are correctly computed."""
    r = SynapticRouter(n_cores=3, neurons_per_core=256)
    r.connect(1, 2)
    # Core 1, neuron 10 → global id = 1*256 + 10 = 266
    fired = np.array([10], dtype=np.uint32)
    pols  = np.ones(1, dtype=np.int8)
    r.route(1, fired, pols, t_us=0.0)
    q = r.flush_queue(2)
    assert q.data[0]["neuron_id"] == 266


def test_connect_grid():
    r = SynapticRouter(n_cores=4)
    r.connect_grid(rows=2, cols=2)
    topo = r.topology()
    # In a 2×2 grid, each core has 2 neighbours
    for core_id in range(4):
        assert len(topo[core_id]) == 2


def test_connect_all_to_all():
    r = SynapticRouter(n_cores=4)
    r.connect_all_to_all([0, 1], [2, 3])
    topo = r.topology()
    assert 2 in topo[0] and 3 in topo[0]
    assert 2 in topo[1] and 3 in topo[1]


def test_invalid_core():
    r = SynapticRouter(n_cores=3)
    with pytest.raises(ValueError):
        r.connect(0, 5)


def test_stats():
    r = SynapticRouter(n_cores=2)
    r.connect(0, 1)
    fired = np.arange(5, dtype=np.uint32)
    pols  = np.ones(5, dtype=np.int8)
    r.route(0, fired, pols, t_us=0.0)
    s = r.stats()
    assert s["total_routed"] == 5
