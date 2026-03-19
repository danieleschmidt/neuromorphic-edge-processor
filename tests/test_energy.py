"""Tests for EnergyProfiler."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from neuromorphic.energy import EnergyProfiler


def test_record_sops():
    ep = EnergyProfiler(energy_pj_per_sop=23.6)
    ep.record_sops(1000, t_us=0.0)
    assert ep.total_sops == 1000
    assert abs(ep.total_energy_pj - 23600.0) < 1e-6


def test_energy_units():
    ep = EnergyProfiler(energy_pj_per_sop=100.0)
    ep.record_sops(1000, t_us=0.0)
    assert ep.total_energy_pj == 100_000.0
    assert ep.total_energy_nj  == 100.0
    assert ep.total_energy_uj  == 0.1


def test_throughput():
    ep = EnergyProfiler(energy_pj_per_sop=23.6)
    ep.record_sops(1_000_000, t_us=0.0)
    tp = ep.throughput_sops(duration_us=1_000_000)  # 1 second
    assert abs(tp - 1_000_000) < 1


def test_power():
    ep = EnergyProfiler(energy_pj_per_sop=23.6)
    ep.record_sops(1000, t_us=0.0)
    # energy = 23600 pJ = 0.0236 nJ = 2.36e-5 µJ
    # duration = 1s = 1e6 µs → power = 2.36e-5 µW
    power = ep.power_uw(duration_us=1_000_000)
    assert power > 0


def test_compare_to_gpu():
    ep = EnergyProfiler(energy_pj_per_sop=23.6)
    ep.record_sops(10_000, t_us=0.0)
    cmp = ep.compare_to_gpu()
    assert cmp["speedup_vs_gpu"] > 1
    assert cmp["speedup_vs_cpu"] > cmp["speedup_vs_gpu"]


def test_compare_to_gpu_with_explicit_macs():
    ep = EnergyProfiler(energy_pj_per_sop=23.6)
    ep.record_sops(1, t_us=0.0)
    cmp = ep.compare_to_gpu(n_equivalent_macs=1000)
    # GPU: 1000 * 1000 pJ = 1e6 pJ; Neuromorphic: 23.6 pJ
    assert abs(cmp["gpu_pj"] - 1_000_000) < 1


def test_record_step():
    ep = EnergyProfiler(energy_pj_per_sop=10.0)
    step_energy = ep.record_step(500, t_us=1000.0)
    assert step_energy == 5000.0
    assert ep.total_sops == 500


def test_sop_rate_history():
    ep = EnergyProfiler()
    ep.record_sops(100, t_us=0.0)
    ep.record_sops(200, t_us=1000.0)
    hist = ep.sop_rate_history()
    assert hist.shape == (2, 2)
    assert hist[0, 1] == 100
    assert hist[1, 1] == 200


def test_reset():
    ep = EnergyProfiler(energy_pj_per_sop=23.6)
    ep.record_sops(5000, t_us=0.0)
    ep.reset()
    assert ep.total_sops == 0
    assert ep.total_energy_pj == 0.0
    assert ep.sop_rate_history().shape[0] == 0


def test_report_contains_key_info():
    ep = EnergyProfiler(energy_pj_per_sop=23.6)
    ep.record_sops(10_000, t_us=0.0)
    report = ep.report(duration_us=100_000)
    assert "23.6" in report
    assert "SOPs" in report or "sop" in report.lower()
    assert "GPU" in report


def test_zero_sops():
    ep = EnergyProfiler()
    assert ep.total_energy_pj == 0.0
    cmp = ep.compare_to_gpu()
    assert cmp["speedup_vs_gpu"] == float("inf")


def test_hardware_benchmarks():
    benchmarks = EnergyProfiler.HARDWARE_BENCHMARKS
    assert "Loihi" in benchmarks
    assert "TrueNorth" in benchmarks
    assert "GPU_MAC" in benchmarks
    assert benchmarks["GPU_MAC"] > benchmarks["Loihi"]
