"""
EnergyProfiler: Neuromorphic energy estimation.

The key metric in neuromorphic computing is the Synaptic Operation (SOP):
one input spike crossing one synapse. Energy scales linearly with SOP count.

Measured values from published hardware:
  - Intel Loihi:     ~23.6 pJ/SOP        (Davies et al., 2018)
  - IBM TrueNorth:   ~26  pJ/SOP         (Merolla et al., 2014)
  - BrainScaleS-2:   ~10  pJ/SOP (est.)
  - SpiNNaker:       ~800 pJ/SOP         (less optimised, CPU-based)
  - GPU (MAC):       ~1   nJ/MAC         (10–100× more than SOP)
  - CPU (MAC):       ~5   nJ/MAC

So a neuromorphic chip doing 1 TSOP/s at 23.6 pJ/SOP = 23.6 W theoretical peak,
but sparsity means actual activity is <<1% of that → mW-range inference.

References:
  Davies et al. (2018). Loihi: A neuromorphic manycore processor with on-chip learning.
  Merolla et al. (2014). A million spiking-neuron integrated circuit with a scalable
    communication network and interface. Science, 345(6197), 668-673.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional, Tuple


class EnergyProfiler:
    """
    Tracks SOP counts over time and estimates energy consumption.

    Supports per-timestep granularity and rolling window analysis.
    """

    # Published pJ/SOP values for reference
    HARDWARE_BENCHMARKS = {
        "Loihi":       23.6,   # pJ/SOP
        "TrueNorth":   26.0,
        "SpiNNaker":   800.0,
        "GPU_MAC":     1000.0,  # 1 nJ/MAC
        "CPU_MAC":     5000.0,  # 5 nJ/MAC
    }

    def __init__(
        self,
        energy_pj_per_sop: float = 23.6,
        window_size: int = 100,
    ):
        """
        Args:
            energy_pj_per_sop: Energy per synaptic operation (picojoules)
            window_size:       Steps to keep in rolling history
        """
        self.energy_pj_per_sop = energy_pj_per_sop
        self.window_size = window_size

        self._sop_history: List[Tuple[float, int]] = []   # (t_us, n_sops)
        self._total_sops = 0
        self._total_energy_pj = 0.0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_sops(self, n_sops: int, t_us: float) -> None:
        """Record a batch of synaptic operations at time t_us."""
        self._total_sops += n_sops
        energy = n_sops * self.energy_pj_per_sop
        self._total_energy_pj += energy
        self._sop_history.append((t_us, n_sops))
        # Trim history
        if len(self._sop_history) > self.window_size * 10:
            self._sop_history = self._sop_history[-self.window_size:]

    def record_step(self, n_sops: int, t_us: float) -> float:
        """Record SOPs for one time step. Returns energy for this step (pJ)."""
        self.record_sops(n_sops, t_us)
        return n_sops * self.energy_pj_per_sop

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def total_sops(self) -> int:
        return self._total_sops

    @property
    def total_energy_pj(self) -> float:
        return self._total_energy_pj

    @property
    def total_energy_nj(self) -> float:
        return self._total_energy_pj / 1000

    @property
    def total_energy_uj(self) -> float:
        return self._total_energy_pj / 1_000_000

    def energy_per_sop(self) -> float:
        """Effective energy per SOP (pJ)."""
        return self.energy_pj_per_sop

    def throughput_sops(self, duration_us: float) -> float:
        """SOP throughput (SOPs/second)."""
        duration_s = duration_us / 1_000_000
        return self._total_sops / duration_s if duration_s > 0 else 0.0

    def power_uw(self, duration_us: float) -> float:
        """Average power draw (microwatts) over simulation duration."""
        duration_s = duration_us / 1_000_000
        energy_uj = self.total_energy_uj
        return energy_uj / duration_s if duration_s > 0 else 0.0

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_to_gpu(self, n_equivalent_macs: Optional[int] = None) -> Dict:
        """
        Compare neuromorphic energy to equivalent GPU computation.

        If n_equivalent_macs is None, uses SOP count as proxy for MACs.
        GPU energy: 1 nJ/MAC = 1000 pJ/MAC.
        """
        n_macs = n_equivalent_macs if n_equivalent_macs is not None else self._total_sops
        gpu_energy_pj    = n_macs * self.HARDWARE_BENCHMARKS["GPU_MAC"]
        cpu_energy_pj    = n_macs * self.HARDWARE_BENCHMARKS["CPU_MAC"]
        neuro_energy_pj  = self._total_energy_pj

        return {
            "neuromorphic_pj":   neuro_energy_pj,
            "gpu_pj":            gpu_energy_pj,
            "cpu_pj":            cpu_energy_pj,
            "speedup_vs_gpu":    gpu_energy_pj   / neuro_energy_pj if neuro_energy_pj > 0 else float("inf"),
            "speedup_vs_cpu":    cpu_energy_pj   / neuro_energy_pj if neuro_energy_pj > 0 else float("inf"),
            "pj_per_sop":        self.energy_pj_per_sop,
            "gpu_pj_per_mac":    self.HARDWARE_BENCHMARKS["GPU_MAC"],
        }

    def sop_rate_history(self) -> np.ndarray:
        """Returns array of (t_us, sops) for all recorded steps."""
        if not self._sop_history:
            return np.empty((0, 2))
        return np.array(self._sop_history)

    def rolling_power_uw(self, window_us: float) -> List[Tuple[float, float]]:
        """
        Compute rolling average power (µW) over a sliding window.

        Returns list of (t_us, power_uw) tuples.
        """
        if not self._sop_history:
            return []

        result = []
        history = self._sop_history

        for i, (t, _) in enumerate(history):
            window_start = t - window_us
            window_sops = sum(
                s for ts, s in history
                if window_start <= ts <= t
            )
            energy_uj = window_sops * self.energy_pj_per_sop / 1_000_000
            duration_s = window_us / 1_000_000
            power = energy_uj / duration_s if duration_s > 0 else 0.0
            result.append((t, power))

        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, duration_us: Optional[float] = None) -> str:
        """Human-readable energy summary."""
        lines = [
            "=" * 50,
            "  Neuromorphic Energy Profile",
            "=" * 50,
            f"  Hardware model:     {self.energy_pj_per_sop} pJ/SOP",
            f"  Total SOPs:         {self._total_sops:,}",
            f"  Total energy:       {self._total_energy_pj:.2f} pJ",
            f"                      {self.total_energy_nj:.4f} nJ",
            f"                      {self.total_energy_uj:.6f} µJ",
        ]
        if duration_us:
            lines += [
                f"  Throughput:         {self.throughput_sops(duration_us):.2e} SOP/s",
                f"  Avg power:          {self.power_uw(duration_us):.4f} µW",
            ]

        cmp = self.compare_to_gpu()
        lines += [
            "",
            "  vs GPU (1 nJ/MAC):",
            f"    GPU energy:       {cmp['gpu_pj']:.2f} pJ",
            f"    Energy reduction: {cmp['speedup_vs_gpu']:.1f}×",
            "=" * 50,
        ]
        return "\n".join(lines)

    def reset(self) -> None:
        self._sop_history.clear()
        self._total_sops = 0
        self._total_energy_pj = 0.0

    def __repr__(self) -> str:
        return (
            f"EnergyProfiler("
            f"pJ/SOP={self.energy_pj_per_sop}, "
            f"total_SOPs={self._total_sops:,}, "
            f"total_energy={self._total_energy_pj:.2f}pJ)"
        )
