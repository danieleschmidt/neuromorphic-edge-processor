"""
NeuromorphicProcessor: Top-level multi-core event-driven processor.

Orchestrates a collection of NeuromorphicCores connected via a SynapticRouter.
Manages the global simulation clock, event injection, and step execution.

Simulation loop (per time step):
  1. Inject external events into designated input cores
  2. For each core: flush incoming event queue → receive_events → tick
  3. Route output spikes to connected downstream cores
  4. Advance time by dt_us

This mirrors the tick-based execution model of Intel Loihi's algorithmic
timestep and TrueNorth's synchronous-asynchronous hybrid scheduling.
"""

from __future__ import annotations
import numpy as np
import time
from typing import List, Dict, Optional, Callable, Tuple
from .core import NeuromorphicCore
from .router import SynapticRouter
from .event_stream import EventStream
from .energy import EnergyProfiler


class NeuromorphicProcessor:
    """
    Multi-core neuromorphic processor simulator.

    Example (3-core pipeline):
        proc = NeuromorphicProcessor(n_cores=3, neurons_per_core=128)
        proc.router.connect_pipeline([0, 1, 2])
        proc.core(0).init_weights(n_inputs=64)
        proc.core(1).init_weights(n_inputs=128)
        proc.core(2).init_weights(n_inputs=128)
        results = proc.run(input_stream, duration_us=100_000)
    """

    def __init__(
        self,
        n_cores: int,
        neurons_per_core: int = 256,
        dt_us: float = 1000.0,
        tau_m_ms: float = 20.0,
        v_threshold: float = 1.0,
        energy_pj_per_sop: float = 23.6,   # Intel Loihi measured value
        rng: Optional[np.random.Generator] = None,
    ):
        self.n_cores = n_cores
        self.neurons_per_core = neurons_per_core
        self.dt_us = dt_us
        self.t_us = 0.0
        self.energy_pj_per_sop = energy_pj_per_sop

        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng

        # Instantiate cores
        self._cores: List[NeuromorphicCore] = [
            NeuromorphicCore(
                core_id=i,
                n_neurons=neurons_per_core,
                dt_us=dt_us,
                tau_m_ms=tau_m_ms,
                v_threshold=v_threshold,
                rng=np.random.default_rng(rng.integers(2**31)),
            )
            for i in range(n_cores)
        ]

        # Router
        self.router = SynapticRouter(n_cores=n_cores, neurons_per_core=neurons_per_core)

        # Energy profiler
        self.profiler = EnergyProfiler(energy_pj_per_sop=energy_pj_per_sop)

        # Output event stream (collected from output cores)
        self.output_streams: Dict[int, EventStream] = {
            i: EventStream(capacity=100_000) for i in range(n_cores)
        }

        # Simulation wall-clock time tracking
        self._wall_time_start: Optional[float] = None

    # ------------------------------------------------------------------
    # Core access
    # ------------------------------------------------------------------

    def core(self, core_id: int) -> NeuromorphicCore:
        """Get a specific core by ID."""
        return self._cores[core_id]

    @property
    def cores(self) -> List[NeuromorphicCore]:
        return self._cores

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run(
        self,
        input_stream: EventStream,
        duration_us: float,
        input_core_ids: Optional[List[int]] = None,
        output_core_ids: Optional[List[int]] = None,
        on_step: Optional[Callable[[float, Dict], None]] = None,
    ) -> Dict:
        """
        Run simulation for duration_us microseconds.

        Args:
            input_stream:   External spike events to inject into input cores
            duration_us:    How long to simulate (microseconds)
            input_core_ids: Cores that receive external input (default: [0])
            output_core_ids: Cores whose output to collect (default: all)
            on_step:        Optional callback(t_us, step_stats) called each step

        Returns:
            Summary dict with spike counts, SOP counts, energy estimates.
        """
        if input_core_ids is None:
            input_core_ids = [0]
        if output_core_ids is None:
            output_core_ids = list(range(self.n_cores))

        n_steps = int(duration_us / self.dt_us)
        self._wall_time_start = time.perf_counter()

        step_spike_counts = []

        for step in range(n_steps):
            t_start = self.t_us
            t_end   = self.t_us + self.dt_us

            # -- 1. Inject external events into input cores --
            for cid in input_core_ids:
                core = self._cores[cid]
                # Get external events for this time window
                window = input_stream.window(int(t_start), int(t_end))
                if len(window) > 0:
                    d = window.data
                    # Translate global IDs to local (modulo neurons_per_core)
                    local_ids = (d["neuron_id"] % core.n_neurons).astype(np.uint32)
                    sops = core.receive_events_batch(local_ids, d["polarity"])
                    self.profiler.record_sops(sops, t_start)

            # -- 2. Process routed events + tick each core --
            step_total_spikes = 0
            for core in self._cores:
                # Deliver routed events from router queue
                incoming = self.router.flush_queue(core.core_id)
                if len(incoming) > 0:
                    d = incoming.data
                    # Translate global→local IDs
                    local_ids = (d["neuron_id"] % core.n_neurons).astype(np.uint32)
                    sops = core.receive_events_batch(local_ids, d["polarity"])
                    self.profiler.record_sops(sops, t_start)

                # Advance one time step
                fired_ids, fired_pols = core.tick()

                # Collect output
                if core.core_id in output_core_ids and len(fired_ids) > 0:
                    ts = np.full(len(fired_ids), int(self.t_us), dtype=np.uint64)
                    self.output_streams[core.core_id].append_batch(
                        ts, fired_ids, fired_pols
                    )

                # Route output spikes to connected downstream cores
                if len(fired_ids) > 0:
                    self.router.route(core.core_id, fired_ids, fired_pols, self.t_us)

                step_total_spikes += len(fired_ids)

            step_spike_counts.append(step_total_spikes)

            if on_step is not None:
                on_step(self.t_us, {"spikes": step_total_spikes})

            self.t_us += self.dt_us

        wall_time = time.perf_counter() - self._wall_time_start
        return self._build_results(duration_us, wall_time, step_spike_counts)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _build_results(
        self,
        duration_us: float,
        wall_time_s: float,
        step_spike_counts: List[int],
    ) -> Dict:
        total_sops   = sum(c.total_synaptic_ops for c in self._cores)
        total_spikes = sum(c.total_spikes_out   for c in self._cores)
        energy_nj    = self.profiler.total_energy_pj / 1000

        core_stats = [c.stats() for c in self._cores]

        # Speedup vs real-time
        simulated_s = duration_us / 1_000_000
        speedup     = simulated_s / wall_time_s if wall_time_s > 0 else float("inf")

        return {
            "duration_us":       duration_us,
            "n_steps":           int(duration_us / self.dt_us),
            "total_spikes":      total_spikes,
            "total_sops":        total_sops,
            "energy_nj":         energy_nj,
            "energy_pj_per_sop": self.energy_pj_per_sop,
            "wall_time_s":       wall_time_s,
            "speedup_vs_realtime": speedup,
            "mean_spikes_per_step": float(np.mean(step_spike_counts)) if step_spike_counts else 0,
            "core_stats":        core_stats,
            "router_stats":      self.router.stats(),
        }

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all core states and output streams."""
        for core in self._cores:
            core.reset()
        for k in self.output_streams:
            self.output_streams[k] = EventStream(capacity=100_000)
        self.t_us = 0.0
        self.profiler.reset()

    def total_sops(self) -> int:
        return sum(c.total_synaptic_ops for c in self._cores)

    def total_energy_pj(self) -> float:
        return self.profiler.total_energy_pj

    def __repr__(self) -> str:
        return (
            f"NeuromorphicProcessor("
            f"cores={self.n_cores}, "
            f"neurons_per_core={self.neurons_per_core}, "
            f"dt={self.dt_us}µs)"
        )
