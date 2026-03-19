"""
NeuromorphicCore: Leaky Integrate-and-Fire (LIF) neuron population.

Models a single processing core as found in Intel Loihi (128 neurons/core)
or IBM TrueNorth (256 neurons/core). Each core integrates incoming spike
events into membrane potentials and generates output spikes when threshold
is exceeded.

LIF dynamics (discrete time):
    V[t] = V[t-1] * decay + I_syn[t]
    if V[t] >= V_th:
        emit spike, V[t] = V_reset

This is purely event-driven — only neurons receiving input are updated.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict
from .event_stream import EventStream


class LIFNeuron:
    """Single Leaky Integrate-and-Fire neuron (for inspection/debugging)."""

    def __init__(
        self,
        tau_m: float = 20.0,       # membrane time constant (ms)
        v_threshold: float = 1.0,  # spike threshold
        v_reset: float = 0.0,      # reset potential
        v_rest: float = 0.0,       # resting potential
        dt_ms: float = 1.0,        # time step (ms)
        refractory_ms: float = 2.0,
    ):
        self.tau_m = tau_m
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.dt_ms = dt_ms
        self.refractory_ms = refractory_ms

        self.decay = np.exp(-dt_ms / tau_m)
        self.v = v_rest
        self.refractory_remaining = 0.0
        self.spike_times: List[float] = []

    def step(self, current: float, t_ms: float) -> bool:
        """Advance one time step. Returns True if neuron spiked."""
        if self.refractory_remaining > 0:
            self.refractory_remaining -= self.dt_ms
            self.v = self.v_reset
            return False

        self.v = self.decay * self.v + (1 - self.decay) * current
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.refractory_remaining = self.refractory_ms
            self.spike_times.append(t_ms)
            return True
        return False


class NeuromorphicCore:
    """
    Vectorised LIF neuron population — one processing core.

    In Loihi architecture, each core handles up to 1024 compartments.
    We keep 256 as a default (matching IBM TrueNorth's per-core count)
    but allow arbitrary sizes.

    The core operates in event-driven mode:
      - receive_events() injects synaptic current from incoming spikes
      - tick() advances simulation by one time step, returns output spikes
    """

    DEFAULT_NEURONS = 256  # neurons per core (TrueNorth default)

    def __init__(
        self,
        core_id: int,
        n_neurons: int = DEFAULT_NEURONS,
        tau_m_ms: float = 20.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        dt_us: float = 1000.0,          # simulation time step (µs)
        refractory_us: float = 2000.0,  # absolute refractory period
        w_scale: float = 0.1,           # default synaptic weight scale
        rng: Optional[np.random.Generator] = None,
    ):
        self.core_id = core_id
        self.n_neurons = n_neurons
        self.dt_us = dt_us
        self.dt_ms = dt_us / 1000.0
        self.refractory_us = refractory_us
        self.w_scale = w_scale

        # Membrane decay per time step
        self.decay = np.exp(-self.dt_ms / tau_m_ms)
        self.v_threshold = v_threshold
        self.v_reset = v_reset

        # State vectors
        self.v = np.zeros(n_neurons)                       # membrane potentials
        self.refractory = np.zeros(n_neurons, dtype=float) # remaining refrac (µs)
        self.i_syn = np.zeros(n_neurons)                   # accumulated synaptic current

        # Synaptic weight matrix: shape (n_input_neurons, n_neurons)
        # Initialized lazily on first receive_events()
        self._weights: Optional[np.ndarray] = None
        self._n_inputs: Optional[int] = None

        # Counters
        self.total_spikes_out = 0
        self.total_synaptic_ops = 0  # SOPs — main energy metric
        self.t_us = 0.0              # current simulation time

        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng

    # ------------------------------------------------------------------
    # Synaptic connectivity
    # ------------------------------------------------------------------

    def init_weights(self, n_inputs: int, connectivity: float = 0.1) -> None:
        """
        Initialise random sparse synaptic weight matrix.

        Args:
            n_inputs:     Number of input neurons (from upstream core/stream)
            connectivity: Fraction of input→output connections to create
        """
        self._n_inputs = n_inputs
        W = np.zeros((n_inputs, self.n_neurons))
        mask = self._rng.random((n_inputs, self.n_neurons)) < connectivity
        W[mask] = self._rng.normal(0.5, 0.1, mask.sum()) * self.w_scale
        self._weights = W

    def set_weights(self, weights: np.ndarray) -> None:
        """Set explicit weight matrix, shape (n_inputs, n_neurons)."""
        self._n_inputs = weights.shape[0]
        self._weights = weights.copy()

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def receive_events(self, event_stream: EventStream, t_start_us: int, t_end_us: int) -> None:
        """
        Inject synaptic current from incoming spike events in [t_start, t_end).

        Each spike from neuron `i` with polarity `p` adds:
            I_syn[j] += W[i, j] * p   for all j connected to i

        This is a Synaptic Operation (SOP) — the fundamental compute unit
        of neuromorphic hardware. Energy scales with SOP count.
        """
        if self._weights is None:
            raise RuntimeError("Weights not initialised. Call init_weights() first.")

        d = event_stream.data
        mask = (d["timestamp_us"] >= t_start_us) & (d["timestamp_us"] < t_end_us)
        events = d[mask]

        n_sops = 0
        for ev in events:
            nid = int(ev["neuron_id"])
            pol = float(ev["polarity"])
            if nid < self._weights.shape[0]:
                w_row = self._weights[nid]
                active = w_row != 0
                self.i_syn[active] += w_row[active] * pol
                n_sops += int(active.sum())

        self.total_synaptic_ops += n_sops

    def receive_events_batch(self, neuron_ids: np.ndarray, polarities: np.ndarray) -> int:
        """
        Fast vectorised event ingestion (when time windowing is not needed).

        Returns number of SOPs performed.
        """
        if self._weights is None:
            raise RuntimeError("Weights not initialised.")

        n_sops = 0
        valid_mask = neuron_ids < self._weights.shape[0]
        valid_ids = neuron_ids[valid_mask].astype(int)
        valid_pols = polarities[valid_mask].astype(float)

        for nid, pol in zip(valid_ids, valid_pols):
            w_row = self._weights[nid]
            active = w_row != 0
            if active.any():
                self.i_syn[active] += w_row[active] * pol
                n_sops += int(active.sum())

        self.total_synaptic_ops += n_sops
        return n_sops

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def tick(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance simulation by one dt_us step.

        Returns:
            fired_ids:  Array of neuron indices that spiked this step
            fired_pols: Polarities of output spikes (all +1 for LIF)
        """
        # Handle refractory period
        in_refrac = self.refractory > 0
        self.refractory[in_refrac] -= self.dt_us
        self.refractory[self.refractory < 0] = 0

        # LIF dynamics (only for non-refractory neurons)
        active = ~in_refrac
        self.v[active] = self.decay * self.v[active] + self.i_syn[active]

        # Spike detection
        fired = (self.v >= self.v_threshold) & active

        # Reset spiked neurons
        self.v[fired] = self.v_reset
        self.refractory[fired] = self.refractory_us

        # Clear synaptic current buffer
        self.i_syn[:] = 0.0

        self.t_us += self.dt_us
        n_fired = fired.sum()
        self.total_spikes_out += n_fired

        fired_ids = np.where(fired)[0].astype(np.uint32)
        fired_pols = np.ones(len(fired_ids), dtype=np.int8)
        return fired_ids, fired_pols

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def membrane_potentials(self) -> np.ndarray:
        return self.v.copy()

    def reset(self) -> None:
        """Reset all neuron states and counters (not weights)."""
        self.v[:] = 0.0
        self.refractory[:] = 0.0
        self.i_syn[:] = 0.0
        self.t_us = 0.0
        self.total_spikes_out = 0
        self.total_synaptic_ops = 0

    def stats(self) -> Dict:
        return {
            "core_id": self.core_id,
            "n_neurons": self.n_neurons,
            "total_spikes_out": self.total_spikes_out,
            "total_sops": self.total_synaptic_ops,
            "mean_v": float(self.v.mean()),
        }

    def __repr__(self) -> str:
        return (
            f"NeuromorphicCore(id={self.core_id}, "
            f"neurons={self.n_neurons}, "
            f"spikes_out={self.total_spikes_out}, "
            f"SOPs={self.total_synaptic_ops})"
        )
