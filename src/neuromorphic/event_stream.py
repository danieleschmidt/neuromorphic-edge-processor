"""
EventStream: Sparse spike-train representation.

In neuromorphic hardware, information is carried by discrete events (spikes).
This module models the Address Event Representation (AER) protocol used in
silicon retinas (DAVIS, DVS cameras) and chips like Intel Loihi.

Each event: (timestamp_us, neuron_id, polarity)
  - timestamp_us: microsecond timestamp (uint64)
  - neuron_id:    source neuron address (uint32)
  - polarity:     +1 (ON spike) or -1 (OFF spike), 0 for inhibitory
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional


@dataclass(frozen=True)
class SpikeEvent:
    """Single Address Event Record (AER)."""
    timestamp_us: int   # microseconds
    neuron_id: int      # source address
    polarity: int       # +1 ON, -1 OFF

    def __repr__(self) -> str:
        pol = "+" if self.polarity > 0 else "-"
        return f"Spike(t={self.timestamp_us}µs, n={self.neuron_id}, {pol})"


class EventStream:
    """
    Sparse event-stream representation of spike trains.

    Internally uses a structured NumPy array for memory efficiency.
    Provides windowed access, rate computation, and encoding utilities.

    Hardware analog: AER bus / event FIFO on Intel Loihi / DVS sensor output.
    """

    # Structured dtype: 8+4+1 = 13 bytes per event (vs 24 bytes for 3 int64)
    DTYPE = np.dtype([
        ("timestamp_us", np.uint64),
        ("neuron_id",    np.uint32),
        ("polarity",     np.int8),
    ])

    def __init__(self, capacity: int = 1_000_000):
        self._data = np.empty(capacity, dtype=self.DTYPE)
        self._count = 0
        self._capacity = capacity

    # ------------------------------------------------------------------
    # Building the stream
    # ------------------------------------------------------------------

    def append(self, timestamp_us: int, neuron_id: int, polarity: int) -> None:
        """Add a single spike event."""
        if self._count >= self._capacity:
            self._grow()
        self._data[self._count] = (timestamp_us, neuron_id, polarity)
        self._count += 1

    def append_batch(
        self,
        timestamps: np.ndarray,
        neuron_ids: np.ndarray,
        polarities: np.ndarray,
    ) -> None:
        """Vectorised batch append (much faster than looping)."""
        n = len(timestamps)
        while self._count + n > self._capacity:
            self._grow()
        sl = slice(self._count, self._count + n)
        self._data[sl]["timestamp_us"] = timestamps
        self._data[sl]["neuron_id"]    = neuron_ids
        self._data[sl]["polarity"]     = polarities
        self._count += n

    def _grow(self) -> None:
        new_cap = self._capacity * 2
        new_data = np.empty(new_cap, dtype=self.DTYPE)
        new_data[: self._capacity] = self._data
        self._data = new_data
        self._capacity = new_cap

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        """Structured array view of all events (read-only copy)."""
        return self._data[: self._count].copy()

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[SpikeEvent]:
        for row in self._data[: self._count]:
            yield SpikeEvent(int(row["timestamp_us"]), int(row["neuron_id"]), int(row["polarity"]))

    def window(self, t_start_us: int, t_end_us: int) -> "EventStream":
        """Return a new EventStream containing only events in [t_start, t_end)."""
        d = self._data[: self._count]
        mask = (d["timestamp_us"] >= t_start_us) & (d["timestamp_us"] < t_end_us)
        sub = d[mask]
        stream = EventStream(capacity=max(len(sub), 1))
        stream.append_batch(sub["timestamp_us"], sub["neuron_id"], sub["polarity"])
        return stream

    def neuron_events(self, neuron_id: int) -> np.ndarray:
        """Timestamps of all spikes from a specific neuron."""
        d = self._data[: self._count]
        return d["timestamp_us"][d["neuron_id"] == neuron_id]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def firing_rate(self, neuron_id: int, duration_us: float) -> float:
        """Mean firing rate (Hz) for a neuron over the stream duration."""
        n_spikes = np.sum(self._data[: self._count]["neuron_id"] == neuron_id)
        duration_s = duration_us / 1_000_000
        return n_spikes / duration_s if duration_s > 0 else 0.0

    def population_rate(self, duration_us: float, n_neurons: int) -> float:
        """Mean population firing rate (Hz)."""
        duration_s = duration_us / 1_000_000
        return (self._count / n_neurons) / duration_s if duration_s > 0 else 0.0

    def sparsity(self, n_neurons: int, duration_us: float, bin_us: float = 1000.0) -> float:
        """
        Fraction of (neuron, time-bin) slots that are silent.
        High sparsity (>95%) is the hallmark of neuromorphic efficiency.
        """
        n_bins = max(1, int(duration_us / bin_us))
        active = set()
        d = self._data[: self._count]
        bins = (d["timestamp_us"] / bin_us).astype(np.int64)
        for nid, b in zip(d["neuron_id"], bins):
            active.add((int(nid), int(b)))
        total_slots = n_neurons * n_bins
        return 1.0 - len(active) / total_slots

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_rate_coding(
        cls,
        rates: np.ndarray,
        duration_us: float,
        dt_us: float = 100.0,
        rng: Optional[np.random.Generator] = None,
    ) -> "EventStream":
        """
        Poisson rate-coded encoding: each neuron fires with probability
        proportional to its rate at each time step.

        Args:
            rates:       Firing rates (Hz) per neuron, shape (N,)
            duration_us: Total duration in microseconds
            dt_us:       Time step in microseconds
            rng:         Optional numpy random generator for reproducibility
        """
        if rng is None:
            rng = np.random.default_rng()

        n_neurons = len(rates)
        n_steps = int(duration_us / dt_us)
        dt_s = dt_us / 1_000_000

        stream = cls(capacity=n_neurons * n_steps)
        probs = rates * dt_s  # probability of spike per step
        probs = np.clip(probs, 0, 1)

        for step in range(n_steps):
            t = int(step * dt_us)
            fired = np.where(rng.random(n_neurons) < probs)[0]
            if len(fired):
                ts = np.full(len(fired), t, dtype=np.uint64)
                pols = np.ones(len(fired), dtype=np.int8)
                stream.append_batch(ts, fired.astype(np.uint32), pols)

        return stream

    @classmethod
    def from_temporal_contrast(
        cls,
        signal: np.ndarray,
        threshold: float = 0.1,
        dt_us: float = 1000.0,
    ) -> "EventStream":
        """
        Temporal contrast encoding — mimics a Dynamic Vision Sensor (DVS).

        Generates ON event (+1) when signal increases by `threshold`,
        OFF event (-1) when it decreases by `threshold`.

        Args:
            signal:    2D array shape (T, N) — signal per neuron over time
            threshold: Relative change threshold for spike generation
            dt_us:     Time step size in microseconds
        """
        T, N = signal.shape
        stream = cls(capacity=T * N)
        log_sig = np.log(np.clip(signal, 1e-9, None))

        for t in range(1, T):
            diff = log_sig[t] - log_sig[t - 1]
            on_neurons  = np.where(diff  >= threshold)[0]
            off_neurons = np.where(diff <= -threshold)[0]

            ts = int(t * dt_us)
            if len(on_neurons):
                stream.append_batch(
                    np.full(len(on_neurons), ts, dtype=np.uint64),
                    on_neurons.astype(np.uint32),
                    np.ones(len(on_neurons), dtype=np.int8),
                )
            if len(off_neurons):
                stream.append_batch(
                    np.full(len(off_neurons), ts, dtype=np.uint64),
                    off_neurons.astype(np.uint32),
                    np.full(len(off_neurons), -1, dtype=np.int8),
                )

        return stream

    def __repr__(self) -> str:
        return (
            f"EventStream(events={self._count:,}, "
            f"capacity={self._capacity:,})"
        )
