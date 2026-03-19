"""
SynapticRouter: Inter-core event routing.

Models the mesh network between neuromorphic cores, analogous to:
  - Intel Loihi: hierarchical mesh with local/global routing
  - IBM TrueNorth: 2D torus-connected core mesh
  - BrainScaleS: on-wafer network

The router maintains a connectivity table (which output cores receive
events from each input core) and dispatches output spike events to the
appropriate downstream cores. This decouples the routing topology from
the neuron model.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from .event_stream import EventStream


class SynapticRouter:
    """
    Routes spike events between NeuromorphicCores.

    Connectivity is specified as a directed adjacency:
        connections[src_core_id] = [dst_core_id, ...]

    Neuron IDs in output spikes are core-local. The router translates them
    to global IDs when needed: global_id = core_id * neurons_per_core + local_id

    Hardware analog: Loihi's spike routing table / TrueNorth's crossbar mesh.
    """

    def __init__(self, n_cores: int, neurons_per_core: int = 256):
        self.n_cores = n_cores
        self.neurons_per_core = neurons_per_core

        # Adjacency list: src_core → list of dst_core ids
        self._connections: Dict[int, List[int]] = {i: [] for i in range(n_cores)}

        # Pending event queues per core: core_id → EventStream
        self._queues: Dict[int, EventStream] = {
            i: EventStream(capacity=10_000) for i in range(n_cores)
        }

        # Statistics
        self.total_routed = 0
        self.routing_table_hits = 0

    # ------------------------------------------------------------------
    # Topology configuration
    # ------------------------------------------------------------------

    def connect(self, src_core: int, dst_core: int) -> None:
        """Add a directed connection from src_core to dst_core."""
        self._validate_core(src_core)
        self._validate_core(dst_core)
        if dst_core not in self._connections[src_core]:
            self._connections[src_core].append(dst_core)

    def connect_pipeline(self, core_ids: List[int]) -> None:
        """Connect cores in a linear pipeline: 0→1→2→..."""
        for i in range(len(core_ids) - 1):
            self.connect(core_ids[i], core_ids[i + 1])

    def connect_all_to_all(self, src_cores: List[int], dst_cores: List[int]) -> None:
        """Connect every source core to every destination core."""
        for src in src_cores:
            for dst in dst_cores:
                if src != dst:
                    self.connect(src, dst)

    def connect_grid(self, rows: int, cols: int, wrap: bool = False) -> None:
        """
        Connect cores in a 2D grid (like TrueNorth's torus mesh).
        Each core connects to its 4 neighbours (N/S/E/W).
        If wrap=True, edges wrap around (torus topology).
        """
        def idx(r, c):
            return r * cols + c

        for r in range(rows):
            for c in range(cols):
                src = idx(r, c)
                neighbours = []
                if r > 0:       neighbours.append(idx(r-1, c))  # N
                if r < rows-1:  neighbours.append(idx(r+1, c))  # S
                if c > 0:       neighbours.append(idx(r, c-1))  # W
                if c < cols-1:  neighbours.append(idx(r, c+1))  # E
                if wrap:
                    neighbours.append(idx((r-1) % rows, c))
                    neighbours.append(idx((r+1) % rows, c))
                    neighbours.append(idx(r, (c-1) % cols))
                    neighbours.append(idx(r, (c+1) % cols))
                for dst in set(neighbours):
                    self.connect(src, dst)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        src_core: int,
        fired_ids: np.ndarray,
        fired_pols: np.ndarray,
        t_us: float,
    ) -> int:
        """
        Route output spikes from src_core to all connected destination cores.

        Translates local neuron IDs to global IDs and enqueues events.

        Returns:
            Number of events dispatched (= len(fired_ids) * n_destinations)
        """
        if len(fired_ids) == 0:
            return 0

        dst_cores = self._connections.get(src_core, [])
        if not dst_cores:
            return 0

        # Global neuron IDs
        global_ids = (src_core * self.neurons_per_core + fired_ids).astype(np.uint32)
        timestamps  = np.full(len(fired_ids), int(t_us), dtype=np.uint64)

        n_dispatched = 0
        for dst in dst_cores:
            self._queues[dst].append_batch(timestamps, global_ids, fired_pols)
            n_dispatched += len(fired_ids)
            self.routing_table_hits += 1

        self.total_routed += n_dispatched
        return n_dispatched

    def flush_queue(self, core_id: int) -> EventStream:
        """
        Return and clear the pending event queue for core_id.

        Called by NeuromorphicProcessor before each core tick so the
        core can process its incoming events.
        """
        queue = self._queues[core_id]
        self._queues[core_id] = EventStream(capacity=10_000)
        return queue

    def peek_queue_size(self, core_id: int) -> int:
        """Return number of pending events for core without clearing."""
        return len(self._queues[core_id])

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def topology(self) -> Dict[int, List[int]]:
        """Return copy of the full connection map."""
        return {k: list(v) for k, v in self._connections.items()}

    def stats(self) -> Dict:
        return {
            "total_routed": self.total_routed,
            "routing_table_hits": self.routing_table_hits,
            "pending_events": {i: len(q) for i, q in self._queues.items()},
        }

    def _validate_core(self, core_id: int) -> None:
        if not (0 <= core_id < self.n_cores):
            raise ValueError(f"core_id {core_id} out of range [0, {self.n_cores})")

    def __repr__(self) -> str:
        edges = sum(len(v) for v in self._connections.values())
        return (
            f"SynapticRouter(cores={self.n_cores}, "
            f"edges={edges}, "
            f"routed={self.total_routed})"
        )
