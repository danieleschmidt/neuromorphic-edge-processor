"""Event-driven processing for neuromorphic computing.

This module implements efficient event-driven processing algorithms
optimized for sparse neural activity and low-power edge devices.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class SpikeEvent:
    """Represents a single spike event in the network."""
    neuron_id: int
    timestamp: float
    layer_id: int
    amplitude: float = 1.0
    metadata: Optional[Dict] = None


class EventQueue:
    """Efficient priority queue for spike events."""
    
    def __init__(self, max_size: int = 10000):
        self.events = deque(maxlen=max_size)
        self.current_time = 0.0
        
    def add_event(self, event: SpikeEvent):
        """Add event to queue in temporal order."""
        # Insert in sorted order (simple implementation)
        inserted = False
        for i, existing_event in enumerate(self.events):
            if event.timestamp < existing_event.timestamp:
                self.events.insert(i, event)
                inserted = True
                break
        
        if not inserted:
            self.events.append(event)
    
    def get_events_until(self, timestamp: float) -> List[SpikeEvent]:
        """Get all events up to specified timestamp."""
        events = []
        while self.events and self.events[0].timestamp <= timestamp:
            events.append(self.events.popleft())
        return events
    
    def peek_next_time(self) -> Optional[float]:
        """Get timestamp of next event without removing it."""
        return self.events[0].timestamp if self.events else None
    
    def clear(self):
        """Clear all events."""
        self.events.clear()
        self.current_time = 0.0


class EventDrivenProcessor:
    """Event-driven processor for sparse neuromorphic computation.
    
    Processes neural activity only when spikes occur, dramatically reducing
    computation for sparse networks typical in neuromorphic systems.
    """
    
    def __init__(
        self,
        network_topology: Dict[int, List[int]],
        neuron_params: Optional[Dict] = None,
        dt: float = 0.1,
        max_queue_size: int = 50000
    ):
        """Initialize event-driven processor.
        
        Args:
            network_topology: Dict mapping neuron_id -> list of target neuron_ids
            neuron_params: Parameters for individual neurons
            dt: Time step resolution (ms)
            max_queue_size: Maximum events in queue
        """
        self.topology = network_topology
        self.dt = dt
        self.max_queue_size = max_queue_size
        
        # Default neuron parameters
        self.neuron_params = neuron_params or {\n            \"tau_mem\": 20.0,\n            \"tau_syn\": 5.0,\n            \"v_thresh\": -50.0,\n            \"v_reset\": -70.0,\n            \"v_rest\": -65.0,\n            \"refractory_period\": 2.0\n        }\n        \n        # Initialize network state\n        self.num_neurons = max(max(self.topology.keys()), \n                              max(max(targets) if targets else 0 \n                                  for targets in self.topology.values())) + 1\n        \n        # Neuron state variables\n        self.v_mem = torch.full((self.num_neurons,), self.neuron_params[\"v_rest\"])\n        self.i_syn = torch.zeros(self.num_neurons)\n        self.refractory_time = torch.zeros(self.num_neurons)\n        self.last_spike_time = torch.full((self.num_neurons,), -float('inf'))\n        \n        # Synaptic weights (can be learned)\n        self.weights = self._initialize_weights()\n        \n        # Event processing\n        self.event_queue = EventQueue(max_queue_size)\n        self.current_time = 0.0\n        \n        # Statistics\n        self.total_events_processed = 0\n        self.computation_savings = 0.0\n        \n    def _initialize_weights(self) -> Dict[Tuple[int, int], float]:\n        \"\"\"Initialize synaptic weights between connected neurons.\"\"\"\n        weights = {}\n        for source_id, targets in self.topology.items():\n            for target_id in targets:\n                # Random initial weights\n                weights[(source_id, target_id)] = torch.randn(1).item() * 0.1\n        return weights\n    \n    def add_external_input(self, neuron_id: int, current: float, timestamp: float):\n        \"\"\"Add external input current to specific neuron.\"\"\"\n        event = SpikeEvent(\n            neuron_id=neuron_id,\n            timestamp=timestamp,\n            layer_id=0,  # Input layer\n            amplitude=current,\n            metadata={\"type\": \"external_input\"}\n        )\n        self.event_queue.add_event(event)\n    \n    def process_until_time(self, end_time: float) -> List[SpikeEvent]:\n        \"\"\"Process all events up to specified time.\n        \n        Args:\n            end_time: Final simulation time (ms)\n            \n        Returns:\n            List of generated spike events\n        \"\"\"\n        output_spikes = []\n        \n        while self.current_time < end_time:\n            # Get next event time or advance by dt\n            next_event_time = self.event_queue.peek_next_time()\n            \n            if next_event_time is not None and next_event_time <= end_time:\n                # Process events at this time\n                events = self.event_queue.get_events_until(next_event_time)\n                self.current_time = next_event_time\n                \n                # Update only neurons that received events\n                active_neurons = set()\n                for event in events:\n                    active_neurons.add(event.neuron_id)\n                    self._process_input_event(event)\n                \n                # Check for spikes in active neurons\n                for neuron_id in active_neurons:\n                    if self._check_spike_condition(neuron_id):\n                        spike_event = self._generate_spike(neuron_id)\n                        output_spikes.append(spike_event)\n                        \n                        # Propagate to connected neurons\n                        self._propagate_spike(spike_event)\n                \n                self.total_events_processed += len(events)\n                \n            else:\n                # No events, advance time\n                self.current_time = min(self.current_time + self.dt, end_time)\n                \n                # Passive decay for all neurons (could be optimized further)\n                self._passive_decay()\n        \n        return output_spikes\n    \n    def _process_input_event(self, event: SpikeEvent):\n        \"\"\"Process input event for specific neuron.\"\"\"\n        neuron_id = event.neuron_id\n        \n        if event.metadata and event.metadata.get(\"type\") == \"external_input\":\n            # External current input\n            self.i_syn[neuron_id] += event.amplitude\n        else:\n            # Synaptic input from other neurons\n            self.i_syn[neuron_id] += event.amplitude\n    \n    def _check_spike_condition(self, neuron_id: int) -> bool:\n        \"\"\"Check if neuron should generate a spike.\"\"\"\n        # Check refractory period\n        if self.refractory_time[neuron_id] > self.current_time:\n            return False\n        \n        # Update membrane potential\n        dt_since_last = min(self.dt, self.current_time - self.last_spike_time[neuron_id])\n        if dt_since_last > 0:\n            # Euler integration\n            tau_mem = self.neuron_params[\"tau_mem\"]\n            dv = (self.neuron_params[\"v_rest\"] - self.v_mem[neuron_id] + self.i_syn[neuron_id]) / tau_mem\n            self.v_mem[neuron_id] += dv * dt_since_last\n        \n        # Check threshold\n        return self.v_mem[neuron_id] >= self.neuron_params[\"v_thresh\"]\n    \n    def _generate_spike(self, neuron_id: int) -> SpikeEvent:\n        \"\"\"Generate spike event and reset neuron.\"\"\"\n        # Reset neuron\n        self.v_mem[neuron_id] = self.neuron_params[\"v_reset\"]\n        self.refractory_time[neuron_id] = self.current_time + self.neuron_params[\"refractory_period\"]\n        self.last_spike_time[neuron_id] = self.current_time\n        \n        return SpikeEvent(\n            neuron_id=neuron_id,\n            timestamp=self.current_time,\n            layer_id=self._get_layer_id(neuron_id),\n            amplitude=1.0\n        )\n    \n    def _propagate_spike(self, spike_event: SpikeEvent):\n        \"\"\"Propagate spike to connected neurons.\"\"\"\n        source_id = spike_event.neuron_id\n        \n        if source_id in self.topology:\n            for target_id in self.topology[source_id]:\n                # Get synaptic weight\n                weight = self.weights.get((source_id, target_id), 0.0)\n                \n                # Create synaptic input event with delay\n                synaptic_delay = 1.0  # 1ms delay\n                synaptic_event = SpikeEvent(\n                    neuron_id=target_id,\n                    timestamp=spike_event.timestamp + synaptic_delay,\n                    layer_id=spike_event.layer_id + 1,\n                    amplitude=weight,\n                    metadata={\"source\": source_id}\n                )\n                \n                self.event_queue.add_event(synaptic_event)\n    \n    def _passive_decay(self):\n        \"\"\"Apply passive membrane and synaptic decay.\"\"\"\n        tau_mem = self.neuron_params[\"tau_mem\"]\n        tau_syn = self.neuron_params[\"tau_syn\"]\n        \n        # Membrane potential decay (towards resting potential)\n        alpha_mem = np.exp(-self.dt / tau_mem)\n        self.v_mem = self.v_mem * alpha_mem + self.neuron_params[\"v_rest\"] * (1 - alpha_mem)\n        \n        # Synaptic current decay\n        alpha_syn = np.exp(-self.dt / tau_syn)\n        self.i_syn *= alpha_syn\n    \n    def _get_layer_id(self, neuron_id: int) -> int:\n        \"\"\"Determine layer ID for neuron (simplified).\"\"\"\n        # Simple heuristic: assume neurons are ordered by layer\n        return neuron_id // 100  # Assume 100 neurons per layer\n    \n    def get_statistics(self) -> Dict:\n        \"\"\"Get processing statistics.\"\"\"\n        total_possible_updates = self.current_time / self.dt * self.num_neurons\n        actual_updates = self.total_events_processed\n        \n        return {\n            \"total_events_processed\": self.total_events_processed,\n            \"simulation_time\": self.current_time,\n            \"num_neurons\": self.num_neurons,\n            \"computation_efficiency\": actual_updates / max(total_possible_updates, 1),\n            \"events_per_ms\": self.total_events_processed / max(self.current_time, 1),\n            \"queue_utilization\": len(self.event_queue.events) / self.max_queue_size\n        }\n    \n    def reset(self):\n        \"\"\"Reset processor state.\"\"\"\n        self.v_mem.fill_(self.neuron_params[\"v_rest\"])\n        self.i_syn.fill_(0.0)\n        self.refractory_time.fill_(0.0)\n        self.last_spike_time.fill_(-float('inf'))\n        self.event_queue.clear()\n        self.current_time = 0.0\n        self.total_events_processed = 0\n    \n    def benchmark_vs_synchronous(self, input_pattern: List[Tuple[int, float, float]]) -> Dict:\n        \"\"\"Benchmark event-driven vs synchronous processing.\n        \n        Args:\n            input_pattern: List of (neuron_id, current, timestamp) tuples\n            \n        Returns:\n            Performance comparison metrics\n        \"\"\"\n        # Reset and run event-driven\n        self.reset()\n        start_time = time.time()\n        \n        for neuron_id, current, timestamp in input_pattern:\n            self.add_external_input(neuron_id, current, timestamp)\n        \n        max_time = max(timestamp for _, _, timestamp in input_pattern) + 100\n        event_spikes = self.process_until_time(max_time)\n        event_driven_time = time.time() - start_time\n        \n        # Estimate synchronous processing time\n        synchronous_updates = (max_time / self.dt) * self.num_neurons\n        event_driven_updates = self.total_events_processed\n        \n        speedup = synchronous_updates / max(event_driven_updates, 1)\n        \n        return {\n            \"event_driven_time\": event_driven_time,\n            \"estimated_synchronous_updates\": int(synchronous_updates),\n            \"event_driven_updates\": event_driven_updates,\n            \"theoretical_speedup\": speedup,\n            \"output_spikes\": len(event_spikes),\n            \"sparsity\": 1.0 - (event_driven_updates / synchronous_updates)\n        }"