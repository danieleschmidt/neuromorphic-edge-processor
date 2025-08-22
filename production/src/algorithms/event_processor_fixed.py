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
        self.neuron_params = neuron_params or {
            "tau_mem": 20.0,
            "tau_syn": 5.0,
            "v_thresh": -50.0,
            "v_reset": -70.0,
            "v_rest": -65.0,
            "refractory_period": 2.0
        }
        
        # Initialize network state
        self.num_neurons = max(max(self.topology.keys()), 
                              max(max(targets) if targets else 0 
                                  for targets in self.topology.values())) + 1
        
        # Neuron state variables
        self.v_mem = torch.full((self.num_neurons,), self.neuron_params["v_rest"])
        self.i_syn = torch.zeros(self.num_neurons)
        self.refractory_time = torch.zeros(self.num_neurons)
        self.last_spike_time = torch.full((self.num_neurons,), -float('inf'))
        
        # Synaptic weights (can be learned)
        self.weights = self._initialize_weights()
        
        # Event processing
        self.event_queue = EventQueue(max_queue_size)
        self.current_time = 0.0
        
        # Statistics
        self.total_events_processed = 0
        self.computation_savings = 0.0
        
    def _initialize_weights(self) -> Dict[Tuple[int, int], float]:
        """Initialize synaptic weights between connected neurons."""
        weights = {}
        for source_id, targets in self.topology.items():
            for target_id in targets:
                # Random initial weights
                weights[(source_id, target_id)] = torch.randn(1).item() * 0.1
        return weights
    
    def add_external_input(self, neuron_id: int, current: float, timestamp: float):
        """Add external input current to specific neuron."""
        event = SpikeEvent(
            neuron_id=neuron_id,
            timestamp=timestamp,
            layer_id=0,  # Input layer
            amplitude=current,
            metadata={"type": "external_input"}
        )
        self.event_queue.add_event(event)
    
    def process_until_time(self, end_time: float) -> List[SpikeEvent]:
        """Process all events up to specified time.
        
        Args:
            end_time: Final simulation time (ms)
            
        Returns:
            List of generated spike events
        """
        output_spikes = []
        
        while self.current_time < end_time:
            # Get next event time or advance by dt
            next_event_time = self.event_queue.peek_next_time()
            
            if next_event_time is not None and next_event_time <= end_time:
                # Process events at this time
                events = self.event_queue.get_events_until(next_event_time)
                self.current_time = next_event_time
                
                # Update only neurons that received events
                active_neurons = set()
                for event in events:
                    active_neurons.add(event.neuron_id)
                    self._process_input_event(event)
                
                # Check for spikes in active neurons
                for neuron_id in active_neurons:
                    if self._check_spike_condition(neuron_id):
                        spike_event = self._generate_spike(neuron_id)
                        output_spikes.append(spike_event)
                        
                        # Propagate to connected neurons
                        self._propagate_spike(spike_event)
                
                self.total_events_processed += len(events)
                
            else:
                # No events, advance time
                self.current_time = min(self.current_time + self.dt, end_time)
                
                # Passive decay for all neurons (could be optimized further)
                self._passive_decay()
        
        return output_spikes
    
    def _process_input_event(self, event: SpikeEvent):
        """Process input event for specific neuron."""
        neuron_id = event.neuron_id
        
        if event.metadata and event.metadata.get("type") == "external_input":
            # External current input
            self.i_syn[neuron_id] += event.amplitude
        else:
            # Synaptic input from other neurons
            self.i_syn[neuron_id] += event.amplitude
    
    def _check_spike_condition(self, neuron_id: int) -> bool:
        """Check if neuron should generate a spike."""
        # Check refractory period
        if self.refractory_time[neuron_id] > self.current_time:
            return False
        
        # Update membrane potential
        dt_since_last = min(self.dt, self.current_time - self.last_spike_time[neuron_id])
        if dt_since_last > 0:
            # Euler integration
            tau_mem = self.neuron_params["tau_mem"]
            dv = (self.neuron_params["v_rest"] - self.v_mem[neuron_id] + self.i_syn[neuron_id]) / tau_mem
            self.v_mem[neuron_id] += dv * dt_since_last
        
        # Check threshold
        return self.v_mem[neuron_id] >= self.neuron_params["v_thresh"]
    
    def _generate_spike(self, neuron_id: int) -> SpikeEvent:
        """Generate spike event and reset neuron."""
        # Reset neuron
        self.v_mem[neuron_id] = self.neuron_params["v_reset"]
        self.refractory_time[neuron_id] = self.current_time + self.neuron_params["refractory_period"]
        self.last_spike_time[neuron_id] = self.current_time
        
        return SpikeEvent(
            neuron_id=neuron_id,
            timestamp=self.current_time,
            layer_id=self._get_layer_id(neuron_id),
            amplitude=1.0
        )
    
    def _propagate_spike(self, spike_event: SpikeEvent):
        """Propagate spike to connected neurons."""
        source_id = spike_event.neuron_id
        
        if source_id in self.topology:
            for target_id in self.topology[source_id]:
                # Get synaptic weight
                weight = self.weights.get((source_id, target_id), 0.0)
                
                # Create synaptic input event with delay
                synaptic_delay = 1.0  # 1ms delay
                synaptic_event = SpikeEvent(
                    neuron_id=target_id,
                    timestamp=spike_event.timestamp + synaptic_delay,
                    layer_id=spike_event.layer_id + 1,
                    amplitude=weight,
                    metadata={"source": source_id}
                )
                
                self.event_queue.add_event(synaptic_event)
    
    def _passive_decay(self):
        """Apply passive membrane and synaptic decay."""
        tau_mem = self.neuron_params["tau_mem"]
        tau_syn = self.neuron_params["tau_syn"]
        
        # Membrane potential decay (towards resting potential)
        alpha_mem = np.exp(-self.dt / tau_mem)
        self.v_mem = self.v_mem * alpha_mem + self.neuron_params["v_rest"] * (1 - alpha_mem)
        
        # Synaptic current decay
        alpha_syn = np.exp(-self.dt / tau_syn)
        self.i_syn *= alpha_syn
    
    def _get_layer_id(self, neuron_id: int) -> int:
        """Determine layer ID for neuron (simplified)."""
        # Simple heuristic: assume neurons are ordered by layer
        return neuron_id // 100  # Assume 100 neurons per layer
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        total_possible_updates = self.current_time / self.dt * self.num_neurons
        actual_updates = self.total_events_processed
        
        return {
            "total_events_processed": self.total_events_processed,
            "simulation_time": self.current_time,
            "num_neurons": self.num_neurons,
            "computation_efficiency": actual_updates / max(total_possible_updates, 1),
            "events_per_ms": self.total_events_processed / max(self.current_time, 1),
            "queue_utilization": len(self.event_queue.events) / self.max_queue_size
        }
    
    def reset(self):
        """Reset processor state."""
        self.v_mem.fill_(self.neuron_params["v_rest"])
        self.i_syn.fill_(0.0)
        self.refractory_time.fill_(0.0)
        self.last_spike_time.fill_(-float('inf'))
        self.event_queue.clear()
        self.current_time = 0.0
        self.total_events_processed = 0
    
    def benchmark_vs_synchronous(self, input_pattern: List[Tuple[int, float, float]]) -> Dict:
        """Benchmark event-driven vs synchronous processing.
        
        Args:
            input_pattern: List of (neuron_id, current, timestamp) tuples
            
        Returns:
            Performance comparison metrics
        """
        # Reset and run event-driven
        self.reset()
        start_time = time.time()
        
        for neuron_id, current, timestamp in input_pattern:
            self.add_external_input(neuron_id, current, timestamp)
        
        max_time = max(timestamp for _, _, timestamp in input_pattern) + 100
        event_spikes = self.process_until_time(max_time)
        event_driven_time = time.time() - start_time
        
        # Estimate synchronous processing time
        synchronous_updates = (max_time / self.dt) * self.num_neurons
        event_driven_updates = self.total_events_processed
        
        speedup = synchronous_updates / max(event_driven_updates, 1)
        
        return {
            "event_driven_time": event_driven_time,
            "estimated_synchronous_updates": int(synchronous_updates),
            "event_driven_updates": event_driven_updates,
            "theoretical_speedup": speedup,
            "output_spikes": len(event_spikes),
            "sparsity": 1.0 - (event_driven_updates / synchronous_updates)
        }