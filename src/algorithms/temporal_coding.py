"""Temporal coding algorithms for spiking neural networks."""

import numpy as np
from typing import Dict, List, Optional, Tuple


class TemporalEncoder:
    """Encode continuous values using temporal spike patterns."""
    
    def __init__(self, dt: float = 1.0, encoding_window: float = 50.0):
        """Initialize temporal encoder.
        
        Args:
            dt: Time step (ms)
            encoding_window: Time window for encoding (ms)
        """
        self.dt = dt
        self.encoding_window = encoding_window
        self.time_steps = int(encoding_window / dt)
    
    def encode_latency(self, values: np.ndarray, min_latency: float = 5.0, max_latency: float = 45.0) -> np.ndarray:
        """Encode values as spike latencies.
        
        Higher values produce earlier spikes (lower latency).
        
        Args:
            values: Input values [batch_size, num_features] normalized to [0,1]
            min_latency: Minimum spike latency (ms)
            max_latency: Maximum spike latency (ms)
            
        Returns:
            Spike trains [batch_size, num_features, time_steps]
        """
        batch_size, num_features = values.shape
        spikes = np.zeros((batch_size, num_features, self.time_steps))
        
        # Map values to latencies (higher value = lower latency)
        latencies = max_latency - values * (max_latency - min_latency)
        
        # Generate spikes at computed latencies
        for b in range(batch_size):
            for f in range(num_features):
                if values[b, f] > 0:  # Only create spike if value > 0
                    spike_time = int(latencies[b, f] / self.dt)
                    if 0 <= spike_time < self.time_steps:
                        spikes[b, f, spike_time] = 1.0
        
        return spikes


class TemporalDecoder:
    """Decode temporal spike patterns back to continuous values."""
    
    def __init__(self, dt: float = 1.0, decoding_window: float = 50.0):
        """Initialize temporal decoder.
        
        Args:
            dt: Time step (ms)
            decoding_window: Time window for decoding (ms)
        """
        self.dt = dt
        self.decoding_window = decoding_window
        self.time_steps = int(decoding_window / dt)
    
    def decode_latency(self, spikes: np.ndarray, min_latency: float = 5.0, max_latency: float = 45.0) -> np.ndarray:
        """Decode spike latencies to continuous values.
        
        Args:
            spikes: Spike trains [batch_size, num_features, time_steps]
            min_latency: Minimum spike latency (ms)
            max_latency: Maximum spike latency (ms)
            
        Returns:
            Decoded values [batch_size, num_features]
        """
        batch_size, num_features, _ = spikes.shape
        values = np.zeros((batch_size, num_features))
        
        # Find first spike time for each feature
        for b in range(batch_size):
            for f in range(num_features):
                spike_times = np.where(spikes[b, f] > 0)[0]
                if len(spike_times) > 0:
                    first_spike_time = spike_times[0] * self.dt
                    # Map latency back to value
                    value = 1.0 - (first_spike_time - min_latency) / (max_latency - min_latency)
                    values[b, f] = np.clip(value, 0, 1)
        
        return values