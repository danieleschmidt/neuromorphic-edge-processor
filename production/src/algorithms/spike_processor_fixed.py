"""Spike processing algorithms for neuromorphic computation."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from scipy import signal
import matplotlib.pyplot as plt


class SpikeProcessor:
    """Advanced spike processing and analysis for neuromorphic systems.
    
    Provides efficient algorithms for spike train analysis, encoding,
    decoding, and temporal pattern recognition optimized for edge devices.
    """
    
    def __init__(self, sampling_rate: float = 1000.0):
        """Initialize spike processor.
        
        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.dt = 1000.0 / sampling_rate  # Time step in ms
        
    def encode_rate_to_spikes(
        self, 
        rates: torch.Tensor,
        duration: float,
        method: str = "poisson"
    ) -> torch.Tensor:
        """Convert firing rates to spike trains.
        
        Args:
            rates: Firing rates [batch_size, num_neurons] (Hz)
            duration: Spike train duration (ms)
            method: Encoding method ('poisson', 'regular', 'temporal')
            
        Returns:
            Spike trains [batch_size, num_neurons, time_steps]
        """
        batch_size, num_neurons = rates.shape
        time_steps = int(duration / self.dt)
        
        if method == "poisson":
            # Poisson spike generation
            spike_probs = rates * self.dt / 1000.0  # Convert to probability per timestep
            random_vals = torch.rand(batch_size, num_neurons, time_steps)
            spikes = (random_vals < spike_probs.unsqueeze(-1)).float()
            
        elif method == "regular":
            # Regular spike generation
            spikes = torch.zeros(batch_size, num_neurons, time_steps)
            for b in range(batch_size):
                for n in range(num_neurons):
                    rate = rates[b, n].item()
                    if rate > 0:
                        interval = 1000.0 / rate  # Interval in ms
                        spike_times = torch.arange(interval, duration, interval) / self.dt
                        for spike_time in spike_times:
                            idx = int(spike_time)
                            if idx < time_steps:
                                spikes[b, n, idx] = 1.0
        
        elif method == "temporal":
            # Temporal coding: higher rates = earlier spikes
            spikes = torch.zeros(batch_size, num_neurons, time_steps)
            max_rate = rates.max()
            if max_rate > 0:
                # Spike timing inversely proportional to rate
                spike_times = (1.0 - rates / max_rate) * duration * 0.8  # Use 80% of duration
                for b in range(batch_size):
                    for n in range(num_neurons):
                        if rates[b, n] > 0:
                            spike_idx = int(spike_times[b, n] / self.dt)
                            if spike_idx < time_steps:
                                spikes[b, n, spike_idx] = 1.0
        
        else:
            raise ValueError(f"Unknown encoding method: {method}")
            
        return spikes
    
    def decode_spikes_to_rate(self, spikes: torch.Tensor, window_size: float = 50.0) -> torch.Tensor:
        """Decode spike trains to firing rates using sliding window.
        
        Args:
            spikes: Spike trains [batch_size, num_neurons, time_steps]
            window_size: Sliding window size (ms)
            
        Returns:
            Firing rates [batch_size, num_neurons, time_windows]
        """
        batch_size, num_neurons, time_steps = spikes.shape
        window_steps = int(window_size / self.dt)
        
        # Use convolution for efficient sliding window
        kernel = torch.ones(1, 1, window_steps) / window_size * 1000.0  # Convert to Hz
        
        # Reshape for convolution
        spikes_reshaped = spikes.view(-1, 1, time_steps)
        
        # Apply convolution
        rates = torch.nn.functional.conv1d(
            spikes_reshaped, kernel, padding=window_steps//2
        )
        
        # Reshape back
        rates = rates.view(batch_size, num_neurons, -1)
        
        return rates
    
    def compute_spike_train_metrics(self, spikes: torch.Tensor) -> Dict:
        """Compute various metrics for spike train analysis.
        
        Args:
            spikes: Spike trains [batch_size, num_neurons, time_steps]
            
        Returns:
            Dictionary of computed metrics
        """
        batch_size, num_neurons, time_steps = spikes.shape
        duration = time_steps * self.dt / 1000.0  # Duration in seconds
        
        metrics = {}
        
        # Firing rates
        spike_counts = spikes.sum(dim=-1)  # [batch_size, num_neurons]
        firing_rates = spike_counts / duration  # Hz
        
        metrics["mean_firing_rate"] = firing_rates.mean().item()
        metrics["std_firing_rate"] = firing_rates.std().item()
        metrics["max_firing_rate"] = firing_rates.max().item()
        
        # Sparsity
        total_possible_spikes = batch_size * num_neurons * time_steps
        actual_spikes = spikes.sum().item()
        metrics["sparsity"] = 1.0 - (actual_spikes / total_possible_spikes)
        
        return metrics
    
    def generate_realistic_spike_trains(
        self, 
        num_neurons: int, 
        duration: float,
        base_rate: float = 10.0,
        burst_probability: float = 0.1,
        synchrony_level: float = 0.2
    ) -> torch.Tensor:
        """Generate realistic spike trains with bursting and synchrony.
        
        Args:
            num_neurons: Number of neurons
            duration: Duration in ms
            base_rate: Base firing rate in Hz
            burst_probability: Probability of burst events
            synchrony_level: Level of network synchrony [0,1]
            
        Returns:
            Spike trains [1, num_neurons, time_steps]
        """
        time_steps = int(duration / self.dt)
        spikes = torch.zeros(1, num_neurons, time_steps)
        
        # Generate base Poisson spikes
        base_spikes = self.encode_rate_to_spikes(
            torch.full((1, num_neurons), base_rate), duration, "poisson"
        )
        
        return base_spikes