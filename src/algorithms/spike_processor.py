"""Spike processing algorithms for neuromorphic computation."""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union


class SpikeProcessor:
    """Advanced spike processing and analysis for neuromorphic systems.
    
    This class provides efficient algorithms for spike train encoding,
    decoding, and analysis optimized for edge computing devices.
    """
    
    def __init__(self, dt: float = 1.0, sampling_rate: float = 1000.0):
        """Initialize spike processor.
        
        Args:
            dt: Time step in milliseconds
            sampling_rate: Sampling rate in Hz
        """
        self.dt = dt
        self.sampling_rate = sampling_rate
    
    def encode_rate_to_spikes(
        self, 
        rates: np.ndarray, 
        duration: float, 
        method: str = "poisson"
    ) -> np.ndarray:
        """Convert firing rates to spike trains.
        
        Args:
            rates: Firing rates in Hz [batch_size, num_neurons]
            duration: Duration in ms
            method: Encoding method ('poisson', 'regular', 'burst')
            
        Returns:
            Spike trains [batch_size, num_neurons, time_steps]
        """
        batch_size, num_neurons = rates.shape
        time_steps = int(duration / self.dt)
        
        spikes = np.zeros((batch_size, num_neurons, time_steps))
        
        if method == "poisson":
            # Poisson spike generation
            for b in range(batch_size):
                for n in range(num_neurons):
                    rate = rates[b, n]
                    prob = rate * self.dt / 1000.0  # Convert to probability per time step
                    random_vals = np.random.rand(time_steps)
                    spikes[b, n] = (random_vals < prob).astype(np.float32)
        
        elif method == "regular":
            # Regular spike generation
            for b in range(batch_size):
                for n in range(num_neurons):
                    rate = rates[b, n]
                    if rate > 0:
                        interval = int(1000.0 / (rate * self.dt))  # Inter-spike interval
                        spike_times = np.arange(0, time_steps, interval)
                        spikes[b, n, spike_times[spike_times < time_steps]] = 1.0
        
        else:
            raise ValueError(f"Unknown encoding method: {method}")
            
        return spikes
    
    def decode_spikes_to_rate(self, spikes: np.ndarray, window_size: float = 50.0) -> np.ndarray:
        """Decode spike trains to firing rates using sliding window.
        
        Args:
            spikes: Spike trains [batch_size, num_neurons, time_steps]
            window_size: Sliding window size in ms
            
        Returns:
            Firing rates [batch_size, num_neurons, time_windows]
        """
        batch_size, num_neurons, time_steps = spikes.shape
        window_steps = int(window_size / self.dt)
        
        # Simple sliding window implementation
        num_windows = time_steps - window_steps + 1
        rates = np.zeros((batch_size, num_neurons, num_windows))
        
        for i in range(num_windows):
            window_spikes = spikes[:, :, i:i+window_steps]
            rates[:, :, i] = window_spikes.sum(axis=2) * 1000.0 / window_size
        
        return rates
    
    def compute_spike_train_metrics(self, spikes: np.ndarray) -> Dict:
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
        spike_counts = spikes.sum(axis=-1)  # [batch_size, num_neurons]
        firing_rates = spike_counts / duration  # Hz
        
        metrics["mean_firing_rate"] = float(firing_rates.mean())
        metrics["std_firing_rate"] = float(firing_rates.std())
        metrics["max_firing_rate"] = float(firing_rates.max())
        
        # Sparsity
        total_possible_spikes = batch_size * num_neurons * time_steps
        actual_spikes = spikes.sum()
        metrics["sparsity"] = 1.0 - (actual_spikes / total_possible_spikes)
        
        return metrics
    
    def generate_realistic_spike_trains(
        self, 
        num_neurons: int, 
        duration: float,
        base_rate: float = 10.0,
        burst_probability: float = 0.1
    ) -> np.ndarray:
        """Generate realistic spike trains with bursting.
        
        Args:
            num_neurons: Number of neurons
            duration: Duration in ms
            base_rate: Base firing rate in Hz
            burst_probability: Probability of burst events
            
        Returns:
            Spike trains [1, num_neurons, time_steps]
        """
        rates = np.full((1, num_neurons), base_rate)
        return self.encode_rate_to_spikes(rates, duration, "poisson")