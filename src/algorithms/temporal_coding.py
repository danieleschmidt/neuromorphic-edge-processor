"""Temporal coding algorithms for spiking neural networks."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import signal


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
    
    def encode_latency(self, values: torch.Tensor, min_latency: float = 5.0, max_latency: float = 45.0) -> torch.Tensor:
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
        spikes = torch.zeros(batch_size, num_features, self.time_steps)
        
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
    
    def encode_population_vector(self, values: torch.Tensor, num_neurons_per_feature: int = 10) -> torch.Tensor:
        """Encode values using population vector coding.
        
        Each feature is encoded by multiple neurons with different preferred values.
        
        Args:
            values: Input values [batch_size, num_features] 
            num_neurons_per_feature: Number of neurons per feature
            
        Returns:
            Spike trains [batch_size, num_features * num_neurons_per_feature, time_steps]
        """
        batch_size, num_features = values.shape
        
        # Create preferred values for each neuron (uniformly distributed)
        preferred_values = torch.linspace(0, 1, num_neurons_per_feature).unsqueeze(0)  # [1, num_neurons_per_feature]
        
        # Expand values and preferred values for broadcasting
        values_expanded = values.unsqueeze(-1)  # [batch_size, num_features, 1]
        preferred_expanded = preferred_values.unsqueeze(0).unsqueeze(0)  # [1, 1, num_neurons_per_feature]
        
        # Gaussian tuning curves
        sigma = 0.2  # Width of tuning curves
        responses = torch.exp(-0.5 * ((values_expanded - preferred_expanded) / sigma) ** 2)
        
        # Reshape for encoding
        responses = responses.view(batch_size, num_features * num_neurons_per_feature)
        
        # Encode responses as latencies
        encoded_spikes = self.encode_latency(responses)
        
        return encoded_spikes
    
    def encode_phase(self, values: torch.Tensor, base_frequency: float = 20.0) -> torch.Tensor:
        """Encode values as phase relationships to an oscillation.
        
        Args:
            values: Input values [batch_size, num_features] normalized to [0,1]
            base_frequency: Base oscillation frequency (Hz)
            
        Returns:
            Spike trains [batch_size, num_features, time_steps]
        """
        batch_size, num_features = values.shape
        spikes = torch.zeros(batch_size, num_features, self.time_steps)
        
        # Time vector
        t = torch.arange(self.time_steps) * self.dt / 1000.0  # Convert to seconds
        
        # Base oscillation phase
        base_phase = 2 * np.pi * base_frequency * t
        
        for b in range(batch_size):
            for f in range(num_features):
                # Phase offset based on value (0 to 2π)
                phase_offset = values[b, f] * 2 * np.pi
                
                # Generate oscillation with phase offset
                oscillation = torch.sin(base_phase + phase_offset)
                
                # Detect positive zero crossings as spikes
                zero_crossings = torch.diff(torch.sign(oscillation)) > 0
                spike_indices = torch.where(zero_crossings)[0] + 1
                
                # Set spikes
                for idx in spike_indices:
                    if idx < self.time_steps:
                        spikes[b, f, idx] = 1.0
        
        return spikes
    
    def encode_rank_order(self, values: torch.Tensor) -> torch.Tensor:
        """Encode values using rank order coding.
        
        Features with higher values spike earlier in order.
        
        Args:
            values: Input values [batch_size, num_features]
            
        Returns:
            Spike trains [batch_size, num_features, time_steps]
        """
        batch_size, num_features = values.shape
        spikes = torch.zeros(batch_size, num_features, self.time_steps)
        
        spike_interval = max(1, self.time_steps // num_features)  # Spacing between spikes
        
        for b in range(batch_size):
            # Sort features by value (descending order)
            sorted_indices = torch.argsort(values[b], descending=True)
            
            # Generate spikes in rank order
            for rank, feature_idx in enumerate(sorted_indices):
                if values[b, feature_idx] > 0:  # Only spike if value > 0
                    spike_time = rank * spike_interval
                    if spike_time < self.time_steps:
                        spikes[b, feature_idx, spike_time] = 1.0
        
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
    
    def decode_latency(self, spikes: torch.Tensor, min_latency: float = 5.0, max_latency: float = 45.0) -> torch.Tensor:
        """Decode spike latencies to continuous values.
        
        Args:
            spikes: Spike trains [batch_size, num_features, time_steps]
            min_latency: Minimum spike latency (ms)
            max_latency: Maximum spike latency (ms)
            
        Returns:
            Decoded values [batch_size, num_features]
        """
        batch_size, num_features, _ = spikes.shape
        values = torch.zeros(batch_size, num_features)
        
        # Find first spike time for each feature
        for b in range(batch_size):
            for f in range(num_features):
                spike_times = torch.where(spikes[b, f] > 0)[0]
                if len(spike_times) > 0:
                    first_spike_time = spike_times[0].item() * self.dt
                    # Map latency back to value
                    value = 1.0 - (first_spike_time - min_latency) / (max_latency - min_latency)
                    values[b, f] = torch.clamp(torch.tensor(value), 0, 1)
        
        return values
    
    def decode_rate_sliding_window(self, spikes: torch.Tensor, window_size: float = 10.0) -> torch.Tensor:
        """Decode spikes using sliding window rate estimation.
        
        Args:
            spikes: Spike trains [batch_size, num_features, time_steps]
            window_size: Sliding window size (ms)
            
        Returns:
            Instantaneous rates [batch_size, num_features, time_windows]
        """
        batch_size, num_features, time_steps = spikes.shape
        window_steps = int(window_size / self.dt)
        
        # Use convolution for efficient sliding window
        kernel = torch.ones(1, 1, window_steps) / window_size * 1000.0  # Convert to Hz
        
        # Process each batch and feature
        rates_list = []
        for b in range(batch_size):
            batch_rates = []
            for f in range(num_features):
                spike_train = spikes[b, f].unsqueeze(0).unsqueeze(0)  # [1, 1, time_steps]
                rate = torch.nn.functional.conv1d(spike_train, kernel, padding=window_steps//2)
                batch_rates.append(rate.squeeze())
            rates_list.append(torch.stack(batch_rates))
        
        return torch.stack(rates_list)
    
    def decode_population_vector(self, spikes: torch.Tensor, num_neurons_per_feature: int = 10) -> torch.Tensor:
        """Decode population vector coded spikes.
        
        Args:
            spikes: Spike trains [batch_size, num_features * num_neurons_per_feature, time_steps]
            num_neurons_per_feature: Number of neurons per original feature
            
        Returns:
            Decoded values [batch_size, num_original_features]
        """
        batch_size, total_neurons, time_steps = spikes.shape
        num_features = total_neurons // num_neurons_per_feature
        
        # Reshape to separate features and neurons
        spikes_reshaped = spikes.view(batch_size, num_features, num_neurons_per_feature, time_steps)
        
        # Compute firing rates for each neuron
        rates = spikes_reshaped.mean(dim=-1)  # [batch_size, num_features, num_neurons_per_feature]
        
        # Preferred values for each neuron
        preferred_values = torch.linspace(0, 1, num_neurons_per_feature)
        
        # Weighted average based on firing rates
        decoded_values = torch.zeros(batch_size, num_features)
        
        for b in range(batch_size):
            for f in range(num_features):
                total_rate = rates[b, f].sum()
                if total_rate > 0:
                    weighted_sum = (rates[b, f] * preferred_values).sum()
                    decoded_values[b, f] = weighted_sum / total_rate
        
        return decoded_values
    
    def decode_phase(self, spikes: torch.Tensor, base_frequency: float = 20.0) -> torch.Tensor:
        """Decode phase-coded spikes.
        
        Args:
            spikes: Spike trains [batch_size, num_features, time_steps]
            base_frequency: Base oscillation frequency (Hz)
            
        Returns:
            Decoded phase values [batch_size, num_features] normalized to [0,1]
        """
        batch_size, num_features, time_steps = spikes.shape
        phases = torch.zeros(batch_size, num_features)
        
        # Time vector
        t = torch.arange(time_steps) * self.dt / 1000.0  # Convert to seconds
        
        for b in range(batch_size):
            for f in range(num_features):
                spike_times_idx = torch.where(spikes[b, f] > 0)[0]
                
                if len(spike_times_idx) > 0:
                    # Use first spike for phase estimation
                    first_spike_time = spike_times_idx[0].item() * self.dt / 1000.0
                    
                    # Estimate phase at first spike
                    phase = 2 * np.pi * base_frequency * first_spike_time
                    
                    # Normalize to [0, 2π] and then to [0, 1]
                    phase_normalized = (phase % (2 * np.pi)) / (2 * np.pi)
                    phases[b, f] = phase_normalized
        
        return phases
    
    def compute_temporal_precision(self, predicted_spikes: torch.Tensor, target_spikes: torch.Tensor) -> Dict[str, float]:
        """Compute temporal precision metrics.
        
        Args:
            predicted_spikes: Predicted spike trains [batch_size, num_features, time_steps]
            target_spikes: Target spike trains [batch_size, num_features, time_steps]
            
        Returns:
            Dictionary of precision metrics
        """
        # Victor-Purpura spike train distance
        vp_distances = []
        
        for b in range(predicted_spikes.size(0)):
            for f in range(predicted_spikes.size(1)):
                pred_times = torch.where(predicted_spikes[b, f] > 0)[0] * self.dt
                target_times = torch.where(target_spikes[b, f] > 0)[0] * self.dt
                
                if len(pred_times) > 0 and len(target_times) > 0:
                    # Simplified VP distance (cost parameter q = 1/ms)
                    q = 1.0  # 1/ms
                    cost_matrix = torch.abs(pred_times.unsqueeze(1) - target_times.unsqueeze(0))
                    
                    # Find minimum cost assignment (simplified)
                    min_costs = cost_matrix.min(dim=1)[0]
                    vp_distance = min_costs.sum().item()
                    vp_distances.append(vp_distance)
        
        # Correlation-based measures
        correlation_scores = []
        
        for b in range(predicted_spikes.size(0)):
            for f in range(predicted_spikes.size(1)):
                pred = predicted_spikes[b, f]
                target = target_spikes[b, f]
                
                # Pearson correlation
                if pred.std() > 0 and target.std() > 0:
                    correlation = torch.corrcoef(torch.stack([pred, target]))[0, 1]
                    if not torch.isnan(correlation):
                        correlation_scores.append(correlation.item())
        
        return {
            "mean_vp_distance": np.mean(vp_distances) if vp_distances else 0.0,
            "std_vp_distance": np.std(vp_distances) if vp_distances else 0.0,
            "mean_correlation": np.mean(correlation_scores) if correlation_scores else 0.0,
            "std_correlation": np.std(correlation_scores) if correlation_scores else 0.0,
            "num_comparisons": len(correlation_scores)
        }