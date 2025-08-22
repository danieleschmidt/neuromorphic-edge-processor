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
    
    def decode_spikes_to_rate(self, spikes: torch.Tensor, window_size: float = 50.0) -> torch.Tensor:\n        \"\"\"Decode spike trains to firing rates using sliding window.\n        \n        Args:\n            spikes: Spike trains [batch_size, num_neurons, time_steps]\n            window_size: Sliding window size (ms)\n            \n        Returns:\n            Firing rates [batch_size, num_neurons, time_windows]\n        \"\"\"\n        batch_size, num_neurons, time_steps = spikes.shape\n        window_steps = int(window_size / self.dt)\n        \n        # Use convolution for efficient sliding window\n        kernel = torch.ones(1, 1, window_steps) / window_size * 1000.0  # Convert to Hz\n        \n        # Reshape for convolution\n        spikes_reshaped = spikes.view(-1, 1, time_steps)\n        \n        # Apply convolution\n        rates = torch.nn.functional.conv1d(\n            spikes_reshaped, kernel, padding=window_steps//2\n        )\n        \n        # Reshape back\n        rates = rates.view(batch_size, num_neurons, -1)\n        \n        return rates\n    \n    def detect_spike_patterns(self, spikes: torch.Tensor, pattern_length: int = 10) -> Dict:\n        \"\"\"Detect recurring spike patterns in the data.\n        \n        Args:\n            spikes: Spike trains [batch_size, num_neurons, time_steps] \n            pattern_length: Length of patterns to detect\n            \n        Returns:\n            Dictionary with detected patterns and their frequencies\n        \"\"\"\n        batch_size, num_neurons, time_steps = spikes.shape\n        patterns = {}\n        \n        for b in range(batch_size):\n            batch_patterns = {}\n            \n            # Sliding window over time\n            for t in range(time_steps - pattern_length + 1):\n                pattern = spikes[b, :, t:t+pattern_length]\n                \n                # Convert to tuple for hashing\n                pattern_key = tuple(pattern.flatten().int().tolist())\n                \n                if pattern_key in batch_patterns:\n                    batch_patterns[pattern_key] += 1\n                else:\n                    batch_patterns[pattern_key] = 1\n            \n            # Only keep patterns that occur more than once\n            significant_patterns = {k: v for k, v in batch_patterns.items() if v > 1}\n            patterns[f\"batch_{b}\"] = significant_patterns\n        \n        return patterns\n    \n    def compute_spike_train_metrics(self, spikes: torch.Tensor) -> Dict:\n        \"\"\"Compute various metrics for spike train analysis.\n        \n        Args:\n            spikes: Spike trains [batch_size, num_neurons, time_steps]\n            \n        Returns:\n            Dictionary of computed metrics\n        \"\"\"\n        batch_size, num_neurons, time_steps = spikes.shape\n        duration = time_steps * self.dt / 1000.0  # Duration in seconds\n        \n        metrics = {}\n        \n        # Firing rates\n        spike_counts = spikes.sum(dim=-1)  # [batch_size, num_neurons]\n        firing_rates = spike_counts / duration  # Hz\n        \n        metrics[\"mean_firing_rate\"] = firing_rates.mean().item()\n        metrics[\"std_firing_rate\"] = firing_rates.std().item()\n        metrics[\"max_firing_rate\"] = firing_rates.max().item()\n        \n        # Inter-spike intervals (ISI)\n        isis = []\n        for b in range(batch_size):\n            for n in range(num_neurons):\n                spike_times = torch.where(spikes[b, n] == 1)[0] * self.dt\n                if len(spike_times) > 1:\n                    isi = torch.diff(spike_times)\n                    isis.extend(isi.tolist())\n        \n        if isis:\n            isis_tensor = torch.tensor(isis)\n            metrics[\"mean_isi\"] = isis_tensor.mean().item()\n            metrics[\"std_isi\"] = isis_tensor.std().item()\n            metrics[\"cv_isi\"] = (isis_tensor.std() / isis_tensor.mean()).item()\n        \n        # Network synchrony\n        network_activity = spikes.sum(dim=1)  # [batch_size, time_steps]\n        synchrony_scores = []\n        \n        for b in range(batch_size):\n            activity = network_activity[b]\n            if activity.sum() > 0:\n                # Normalized variance of network activity\n                mean_activity = activity.mean()\n                var_activity = activity.var()\n                synchrony = var_activity / (mean_activity + 1e-8)\n                synchrony_scores.append(synchrony.item())\n        \n        if synchrony_scores:\n            metrics[\"mean_synchrony\"] = np.mean(synchrony_scores)\n        \n        # Sparsity\n        total_possible_spikes = batch_size * num_neurons * time_steps\n        actual_spikes = spikes.sum().item()\n        metrics[\"sparsity\"] = 1.0 - (actual_spikes / total_possible_spikes)\n        \n        return metrics\n    \n    def compute_cross_correlation(self, spikes1: torch.Tensor, spikes2: torch.Tensor, max_lag: int = 50) -> torch.Tensor:\n        \"\"\"Compute cross-correlation between spike trains.\n        \n        Args:\n            spikes1, spikes2: Spike trains [time_steps]\n            max_lag: Maximum lag to compute (in time steps)\n            \n        Returns:\n            Cross-correlation values [-max_lag:max_lag+1]\n        \"\"\"\n        # Convert to numpy for scipy\n        s1 = spikes1.cpu().numpy()\n        s2 = spikes2.cpu().numpy()\n        \n        # Compute full cross-correlation\n        correlation = signal.correlate(s1, s2, mode='full')\n        \n        # Extract relevant range\n        center = len(correlation) // 2\n        start = max(0, center - max_lag)\n        end = min(len(correlation), center + max_lag + 1)\n        \n        return torch.tensor(correlation[start:end])\n    \n    def compute_spike_field_coherence(self, spikes: torch.Tensor, lfp: torch.Tensor, frequencies: Optional[torch.Tensor] = None) -> Dict:\n        \"\"\"Compute spike-field coherence between spikes and LFP.\n        \n        Args:\n            spikes: Spike train [time_steps]\n            lfp: Local field potential [time_steps] \n            frequencies: Frequency bins for analysis\n            \n        Returns:\n            Coherence metrics across frequencies\n        \"\"\"\n        if frequencies is None:\n            frequencies = torch.linspace(1, 100, 50)  # 1-100 Hz\n        \n        # Convert to numpy\n        spikes_np = spikes.cpu().numpy()\n        lfp_np = lfp.cpu().numpy()\n        \n        coherence_values = []\n        \n        for freq in frequencies:\n            # Bandpass filter LFP around frequency\n            from scipy.signal import butter, filtfilt\n            \n            low = max(1, freq - 2)\n            high = freq + 2\n            nyquist = self.sampling_rate / 2\n            \n            if high < nyquist:\n                b, a = butter(3, [low/nyquist, high/nyquist], btype='band')\n                lfp_filtered = filtfilt(b, a, lfp_np)\n                \n                # Compute phase of filtered LFP\n                analytic_signal = signal.hilbert(lfp_filtered)\n                lfp_phase = np.angle(analytic_signal)\n                \n                # Get phases at spike times\n                spike_times = np.where(spikes_np == 1)[0]\n                if len(spike_times) > 10:  # Need sufficient spikes\n                    spike_phases = lfp_phase[spike_times]\n                    \n                    # Compute vector strength (coherence measure)\n                    mean_vector = np.mean(np.exp(1j * spike_phases))\n                    coherence = np.abs(mean_vector)\n                    coherence_values.append(coherence)\n                else:\n                    coherence_values.append(0.0)\n            else:\n                coherence_values.append(0.0)\n        \n        return {\n            \"frequencies\": frequencies.tolist(),\n            \"coherence\": coherence_values,\n            \"peak_frequency\": frequencies[np.argmax(coherence_values)].item(),\n            \"peak_coherence\": max(coherence_values)\n        }\n    \n    def generate_realistic_spike_trains(\n        self, \n        num_neurons: int, \n        duration: float,\n        base_rate: float = 10.0,\n        burst_probability: float = 0.1,\n        synchrony_level: float = 0.2\n    ) -> torch.Tensor:\n        \"\"\"Generate realistic spike trains with bursting and synchrony.\n        \n        Args:\n            num_neurons: Number of neurons\n            duration: Duration in ms\n            base_rate: Base firing rate in Hz\n            burst_probability: Probability of burst events\n            synchrony_level: Level of network synchrony [0,1]\n            \n        Returns:\n            Spike trains [1, num_neurons, time_steps]\n        \"\"\"\n        time_steps = int(duration / self.dt)\n        spikes = torch.zeros(1, num_neurons, time_steps)\n        \n        # Generate base Poisson spikes\n        base_spikes = self.encode_rate_to_spikes(\n            torch.full((1, num_neurons), base_rate), duration, \"poisson\"\n        )\n        \n        # Add bursting\n        for t in range(time_steps):\n            if torch.rand(1) < burst_probability * self.dt / 1000.0:\n                # Burst event: temporarily increase firing\n                burst_duration = int(50 / self.dt)  # 50ms bursts\n                burst_start = t\n                burst_end = min(t + burst_duration, time_steps)\n                \n                burst_neurons = torch.randperm(num_neurons)[:num_neurons//3]  # 1/3 of neurons\n                \n                for n in burst_neurons:\n                    # Higher firing rate during burst\n                    burst_rate = base_rate * 5\n                    burst_prob = burst_rate * self.dt / 1000.0\n                    \n                    for burst_t in range(burst_start, burst_end):\n                        if torch.rand(1) < burst_prob:\n                            spikes[0, n, burst_t] = 1.0\n        \n        # Add synchrony\n        if synchrony_level > 0:\n            for t in range(time_steps):\n                # Random synchronous events\n                if torch.rand(1) < synchrony_level * 0.01:\n                    sync_neurons = torch.randperm(num_neurons)[:int(num_neurons * synchrony_level)]\n                    spikes[0, sync_neurons, t] = 1.0\n        \n        # Combine with base spikes\n        combined_spikes = torch.clamp(base_spikes + spikes, 0, 1)\n        \n        return combined_spikes\n    \n    def visualize_spike_trains(\n        self,\n        spikes: torch.Tensor,\n        save_path: Optional[str] = None,\n        max_neurons: int = 50,\n        time_range: Optional[Tuple[float, float]] = None\n    ):\n        \"\"\"Visualize spike trains as a raster plot.\n        \n        Args:\n            spikes: Spike trains [batch_size, num_neurons, time_steps]\n            save_path: Path to save the plot\n            max_neurons: Maximum number of neurons to plot\n            time_range: Time range to plot (start_ms, end_ms)\n        \"\"\"\n        if spikes.dim() == 3:\n            spikes = spikes[0]  # Take first batch\n        \n        num_neurons, time_steps = spikes.shape[-2:]\n        num_neurons = min(num_neurons, max_neurons)\n        \n        # Time axis\n        time_axis = torch.arange(time_steps) * self.dt\n        \n        # Apply time range filter\n        if time_range is not None:\n            start_idx = int(time_range[0] / self.dt)\n            end_idx = int(time_range[1] / self.dt)\n            time_axis = time_axis[start_idx:end_idx]\n            spikes = spikes[:, start_idx:end_idx]\n        \n        # Create raster plot\n        plt.figure(figsize=(12, 8))\n        \n        for neuron_id in range(num_neurons):\n            spike_times = time_axis[spikes[neuron_id] == 1]\n            if len(spike_times) > 0:\n                plt.scatter(spike_times, [neuron_id] * len(spike_times), \n                           s=1, c='black', alpha=0.7)\n        \n        plt.xlabel('Time (ms)')\n        plt.ylabel('Neuron ID')\n        plt.title('Spike Train Raster Plot')\n        plt.grid(True, alpha=0.3)\n        \n        if save_path:\n            plt.savefig(save_path, dpi=300, bbox_inches='tight')\n            plt.close()\n        else:\n            plt.show()"