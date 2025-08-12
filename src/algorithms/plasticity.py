"""Synaptic plasticity algorithms for learning in spiking neural networks."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


class STDPLearning:
    """Spike-Timing Dependent Plasticity (STDP) learning rule.
    
    Implements both classical STDP and advanced variants for synaptic weight updates
    based on the relative timing of pre- and post-synaptic spikes.
    """
    
    def __init__(
        self,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        w_min: float = 0.0,
        w_max: float = 1.0,
        stdp_type: str = "classical"
    ):
        """Initialize STDP learning.
        
        Args:
            tau_plus: Time constant for potentiation (ms)
            tau_minus: Time constant for depression (ms) 
            A_plus: Amplitude of potentiation
            A_minus: Amplitude of depression
            w_min: Minimum weight value
            w_max: Maximum weight value
            stdp_type: Type of STDP ('classical', 'triplet', 'voltage_dependent')
        """
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.w_min = w_min
        self.w_max = w_max
        self.stdp_type = stdp_type
        
    def update_weights(
        self, 
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        dt: float = 1.0
    ) -> torch.Tensor:
        """Update synaptic weights based on STDP.
        
        Args:
            weights: Current weights [output_size, input_size]
            pre_spikes: Pre-synaptic spikes [batch_size, input_size, time_steps]
            post_spikes: Post-synaptic spikes [batch_size, output_size, time_steps]
            dt: Time step (ms)
            
        Returns:
            Updated weights
        """
        if self.stdp_type == "classical":
            return self._classical_stdp(weights, pre_spikes, post_spikes, dt)
        elif self.stdp_type == "triplet":
            return self._triplet_stdp(weights, pre_spikes, post_spikes, dt)
        else:
            raise ValueError(f"Unknown STDP type: {self.stdp_type}")
    
    def _classical_stdp(
        self, 
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """Classical pairwise STDP implementation."""
        batch_size, input_size, time_steps = pre_spikes.shape
        output_size = post_spikes.size(1)
        
        weight_updates = torch.zeros_like(weights)
        
        # Exponential decay kernels
        time_kernel = torch.arange(time_steps, dtype=torch.float32) * dt
        
        for b in range(batch_size):
            for i in range(input_size):
                for j in range(output_size):
                    # Get spike times
                    pre_times = torch.where(pre_spikes[b, i] > 0)[0] * dt
                    post_times = torch.where(post_spikes[b, j] > 0)[0] * dt
                    
                    if len(pre_times) > 0 and len(post_times) > 0:
                        # Calculate all pairwise spike time differences
                        time_diffs = post_times.unsqueeze(1) - pre_times.unsqueeze(0)
                        
                        # Potentiation: post after pre (Δt > 0)
                        potentiation_mask = time_diffs > 0
                        if potentiation_mask.any():
                            pot_values = self.A_plus * torch.exp(-time_diffs[potentiation_mask] / self.tau_plus)
                            weight_updates[j, i] += pot_values.sum()
                        
                        # Depression: pre after post (Δt < 0)
                        depression_mask = time_diffs < 0
                        if depression_mask.any():
                            dep_values = -self.A_minus * torch.exp(time_diffs[depression_mask] / self.tau_minus)
                            weight_updates[j, i] += dep_values.sum()
        
        # Apply updates and clamp weights
        updated_weights = weights + weight_updates / batch_size
        return torch.clamp(updated_weights, self.w_min, self.w_max)
    
    def _triplet_stdp(
        self, 
        weights: torch.Tensor,
        pre_spikes: torch.Tensor, 
        post_spikes: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """Triplet STDP with three-spike interactions."""
        # Simplified triplet STDP implementation
        # For full implementation, would need to track triplet correlations
        
        # First apply classical STDP
        weights_updated = self._classical_stdp(weights, pre_spikes, post_spikes, dt)
        
        # Add triplet terms (simplified)
        batch_size, input_size, time_steps = pre_spikes.shape
        output_size = post_spikes.size(1)
        
        # Additional parameters for triplets
        A3_plus = 0.001  # Triplet potentiation amplitude
        A3_minus = 0.001  # Triplet depression amplitude
        tau_y = 114.0  # ms
        tau_x = 101.0  # ms
        
        triplet_updates = torch.zeros_like(weights)
        
        for b in range(batch_size):
            # Compute spike-triggered averages for triplet terms
            pre_filtered = self._filter_spikes(pre_spikes[b], tau_x, dt)
            post_filtered = self._filter_spikes(post_spikes[b], tau_y, dt)
            
            # Triplet interactions (simplified calculation)
            for t in range(1, time_steps):
                pre_t = pre_spikes[b, :, t]
                post_t = post_spikes[b, :, t]
                
                if pre_t.sum() > 0 and post_t.sum() > 0:
                    # Add triplet terms
                    for i in range(input_size):
                        for j in range(output_size):
                            if pre_t[i] > 0:
                                triplet_updates[j, i] += A3_plus * post_filtered[j, t-1]
                            if post_t[j] > 0:
                                triplet_updates[j, i] -= A3_minus * pre_filtered[i, t-1]
        
        # Apply triplet updates
        weights_updated += triplet_updates / batch_size
        return torch.clamp(weights_updated, self.w_min, self.w_max)
    
    def _filter_spikes(self, spikes: torch.Tensor, tau: float, dt: float) -> torch.Tensor:
        """Apply exponential filter to spike trains."""
        alpha = torch.exp(torch.tensor(-dt / tau))
        filtered = torch.zeros_like(spikes)
        
        for t in range(1, spikes.size(-1)):
            filtered[:, t] = alpha * filtered[:, t-1] + spikes[:, t]
        
        return filtered


class HomeostaticPlasticity:
    """Homeostatic plasticity mechanisms for maintaining network stability."""
    
    def __init__(
        self,
        target_rate: float = 10.0,  # Hz
        tau_homeostatic: float = 1000.0,  # ms
        eta_intrinsic: float = 0.001,
        eta_synaptic: float = 0.0001,
        scaling_type: str = "multiplicative"
    ):
        """Initialize homeostatic plasticity.
        
        Args:
            target_rate: Target firing rate for each neuron (Hz)
            tau_homeostatic: Time constant for homeostatic adaptation (ms)
            eta_intrinsic: Learning rate for intrinsic excitability
            eta_synaptic: Learning rate for synaptic scaling
            scaling_type: Type of synaptic scaling ('multiplicative', 'additive')
        """
        self.target_rate = target_rate
        self.tau_homeostatic = tau_homeostatic
        self.eta_intrinsic = eta_intrinsic
        self.eta_synaptic = eta_synaptic
        self.scaling_type = scaling_type
        
        # Running average of firing rates
        self.avg_rates = None
        self.alpha = None
        
    def update_homeostasis(
        self,
        weights: torch.Tensor,
        thresholds: torch.Tensor,
        post_spikes: torch.Tensor,
        dt: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply homeostatic plasticity updates.
        
        Args:
            weights: Synaptic weights [output_size, input_size]
            thresholds: Neuron thresholds [output_size]
            post_spikes: Post-synaptic spikes [batch_size, output_size, time_steps]
            dt: Time step (ms)
            
        Returns:
            Updated weights and thresholds
        """
        batch_size, output_size, time_steps = post_spikes.shape
        duration = time_steps * dt / 1000.0  # Convert to seconds
        
        # Calculate current firing rates
        current_rates = post_spikes.sum(dim=-1).float() / duration  # [batch_size, output_size]
        batch_avg_rates = current_rates.mean(dim=0)  # [output_size]
        
        # Initialize or update running average
        if self.avg_rates is None:
            self.avg_rates = batch_avg_rates.clone()
            self.alpha = torch.exp(torch.tensor(-dt / self.tau_homeostatic))
        else:
            self.avg_rates = self.alpha * self.avg_rates + (1 - self.alpha) * batch_avg_rates
        
        # Compute rate deviations
        rate_error = self.avg_rates - self.target_rate
        
        # Intrinsic excitability adjustment
        threshold_updates = self.eta_intrinsic * rate_error
        updated_thresholds = thresholds + threshold_updates
        
        # Synaptic scaling
        if self.scaling_type == "multiplicative":
            # Multiplicative synaptic scaling
            scaling_factors = 1.0 - self.eta_synaptic * rate_error
            scaling_factors = torch.clamp(scaling_factors, 0.5, 2.0)  # Prevent extreme scaling
            
            updated_weights = weights * scaling_factors.unsqueeze(1)
            
        elif self.scaling_type == "additive":
            # Additive synaptic scaling
            weight_updates = -self.eta_synaptic * rate_error.unsqueeze(1).expand(-1, weights.size(1))
            updated_weights = weights + weight_updates
            
        else:
            raise ValueError(f"Unknown scaling type: {self.scaling_type}")
        
        # Ensure weights remain positive
        updated_weights = torch.clamp(updated_weights, 0.0, 2.0)
        
        return updated_weights, updated_thresholds
    
    def get_homeostatic_metrics(self) -> Dict[str, float]:
        """Get current homeostatic metrics."""
        if self.avg_rates is None:
            return {}
        
        return {
            "mean_firing_rate": self.avg_rates.mean().item(),
            "std_firing_rate": self.avg_rates.std().item(),
            "min_firing_rate": self.avg_rates.min().item(),
            "max_firing_rate": self.avg_rates.max().item(),
            "rate_deviation_from_target": (self.avg_rates - self.target_rate).abs().mean().item()
        }


class MetaplasticityRule:
    """Metaplasticity: plasticity of plasticity based on neural activity history."""
    
    def __init__(
        self,
        tau_meta: float = 10000.0,  # ms
        theta_low: float = 5.0,     # Hz
        theta_high: float = 20.0,   # Hz
        meta_rate: float = 0.01
    ):
        """Initialize metaplasticity rule.
        
        Args:
            tau_meta: Time constant for metaplastic changes (ms)
            theta_low: Low activity threshold (Hz)
            theta_high: High activity threshold (Hz) 
            meta_rate: Rate of metaplastic changes
        """
        self.tau_meta = tau_meta
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.meta_rate = meta_rate
        
        # Metaplastic state variables
        self.plasticity_threshold = None
        self.activity_history = None
        
    def update_metaplasticity(
        self,
        current_rates: torch.Tensor,
        stdp_params: Dict[str, float],
        dt: float = 1.0
    ) -> Dict[str, float]:
        """Update metaplastic variables and modify STDP parameters.
        
        Args:
            current_rates: Current firing rates [num_neurons]
            stdp_params: Current STDP parameters
            dt: Time step (ms)
            
        Returns:
            Updated STDP parameters
        """
        num_neurons = current_rates.size(0)
        
        # Initialize metaplastic variables
        if self.plasticity_threshold is None:
            self.plasticity_threshold = torch.ones(num_neurons)
            self.activity_history = current_rates.clone()
        
        # Update activity history with exponential filter
        alpha_meta = torch.exp(torch.tensor(-dt / self.tau_meta))
        self.activity_history = alpha_meta * self.activity_history + (1 - alpha_meta) * current_rates
        
        # Update plasticity threshold based on activity history
        for i in range(num_neurons):
            avg_rate = self.activity_history[i]
            
            if avg_rate < self.theta_low:
                # Low activity: increase plasticity
                self.plasticity_threshold[i] *= (1 - self.meta_rate)
                
            elif avg_rate > self.theta_high:
                # High activity: decrease plasticity  
                self.plasticity_threshold[i] *= (1 + self.meta_rate)
        
        # Clamp plasticity threshold
        self.plasticity_threshold = torch.clamp(self.plasticity_threshold, 0.1, 2.0)
        
        # Modify STDP parameters based on plasticity threshold
        updated_params = stdp_params.copy()
        
        # Scale learning rates by plasticity threshold
        avg_threshold = self.plasticity_threshold.mean().item()
        updated_params["A_plus"] = stdp_params["A_plus"] * avg_threshold
        updated_params["A_minus"] = stdp_params["A_minus"] * avg_threshold
        
        return updated_params
    
    def get_metaplastic_state(self) -> Dict[str, torch.Tensor]:
        """Get current metaplastic state."""
        return {
            "plasticity_threshold": self.plasticity_threshold.clone() if self.plasticity_threshold is not None else None,
            "activity_history": self.activity_history.clone() if self.activity_history is not None else None
        }


class ReinforcementModulatedSTDP:
    """STDP modulated by reward/punishment signals."""
    
    def __init__(
        self,
        base_stdp: STDPLearning,
        tau_dopamine: float = 1000.0,  # ms
        dopamine_modulation: float = 1.0,
        eligibility_trace_decay: float = 0.95
    ):
        """Initialize reinforcement-modulated STDP.
        
        Args:
            base_stdp: Base STDP learning rule
            tau_dopamine: Time constant for dopamine signal (ms)
            dopamine_modulation: Strength of dopaminergic modulation
            eligibility_trace_decay: Decay rate for eligibility traces
        """
        self.base_stdp = base_stdp
        self.tau_dopamine = tau_dopamine
        self.dopamine_modulation = dopamine_modulation
        self.eligibility_trace_decay = eligibility_trace_decay
        
        # Eligibility traces
        self.eligibility_traces = None
        self.dopamine_level = 0.0
        
    def update_weights_with_reward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        reward_signal: float,
        dt: float = 1.0
    ) -> torch.Tensor:
        """Update weights with reward-modulated STDP.
        
        Args:
            weights: Current weights [output_size, input_size]
            pre_spikes: Pre-synaptic spikes [batch_size, input_size, time_steps]
            post_spikes: Post-synaptic spikes [batch_size, output_size, time_steps]
            reward_signal: Reward signal (-1 to +1)
            dt: Time step (ms)
            
        Returns:
            Updated weights
        """
        # Initialize eligibility traces
        if self.eligibility_traces is None:
            self.eligibility_traces = torch.zeros_like(weights)
        
        # Compute standard STDP updates
        stdp_updates = self.base_stdp.update_weights(
            torch.zeros_like(weights), pre_spikes, post_spikes, dt
        )
        
        # Update eligibility traces
        self.eligibility_traces = (
            self.eligibility_trace_decay * self.eligibility_traces + 
            stdp_updates
        )
        
        # Update dopamine level with reward signal
        alpha_da = torch.exp(torch.tensor(-dt / self.tau_dopamine))
        self.dopamine_level = alpha_da * self.dopamine_level + (1 - alpha_da) * reward_signal
        
        # Apply dopamine-modulated weight changes
        modulated_updates = self.dopamine_modulation * self.dopamine_level * self.eligibility_traces
        
        updated_weights = weights + modulated_updates
        return torch.clamp(updated_weights, self.base_stdp.w_min, self.base_stdp.w_max)
    
    def reset_traces(self):
        """Reset eligibility traces."""
        if self.eligibility_traces is not None:
            self.eligibility_traces.zero_()
        self.dopamine_level = 0.0