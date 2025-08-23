"""Synaptic plasticity algorithms for learning in spiking neural networks."""

import numpy as np
from typing import Dict, List, Optional, Tuple


class STDPLearning:
    """Spike-Timing Dependent Plasticity (STDP) learning rule.
    
    This implementation provides a computationally efficient STDP algorithm
    optimized for edge computing with minimal memory overhead.
    """
    
    def __init__(
        self,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        A_pre: float = 0.01,
        A_post: float = 0.01,
        w_min: float = 0.0,
        w_max: float = 1.0
    ):
        """Initialize STDP learning parameters.
        
        Args:
            tau_pre: Pre-synaptic time constant (ms)
            tau_post: Post-synaptic time constant (ms)
            A_pre: Pre-synaptic learning rate
            A_post: Post-synaptic learning rate
            w_min: Minimum weight value
            w_max: Maximum weight value
        """
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.A_pre = A_pre
        self.A_post = A_post
        self.w_min = w_min
        self.w_max = w_max
        
        # Exponential decay factors
        self.exp_pre = np.exp(-1.0 / tau_pre)
        self.exp_post = np.exp(-1.0 / tau_post)
    
    def update_weights(
        self,
        weights: np.ndarray,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        dt: float = 1.0
    ) -> np.ndarray:
        """Update synaptic weights using STDP rule.
        
        Args:
            weights: Current weight matrix [pre_neurons, post_neurons]
            pre_spikes: Pre-synaptic spikes [pre_neurons, time_steps]
            post_spikes: Post-synaptic spikes [post_neurons, time_steps]
            dt: Time step in ms
            
        Returns:
            Updated weight matrix [pre_neurons, post_neurons]
        """
        pre_neurons, time_steps = pre_spikes.shape
        post_neurons = post_spikes.shape[0]
        
        # Initialize traces
        pre_trace = np.zeros(pre_neurons)
        post_trace = np.zeros(post_neurons)
        
        # Initialize weight updates
        weight_updates = np.zeros_like(weights)
        
        # Process each time step
        for t in range(time_steps):
            # Decay traces
            pre_trace *= self.exp_pre
            post_trace *= self.exp_post
            
            # Update traces with current spikes
            pre_trace += pre_spikes[:, t]
            post_trace += post_spikes[:, t]
            
            # Apply STDP updates
            for i in range(pre_neurons):
                for j in range(post_neurons):
                    # Depression: pre spike, existing post trace
                    if pre_spikes[i, t] > 0:
                        weight_updates[i, j] -= self.A_pre * post_trace[j]
                    
                    # Potentiation: post spike, existing pre trace
                    if post_spikes[j, t] > 0:
                        weight_updates[i, j] += self.A_post * pre_trace[i]
        
        # Apply updates and clip to bounds
        updated_weights = weights + weight_updates
        updated_weights = np.clip(updated_weights, self.w_min, self.w_max)
        
        return updated_weights


class HomeostaticPlasticity:
    """Homeostatic plasticity for maintaining stable network activity."""
    
    def __init__(
        self,
        target_rate: float = 10.0,
        learning_rate: float = 0.001,
        time_constant: float = 1000.0
    ):
        """Initialize homeostatic plasticity.
        
        Args:
            target_rate: Target firing rate (Hz)
            learning_rate: Homeostatic learning rate
            time_constant: Time constant for rate estimation (ms)
        """
        self.target_rate = target_rate
        self.learning_rate = learning_rate
        self.time_constant = time_constant
        self.decay_factor = np.exp(-1.0 / time_constant)
        
        # State variables
        self.estimated_rates = None
    
    def update_thresholds(
        self,
        thresholds: np.ndarray,
        spikes: np.ndarray,
        dt: float = 1.0
    ) -> np.ndarray:
        """Update neuron thresholds based on activity.
        
        Args:
            thresholds: Current firing thresholds [num_neurons]
            spikes: Recent spike activity [num_neurons, time_steps]
            dt: Time step in ms
            
        Returns:
            Updated thresholds [num_neurons]
        """
        num_neurons, time_steps = spikes.shape
        
        # Initialize rate estimates if needed
        if self.estimated_rates is None:
            self.estimated_rates = np.zeros(num_neurons)
        
        # Update running average of firing rates
        recent_rates = spikes.sum(axis=1) * 1000.0 / (time_steps * dt)
        
        self.estimated_rates = (
            self.decay_factor * self.estimated_rates +
            (1 - self.decay_factor) * recent_rates
        )
        
        # Adjust thresholds based on rate error
        rate_error = self.estimated_rates - self.target_rate
        threshold_updates = self.learning_rate * rate_error
        
        updated_thresholds = thresholds + threshold_updates
        
        # Keep thresholds positive
        updated_thresholds = np.maximum(updated_thresholds, 0.1)
        
        return updated_thresholds