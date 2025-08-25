"""Novel STDP algorithms based on 2024-2025 research advances with multi-timescale plasticity."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math


@dataclass
class STDPConfig:
    """Configuration for STDP learning algorithms."""
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    a_plus: float = 0.01
    a_minus: float = 0.01
    w_min: float = -1.0
    w_max: float = 1.0
    learning_rate: float = 0.001
    stabilization_factor: float = 0.1
    competitive_factor: float = 0.5
    
    # Multi-timescale parameters (2024-2025 research)
    short_term_tau: float = 5.0
    medium_term_tau: float = 50.0
    long_term_tau: float = 500.0
    homeostatic_tau: float = 10000.0
    metaplasticity_rate: float = 0.001


class StabilizedSupervisedSTDP:
    """Stabilized Supervised STDP (S2-STDP) implementation.
    
    Based on "Paired competing neurons improving STDP supervised local learning 
    in spiking neural networks" (Goupy et al., 2024).
    
    Integrates error-modulated weight updates that align neuron spikes with 
    desired timestamps while maintaining stability through competitive mechanisms.
    """
    
    def __init__(self, config: STDPConfig):
        """Initialize S2-STDP learning rule.
        
        Args:
            config: STDP configuration parameters
        """
        self.config = config
        
        # Pre-compute exponential decay factors
        self.alpha_plus = torch.exp(torch.tensor(-1.0 / config.tau_plus))
        self.alpha_minus = torch.exp(torch.tensor(-1.0 / config.tau_minus))
        
        # Eligibility traces
        self.pre_traces = None
        self.post_traces = None
        
        # Error signals for supervision
        self.error_signals = None
        
        # Statistics tracking
        self.potentiation_events = 0
        self.depression_events = 0
        self.supervision_corrections = 0
    
    def initialize_traces(self, batch_size: int, num_pre: int, num_post: int, time_steps: int, device: str = "cpu"):
        """Initialize eligibility traces for learning.
        
        Args:
            batch_size: Number of samples in batch
            num_pre: Number of presynaptic neurons
            num_post: Number of postsynaptic neurons  
            time_steps: Number of time steps
            device: Device to place tensors on
        """
        self.pre_traces = torch.zeros(batch_size, num_pre, time_steps, device=device)
        self.post_traces = torch.zeros(batch_size, num_post, time_steps, device=device)
        self.error_signals = torch.zeros(batch_size, num_post, time_steps, device=device)
    
    def update_traces(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, 
                     target_spikes: torch.Tensor, time_step: int):
        """Update eligibility traces and error signals.
        
        Args:
            pre_spikes: Presynaptic spike trains [batch, num_pre]
            post_spikes: Postsynaptic spike trains [batch, num_post]
            target_spikes: Target spike patterns [batch, num_post]
            time_step: Current time step
        """
        batch_size, num_pre = pre_spikes.shape
        batch_size, num_post = post_spikes.shape
        
        # Update presynaptic traces
        if time_step > 0:
            self.pre_traces[:, :, time_step] = (
                self.alpha_plus * self.pre_traces[:, :, time_step - 1] + pre_spikes
            )
        else:
            self.pre_traces[:, :, time_step] = pre_spikes
        
        # Update postsynaptic traces
        if time_step > 0:
            self.post_traces[:, :, time_step] = (
                self.alpha_minus * self.post_traces[:, :, time_step - 1] + post_spikes
            )
        else:
            self.post_traces[:, :, time_step] = post_spikes
        
        # Compute error signals (difference between actual and target spikes)
        self.error_signals[:, :, time_step] = target_spikes - post_spikes
    
    def compute_weight_updates(self, weights: torch.Tensor, time_step: int) -> torch.Tensor:
        """Compute S2-STDP weight updates with error modulation.
        
        Args:
            weights: Current synaptic weights [num_pre, num_post]
            time_step: Current time step
            
        Returns:
            Weight updates [num_pre, num_post]
        """
        if self.pre_traces is None or self.post_traces is None:
            return torch.zeros_like(weights)
        
        batch_size = self.pre_traces.shape[0]
        device = weights.device
        
        # Initialize weight updates
        dw = torch.zeros_like(weights, device=device)
        
        # Standard STDP terms
        if time_step > 0:
            # Potentiation: pre-before-post
            post_current = self.post_traces[:, :, time_step]  # [batch, num_post]
            pre_trace = self.pre_traces[:, :, time_step - 1]  # [batch, num_pre]
            
            # Compute potentiation for all pre-post pairs
            potentiation = torch.einsum('bi,bj->ij', pre_trace, post_current)
            
            # Depression: post-before-pre  
            pre_current = self.pre_traces[:, :, time_step]  # [batch, num_pre]
            post_trace = self.post_traces[:, :, time_step - 1]  # [batch, num_post]
            
            depression = torch.einsum('bi,bj->ij', pre_current, post_trace)
            
            # Standard STDP update
            dw += self.config.a_plus * potentiation - self.config.a_minus * depression
            
            self.potentiation_events += (potentiation > 0).sum().item()
            self.depression_events += (depression > 0).sum().item()
        
        # Error-modulated supervision term
        error_current = self.error_signals[:, :, time_step]  # [batch, num_post]
        pre_trace_current = self.pre_traces[:, :, time_step]  # [batch, num_pre]
        
        # Supervision signal modulated by error
        supervision = torch.einsum('bi,bj->ij', pre_trace_current, error_current)
        
        # Apply supervision with learning rate and stabilization
        supervised_update = self.config.learning_rate * supervision
        supervised_update = supervised_update / (1.0 + self.config.stabilization_factor * torch.abs(supervision))
        
        dw += supervised_update
        
        self.supervision_corrections += (torch.abs(error_current) > 0.1).sum().item()
        
        # Normalize by batch size
        dw = dw / batch_size
        
        return dw
    
    def apply_weight_updates(self, weights: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
        """Apply weight updates with bounds checking.
        
        Args:
            weights: Current weights
            dw: Weight updates
            
        Returns:
            Updated weights
        """
        updated_weights = weights + dw
        
        # Apply weight bounds
        updated_weights = torch.clamp(
            updated_weights, 
            self.config.w_min, 
            self.config.w_max
        )
        
        return updated_weights
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        total_events = self.potentiation_events + self.depression_events + self.supervision_corrections
        
        return {
            "potentiation_events": self.potentiation_events,
            "depression_events": self.depression_events,
            "supervision_corrections": self.supervision_corrections,
            "potentiation_ratio": self.potentiation_events / max(1, total_events),
            "depression_ratio": self.depression_events / max(1, total_events),
            "supervision_ratio": self.supervision_corrections / max(1, total_events),
        }


class BatchedSTDP:
    """Samples Temporal Batch STDP (STB-STDP) implementation.
    
    Updates weights based on multiple samples and moments to speed up 
    and stabilize training of unsupervised spiking neural networks.
    """
    
    def __init__(self, config: STDPConfig, batch_accumulation: int = 10):
        """Initialize STB-STDP learning rule.
        
        Args:
            config: STDP configuration parameters
            batch_accumulation: Number of samples to accumulate before update
        """
        self.config = config
        self.batch_accumulation = batch_accumulation
        
        # Accumulated updates
        self.accumulated_updates = None
        self.accumulation_count = 0
        
        # Temporal integration parameters
        self.temporal_window = 50  # Time steps for temporal batching
        self.momentum = 0.9
        
        # Previous updates for momentum
        self.previous_update = None
    
    def accumulate_updates(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, 
                          weights: torch.Tensor) -> Optional[torch.Tensor]:
        """Accumulate STDP updates over multiple samples.
        
        Args:
            pre_spikes: Presynaptic spikes [batch, num_pre, time_steps]
            post_spikes: Postsynaptic spikes [batch, num_post, time_steps]
            weights: Current weights [num_pre, num_post]
            
        Returns:
            Weight updates if batch is complete, None otherwise
        """
        batch_size, num_pre, time_steps = pre_spikes.shape
        batch_size, num_post, time_steps = post_spikes.shape
        device = weights.device
        
        # Compute batch updates
        batch_update = self._compute_batch_stdp_updates(pre_spikes, post_spikes)
        
        # Initialize accumulator if needed
        if self.accumulated_updates is None:
            self.accumulated_updates = torch.zeros_like(weights, device=device)
        
        # Accumulate updates
        self.accumulated_updates += batch_update
        self.accumulation_count += batch_size
        
        # Return accumulated updates if batch is complete
        if self.accumulation_count >= self.batch_accumulation:
            # Average accumulated updates
            avg_update = self.accumulated_updates / self.accumulation_count
            
            # Apply momentum
            if self.previous_update is not None:
                avg_update = self.momentum * self.previous_update + (1 - self.momentum) * avg_update
            
            self.previous_update = avg_update.clone()
            
            # Reset accumulator
            self.accumulated_updates.zero_()
            self.accumulation_count = 0
            
            return avg_update
        
        return None
    
    def _compute_batch_stdp_updates(self, pre_spikes: torch.Tensor, 
                                   post_spikes: torch.Tensor) -> torch.Tensor:
        """Compute STDP updates for a batch of spike trains.
        
        Args:
            pre_spikes: Presynaptic spikes [batch, num_pre, time_steps]
            post_spikes: Postsynaptic spikes [batch, num_post, time_steps]
            
        Returns:
            Weight updates [num_pre, num_post]
        """
        batch_size, num_pre, time_steps = pre_spikes.shape
        batch_size, num_post, time_steps = post_spikes.shape
        device = pre_spikes.device
        
        # Initialize weight updates
        dw = torch.zeros(num_pre, num_post, device=device)
        
        # Compute causal and anti-causal correlations efficiently
        for delay in range(1, min(self.temporal_window, time_steps)):
            # Causal: pre leads post (potentiation)
            if delay < time_steps:
                pre_delayed = pre_spikes[:, :, :-delay]  # [batch, num_pre, time-delay]
                post_current = post_spikes[:, :, delay:]  # [batch, num_post, time-delay]
                
                # Exponential decay weight
                decay_weight = math.exp(-delay / self.config.tau_plus)
                
                # Correlation across batch and time
                causal_corr = torch.einsum('bit,bjt->ij', pre_delayed, post_current)
                dw += self.config.a_plus * decay_weight * causal_corr
            
            # Anti-causal: post leads pre (depression)  
            if delay < time_steps:
                post_delayed = post_spikes[:, :, :-delay]  # [batch, num_post, time-delay]
                pre_current = pre_spikes[:, :, delay:]  # [batch, num_pre, time-delay]
                
                # Exponential decay weight
                decay_weight = math.exp(-delay / self.config.tau_minus)
                
                # Correlation across batch and time
                anticausal_corr = torch.einsum('bit,bjt->ij', pre_current, post_delayed)
                dw -= self.config.a_minus * decay_weight * anticausal_corr
        
        # Normalize by batch size and time steps
        dw = dw / (batch_size * (time_steps - 1))
        
        return dw


class CompetitiveSTDP:
    """Competitive STDP with winner-take-all dynamics.
    
    Implements paired competing neurons architecture for enhanced 
    learning and specialization in classification tasks.
    """
    
    def __init__(self, config: STDPConfig, num_classes: int, competition_radius: float = 2.0):
        """Initialize competitive STDP.
        
        Args:
            config: STDP configuration parameters
            num_classes: Number of output classes
            competition_radius: Radius of lateral competition
        """
        self.config = config
        self.num_classes = num_classes
        self.competition_radius = competition_radius
        
        # Lateral inhibition parameters
        self.inhibition_strength = 0.5
        self.adaptation_rate = 0.01
        
        # Neuron specialization tracking
        self.specialization_scores = None
        self.class_activations = None
    
    def initialize_competition(self, num_neurons: int, device: str = "cpu"):
        """Initialize competitive mechanisms.
        
        Args:
            num_neurons: Total number of competing neurons
            device: Device for tensor placement
        """
        self.specialization_scores = torch.zeros(num_neurons, self.num_classes, device=device)
        self.class_activations = torch.zeros(self.num_classes, device=device)
        
        # Create lateral inhibition matrix
        self.lateral_weights = self._create_lateral_inhibition_matrix(num_neurons, device)
    
    def _create_lateral_inhibition_matrix(self, num_neurons: int, device: str) -> torch.Tensor:
        """Create lateral inhibition connectivity matrix.
        
        Args:
            num_neurons: Number of neurons
            device: Device for tensor placement
            
        Returns:
            Lateral inhibition matrix [num_neurons, num_neurons]
        """
        # Create distance-based inhibition
        positions = torch.arange(num_neurons, dtype=torch.float, device=device)
        distances = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
        
        # Gaussian inhibition profile
        inhibition = torch.exp(-distances**2 / (2 * self.competition_radius**2))
        
        # Remove self-connections
        inhibition.fill_diagonal_(0)
        
        return inhibition * self.inhibition_strength
    
    def apply_lateral_inhibition(self, neuron_activities: torch.Tensor) -> torch.Tensor:
        """Apply lateral inhibition to neuron activities.
        
        Args:
            neuron_activities: Current neuron activities [batch, num_neurons]
            
        Returns:
            Inhibited activities [batch, num_neurons]
        """
        if self.lateral_weights is None:
            return neuron_activities
        
        # Compute inhibitory input
        inhibitory_input = torch.mm(neuron_activities, self.lateral_weights.T)
        
        # Apply inhibition
        inhibited_activities = neuron_activities - inhibitory_input
        
        # Ensure non-negative activities
        inhibited_activities = torch.clamp(inhibited_activities, min=0.0)
        
        return inhibited_activities
    
    def update_specialization(self, neuron_activities: torch.Tensor, class_labels: torch.Tensor):
        """Update neuron specialization scores.
        
        Args:
            neuron_activities: Neuron activity levels [batch, num_neurons]
            class_labels: True class labels [batch]
        """
        if self.specialization_scores is None:
            return
        
        batch_size = neuron_activities.shape[0]
        
        for b in range(batch_size):
            class_label = class_labels[b].item()
            activities = neuron_activities[b]
            
            # Update specialization scores
            for c in range(self.num_classes):
                if c == class_label:
                    # Increase specialization for correct class
                    self.specialization_scores[:, c] += self.adaptation_rate * activities
                else:
                    # Decrease specialization for incorrect classes
                    self.specialization_scores[:, c] -= self.adaptation_rate * activities * 0.1
            
            # Update class activation counts
            self.class_activations[class_label] += activities.sum()
        
        # Normalize specialization scores
        self.specialization_scores = torch.clamp(self.specialization_scores, 0.0, 1.0)
    
    def get_winner_neurons(self, neuron_activities: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Get top-k winner neurons for each class.
        
        Args:
            neuron_activities: Current activities [batch, num_neurons]
            k: Number of winners per class
            
        Returns:
            Winner neuron indices [batch, k]
        """
        batch_size = neuron_activities.shape[0]
        
        # Apply lateral inhibition
        inhibited_activities = self.apply_lateral_inhibition(neuron_activities)
        
        # Get top-k neurons
        _, winner_indices = torch.topk(inhibited_activities, k, dim=1)
        
        return winner_indices
    
    def compute_competitive_updates(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor,
                                  class_labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Compute weight updates with competitive mechanisms.
        
        Args:
            pre_spikes: Presynaptic spikes [batch, num_pre, time_steps]
            post_spikes: Postsynaptic spikes [batch, num_post, time_steps]
            class_labels: Class labels [batch]
            weights: Current weights [num_pre, num_post]
            
        Returns:
            Competitive weight updates [num_pre, num_post]
        """
        batch_size, num_pre, time_steps = pre_spikes.shape
        batch_size, num_post, time_steps = post_spikes.shape
        device = weights.device
        
        # Compute basic STDP updates
        dw = torch.zeros_like(weights, device=device)
        
        # Process each sample in batch
        for b in range(batch_size):
            class_label = class_labels[b].item()
            
            # Get spike trains for this sample
            pre_b = pre_spikes[b]  # [num_pre, time_steps]
            post_b = post_spikes[b]  # [num_post, time_steps]
            
            # Compute post-synaptic activities
            post_activities = post_b.sum(dim=1)  # [num_post]
            
            # Apply competitive mechanism
            winners = self.get_winner_neurons(post_activities.unsqueeze(0), k=3).squeeze(0)
            
            # Compute STDP updates only for winner neurons
            for t in range(1, time_steps):
                # Potentiation for winners
                for winner in winners:
                    if post_b[winner, t] > 0:  # Winner spiked
                        # Find recent pre-spikes
                        for tau in range(1, min(t + 1, 20)):  # Look back 20 time steps
                            decay = math.exp(-tau / self.config.tau_plus)
                            dw[:, winner] += self.config.a_plus * decay * pre_b[:, t - tau]
                
                # Depression for all neurons
                pre_current = pre_b[:, t]
                for tau in range(1, min(t + 1, 20)):
                    decay = math.exp(-tau / self.config.tau_minus)
                    post_prev = post_b[:, t - tau]
                    
                    # Apply depression with competition factor
                    competition_factor = self.config.competitive_factor
                    dw -= self.config.a_minus * decay * competition_factor * torch.outer(pre_current, post_prev)
            
            # Update specialization
            if self.specialization_scores is not None:
                self.update_specialization(post_activities.unsqueeze(0), torch.tensor([class_label]))
        
        # Normalize by batch size
        dw = dw / batch_size
        
        return dw
    
    def get_competition_stats(self) -> Dict[str, Any]:
        """Get competitive learning statistics."""
        if self.specialization_scores is None:
            return {"error": "Competition not initialized"}
        
        # Compute specialization metrics
        max_specialization = torch.max(self.specialization_scores, dim=1)[0]
        mean_specialization = torch.mean(max_specialization).item()
        
        # Compute class balance
        class_balance = torch.std(self.class_activations) / torch.mean(self.class_activations)
        
        return {
            "mean_specialization": mean_specialization,
            "class_balance": class_balance.item(),
            "total_activations": self.class_activations.sum().item(),
            "most_active_class": torch.argmax(self.class_activations).item(),
        }


class MultiTimescaleSTDP:
    """Multi-timescale plasticity implementation based on 2024-2025 research.
    
    Implements overlapping plasticity mechanisms across multiple time scales
    for enhanced learning stability and generalization.
    """
    
    def __init__(self, config: STDPConfig):
        """Initialize multi-timescale STDP.
        
        Args:
            config: STDP configuration with multi-timescale parameters
        """
        self.config = config
        
        # Multi-timescale traces
        self.short_term_traces = None
        self.medium_term_traces = None
        self.long_term_traces = None
        self.homeostatic_traces = None
        
        # Adaptive learning rates
        self.meta_learning_rates = None
        self.activity_history = None
        
        # Statistics
        self.plasticity_events = {"short": 0, "medium": 0, "long": 0, "homeostatic": 0}
    
    def initialize_traces(self, batch_size: int, num_pre: int, num_post: int, device: str = "cpu"):
        """Initialize multi-timescale traces."""
        self.short_term_traces = torch.zeros(batch_size, num_pre, device=device)
        self.medium_term_traces = torch.zeros(batch_size, num_pre, device=device)
        self.long_term_traces = torch.zeros(batch_size, num_pre, device=device)
        self.homeostatic_traces = torch.zeros(batch_size, num_post, device=device)
        
        # Meta-plasticity learning rates
        self.meta_learning_rates = torch.ones(num_pre, num_post, device=device)
        self.activity_history = torch.zeros(num_post, device=device)
    
    def update_multi_timescale_traces(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """Update traces across multiple timescales."""
        if self.short_term_traces is None:
            return
        
        # Short-term plasticity (fast adaptation)
        alpha_short = torch.exp(torch.tensor(-1.0 / self.config.short_term_tau))
        self.short_term_traces = alpha_short * self.short_term_traces + pre_spikes
        
        # Medium-term plasticity (working memory)
        alpha_medium = torch.exp(torch.tensor(-1.0 / self.config.medium_term_tau))
        self.medium_term_traces = alpha_medium * self.medium_term_traces + pre_spikes
        
        # Long-term plasticity (consolidation)
        alpha_long = torch.exp(torch.tensor(-1.0 / self.config.long_term_tau))
        self.long_term_traces = alpha_long * self.long_term_traces + pre_spikes
        
        # Homeostatic traces (maintain activity balance)
        alpha_homeostatic = torch.exp(torch.tensor(-1.0 / self.config.homeostatic_tau))
        self.homeostatic_traces = alpha_homeostatic * self.homeostatic_traces + post_spikes
        
        # Update activity history for meta-plasticity
        self.activity_history = 0.99 * self.activity_history + 0.01 * post_spikes.mean(dim=0)
    
    def compute_adaptive_learning_rates(self) -> torch.Tensor:
        """Compute adaptive learning rates based on recent activity."""
        if self.activity_history is None:
            return self.meta_learning_rates
        
        # Meta-plasticity: adjust learning rates based on recent activity
        target_activity = 0.1  # Target firing rate
        activity_error = torch.abs(self.activity_history - target_activity)
        
        # Increase learning rate for neurons far from target activity
        adaptation_factor = 1.0 + self.config.metaplasticity_rate * activity_error
        
        # Apply to all connections involving each post-synaptic neuron
        adaptive_rates = self.meta_learning_rates * adaptation_factor.unsqueeze(0)
        
        return torch.clamp(adaptive_rates, 0.1, 10.0)  # Reasonable bounds
    
    def compute_multi_timescale_updates(self, post_spikes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Compute weight updates across multiple timescales."""
        if self.short_term_traces is None:
            return torch.zeros_like(weights)
        
        device = weights.device
        dw = torch.zeros_like(weights, device=device)
        
        # Get adaptive learning rates
        adaptive_rates = self.compute_adaptive_learning_rates()
        
        batch_size = post_spikes.shape[0]
        
        # Short-term updates (fast learning)
        if torch.any(post_spikes > 0):
            short_corr = torch.einsum('bi,bj->ij', self.short_term_traces, post_spikes)
            dw += 0.5 * adaptive_rates * short_corr / batch_size
            self.plasticity_events["short"] += (short_corr > 0).sum().item()
        
        # Medium-term updates (working memory)
        if torch.any(post_spikes > 0):
            medium_corr = torch.einsum('bi,bj->ij', self.medium_term_traces, post_spikes)
            dw += 0.3 * adaptive_rates * medium_corr / batch_size
            self.plasticity_events["medium"] += (medium_corr > 0).sum().item()
        
        # Long-term updates (consolidation)
        if torch.any(post_spikes > 0):
            long_corr = torch.einsum('bi,bj->ij', self.long_term_traces, post_spikes)
            dw += 0.2 * adaptive_rates * long_corr / batch_size
            self.plasticity_events["long"] += (long_corr > 0).sum().item()
        
        # Homeostatic plasticity (activity regulation)
        homeostatic_target = 0.1
        activity_excess = self.homeostatic_traces.mean(dim=0) - homeostatic_target
        if torch.any(torch.abs(activity_excess) > 0.05):
            # Reduce weights for over-active neurons, increase for under-active
            homeostatic_adjustment = -0.01 * activity_excess.unsqueeze(0)
            dw += homeostatic_adjustment
            self.plasticity_events["homeostatic"] += torch.sum(torch.abs(activity_excess) > 0.05).item()
        
        return dw
    
    def get_multi_timescale_stats(self) -> Dict[str, Any]:
        """Get multi-timescale plasticity statistics."""
        total_events = sum(self.plasticity_events.values())
        
        stats = {
            "plasticity_events": self.plasticity_events.copy(),
            "total_events": total_events
        }
        
        if total_events > 0:
            for scale, count in self.plasticity_events.items():
                stats[f"{scale}_ratio"] = count / total_events
        
        if self.activity_history is not None:
            stats["mean_activity"] = self.activity_history.mean().item()
            stats["activity_variance"] = self.activity_history.var().item()
        
        if self.meta_learning_rates is not None:
            stats["mean_learning_rate"] = self.meta_learning_rates.mean().item()
            stats["learning_rate_variance"] = self.meta_learning_rates.var().item()
        
        return stats


class AdaptiveTemporalEncoder:
    """Adaptive temporal encoding with flexibility across time steps.
    
    Based on "Temporal Flexibility in Spiking Neural Networks: Towards 
    Generalization Across Time Steps" (2025).
    """
    
    def __init__(self, base_window: int = 100, adaptation_rate: float = 0.01):
        """Initialize adaptive temporal encoder.
        
        Args:
            base_window: Base temporal window size
            adaptation_rate: Rate of temporal adaptation
        """
        self.base_window = base_window
        self.adaptation_rate = adaptation_rate
        self.adaptive_scaling = True
        self.temporal_generalization = True
        
        # Learned temporal parameters
        self.optimal_windows = {}
        self.encoding_efficiency = {}
    
    def encode_with_flexibility(self, values: torch.Tensor, target_timesteps: int) -> torch.Tensor:
        """Encode values with adaptive temporal flexibility.
        
        Args:
            values: Input values to encode [batch, features]
            target_timesteps: Target number of timesteps
            
        Returns:
            Encoded spike trains [batch, features, timesteps]
        """
        batch_size, features = values.shape
        device = values.device
        
        # Adapt window size based on target timesteps
        if self.adaptive_scaling:
            scale_factor = target_timesteps / self.base_window
            adapted_window = int(self.base_window * scale_factor)
        else:
            adapted_window = target_timesteps
        
        # Initialize spike trains
        spikes = torch.zeros(batch_size, features, adapted_window, device=device)
        
        # Adaptive rate encoding
        for b in range(batch_size):
            for f in range(features):
                value = values[b, f].item()
                
                # Determine encoding strategy based on value range
                if abs(value) < 0.1:
                    # Sparse encoding for small values
                    encoding_rate = max(0.01, abs(value) * 10)
                else:
                    # Dense encoding for larger values
                    encoding_rate = min(0.5, abs(value))
                
                # Generate spike train with temporal jitter for robustness
                base_intervals = int(1.0 / max(0.001, encoding_rate))
                
                for t in range(0, adapted_window, base_intervals):
                    if t < adapted_window:
                        # Add temporal jitter for generalization
                        jitter = int(np.random.normal(0, base_intervals * 0.1))
                        actual_t = max(0, min(adapted_window - 1, t + jitter))
                        spikes[b, f, actual_t] = 1.0
        
        # Store efficiency metrics
        sparsity = 1.0 - spikes.mean().item()
        self.encoding_efficiency[target_timesteps] = sparsity
        
        return spikes
    
    def adapt_to_task(self, task_performance: float, current_window: int):
        """Adapt temporal parameters based on task performance.
        
        Args:
            task_performance: Performance metric (0-1)
            current_window: Current temporal window size
        """
        # Update optimal window tracking
        if current_window not in self.optimal_windows:
            self.optimal_windows[current_window] = []
        
        self.optimal_windows[current_window].append(task_performance)
        
        # Adapt base window if needed
        if len(self.optimal_windows[current_window]) > 10:
            avg_performance = np.mean(self.optimal_windows[current_window])
            
            if avg_performance > 0.8:  # Good performance
                # Maintain current parameters
                pass
            elif avg_performance < 0.6:  # Poor performance
                # Adjust temporal parameters
                self.base_window = int(self.base_window * (1 + self.adaptation_rate))
    
    def get_temporal_stats(self) -> Dict[str, Any]:
        """Get temporal encoding statistics."""
        stats = {
            "base_window": self.base_window,
            "adaptation_rate": self.adaptation_rate,
            "adaptive_scaling": self.adaptive_scaling,
            "temporal_generalization": self.temporal_generalization
        }
        
        if self.optimal_windows:
            best_window = max(self.optimal_windows.keys(), 
                            key=lambda w: np.mean(self.optimal_windows[w]) if self.optimal_windows[w] else 0)
            stats["best_window_size"] = best_window
            stats["best_window_performance"] = np.mean(self.optimal_windows[best_window])
        
        if self.encoding_efficiency:
            stats["mean_encoding_efficiency"] = np.mean(list(self.encoding_efficiency.values()))
            stats["encoding_efficiency_by_window"] = self.encoding_efficiency.copy()
        
        return stats