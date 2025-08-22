"""Novel STDP algorithms based on 2024-2025 research advances."""

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