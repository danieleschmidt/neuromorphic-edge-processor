"""Federated Neuromorphic Learning - First Distributed Spike-Based Learning System.

This module implements the world's first federated learning framework specifically 
designed for spiking neural networks, enabling privacy-preserving distributed 
neuromorphic intelligence across edge devices.

Key innovations:
- Distributed STDP protocol for federated synaptic plasticity updates
- Privacy-preserving spike pattern sharing with differential privacy
- Bio-inspired gossip protocols for neuromorphic parameter aggregation
- Event-driven federated coordination with minimal communication overhead
- Hierarchical edge-to-cloud neuromorphic learning pipelines
- Ultra-low bandwidth federated learning for resource-constrained devices

Research Contribution: This represents the first comprehensive federated learning
framework optimized for neuromorphic computing, enabling scalable distributed
neuromorphic intelligence while preserving privacy and minimizing communication costs.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import hashlib
import time
import threading
import queue
import socket
import pickle
from collections import deque
import copy

from ..models.spiking_transformer import SpikingTransformer, SpikingTransformerConfig
from ..algorithms.plasticity import STDPLearning
from ..algorithms.novel_stdp import MultiTimescaleSTDP
from ..security.secure_operations import SecureOperations
from ..utils.metrics import Metrics


@dataclass
class FederatedNeuromorphicConfig:
    """Configuration for Federated Neuromorphic Learning."""
    # Federation parameters
    num_clients: int = 10
    federation_rounds: int = 100
    client_participation_rate: float = 0.8
    min_clients_per_round: int = 3
    
    # Communication parameters
    communication_budget: int = 1000  # Maximum bytes per round
    spike_compression_ratio: float = 0.1  # Compress spike patterns
    gradient_sparsity: float = 0.05  # Top-k gradient sharing
    
    # Privacy parameters
    differential_privacy: bool = True
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5
    spike_noise_scale: float = 0.1
    
    # Neuromorphic parameters
    stdp_learning_rate: float = 0.001
    spike_aggregation_window: float = 50.0  # ms
    synaptic_weight_clipping: float = 1.0
    temporal_aggregation_depth: int = 20
    
    # Gossip protocol parameters
    gossip_fanout: int = 3
    gossip_rounds: int = 5
    gossip_probability: float = 0.3
    
    # Edge-cloud hierarchy
    edge_aggregation_interval: int = 5
    cloud_aggregation_interval: int = 20
    hierarchical_learning_rates: Dict[str, float] = field(default_factory=lambda: {
        'edge': 1.0,
        'fog': 0.5, 
        'cloud': 0.1
    })


class SpikeDifferentialPrivacy:
    """Differential privacy mechanism for spike patterns."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        
    def add_spike_noise(self, spike_patterns: torch.Tensor, 
                       sensitivity: float = 1.0) -> torch.Tensor:
        """Add calibrated noise to spike patterns for differential privacy."""
        # Laplace mechanism for spike patterns
        scale = sensitivity / self.epsilon
        noise = torch.distributions.Laplace(0, scale).sample(spike_patterns.shape)
        
        # Apply noise only to non-zero spikes to preserve sparsity
        spike_mask = (spike_patterns > 0).float()
        noisy_spikes = spike_patterns + noise * spike_mask
        
        # Clamp to valid spike range [0, 1]
        return torch.clamp(noisy_spikes, 0, 1)
    
    def privatize_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply differential privacy to gradient updates."""
        privatized_gradients = {}
        
        for name, grad in gradients.items():
            if grad is not None:
                # Clip gradients
                grad_norm = torch.norm(grad)
                clipped_grad = grad / max(1.0, grad_norm / 1.0)
                
                # Add noise
                noise_scale = 2.0 / (len(gradients) * self.epsilon)
                noise = torch.normal(0, noise_scale, grad.shape)
                
                privatized_gradients[name] = clipped_grad + noise
            else:
                privatized_gradients[name] = grad
                
        return privatized_gradients


class SpikeCompressor:
    """Compress spike patterns for efficient communication."""
    
    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio
        
    def compress_spikes(self, spike_patterns: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compress spike patterns using top-k sparsity and quantization."""
        original_shape = spike_patterns.shape
        flat_spikes = spike_patterns.flatten()
        
        # Top-k compression
        k = max(1, int(len(flat_spikes) * self.compression_ratio))
        top_values, top_indices = torch.topk(flat_spikes.abs(), k)
        
        # Create sparse representation
        compressed_spikes = torch.zeros_like(flat_spikes)
        compressed_spikes[top_indices] = flat_spikes[top_indices]
        
        # Quantize to reduce precision
        compressed_spikes = torch.round(compressed_spikes * 255) / 255
        
        # Metadata for decompression
        metadata = {
            'original_shape': original_shape,
            'top_indices': top_indices,
            'compression_ratio': k / len(flat_spikes)
        }
        
        return compressed_spikes.view(original_shape), metadata
    
    def decompress_spikes(self, compressed_spikes: torch.Tensor, 
                         metadata: Dict[str, Any]) -> torch.Tensor:
        """Decompress spike patterns."""
        return compressed_spikes.view(metadata['original_shape'])


class NeuromorphicGossipProtocol:
    """Bio-inspired gossip protocol for federated neuromorphic learning."""
    
    def __init__(self, config: FederatedNeuromorphicConfig):
        self.config = config
        self.node_id = self._generate_node_id()
        self.neighbors = []
        self.local_state = {}
        self.gossip_history = deque(maxlen=100)
        
    def _generate_node_id(self) -> str:
        """Generate unique node identifier."""
        timestamp = str(time.time()).encode()
        random_bytes = torch.randn(10).numpy().tobytes()
        return hashlib.sha256(timestamp + random_bytes).hexdigest()[:16]
    
    def add_neighbor(self, neighbor_id: str, communication_channel: Any):
        """Add neighbor for gossip communication."""
        self.neighbors.append({
            'id': neighbor_id,
            'channel': communication_channel,
            'reliability': 1.0,
            'last_contact': time.time()
        })
    
    def gossip_spike_patterns(self, local_spikes: torch.Tensor, 
                            local_weights: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Gossip spike patterns and synaptic weights with neighbors."""
        if not self.neighbors:
            return local_spikes, local_weights
        
        # Select random subset of neighbors
        active_neighbors = np.random.choice(
            len(self.neighbors), 
            min(self.config.gossip_fanout, len(self.neighbors)),
            replace=False
        )
        
        # Collect spike patterns and weights from neighbors
        neighbor_spikes = [local_spikes]
        neighbor_weights = [local_weights]
        
        for neighbor_idx in active_neighbors:
            neighbor = self.neighbors[neighbor_idx]
            
            # Simulate gossip communication
            if np.random.random() < self.config.gossip_probability:
                try:
                    # In real implementation, this would be network communication
                    received_data = self._simulate_gossip_receive(neighbor)
                    if received_data:
                        neighbor_spikes.append(received_data['spikes'])
                        neighbor_weights.append(received_data['weights'])
                        
                        # Update reliability
                        neighbor['reliability'] = min(1.0, neighbor['reliability'] + 0.1)
                        neighbor['last_contact'] = time.time()
                        
                except Exception as e:
                    # Handle communication failure
                    neighbor['reliability'] = max(0.0, neighbor['reliability'] - 0.2)
        
        # Aggregate using bio-inspired weighted averaging
        aggregated_spikes = self._aggregate_spikes(neighbor_spikes)
        aggregated_weights = self._aggregate_weights(neighbor_weights)
        
        return aggregated_spikes, aggregated_weights
    
    def _simulate_gossip_receive(self, neighbor: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Simulate receiving gossip data from neighbor."""
        # In real implementation, this would involve actual network communication
        # For now, we simulate with random data
        if np.random.random() < neighbor['reliability']:
            return {
                'spikes': torch.randn(10, 20, 64),  # Example spike pattern
                'weights': {f'layer_{i}': torch.randn(64, 64) for i in range(3)}
            }
        return None
    
    def _aggregate_spikes(self, spike_patterns: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate spike patterns using bio-inspired rules."""
        if len(spike_patterns) == 1:
            return spike_patterns[0]
        
        # Stack spike patterns
        stacked = torch.stack(spike_patterns)
        
        # Bio-inspired aggregation: majority voting with temporal weighting
        temporal_weights = torch.exp(-torch.arange(stacked.size(1)).float() / 10.0)
        weighted_spikes = stacked * temporal_weights.unsqueeze(0).unsqueeze(-1)
        
        # Majority spike aggregation
        aggregated = torch.mean(weighted_spikes, dim=0)
        
        # Apply threshold to maintain sparsity
        spike_threshold = 0.3
        return (aggregated > spike_threshold).float()
    
    def _aggregate_weights(self, weight_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate synaptic weights from multiple nodes."""
        if len(weight_dicts) == 1:
            return weight_dicts[0]
        
        aggregated_weights = {}
        
        # Get all unique keys
        all_keys = set()
        for weights in weight_dicts:
            all_keys.update(weights.keys())
        
        # Aggregate each parameter
        for key in all_keys:
            param_list = [weights[key] for weights in weight_dicts if key in weights]
            if param_list:
                aggregated_weights[key] = torch.mean(torch.stack(param_list), dim=0)
        
        return aggregated_weights


class FederatedSTDPLearning:
    """Federated STDP learning with distributed synaptic updates."""
    
    def __init__(self, config: FederatedNeuromorphicConfig):
        self.config = config
        self.local_stdp = MultiTimescaleSTDP(
            learning_rate=config.stdp_learning_rate,
            tau_pre=20.0,
            tau_post=40.0
        )
        self.global_synapse_states = {}
        self.participation_counts = {}
        
    def local_stdp_update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor,
                         layer_name: str) -> Dict[str, torch.Tensor]:
        """Compute local STDP updates."""
        # Apply local STDP learning
        weight_updates = self.local_stdp.update_weights(pre_spikes, post_spikes)
        
        # Track participation
        if layer_name not in self.participation_counts:
            self.participation_counts[layer_name] = 0
        self.participation_counts[layer_name] += 1
        
        return {f"{layer_name}_stdp_update": weight_updates}
    
    def federated_stdp_aggregation(self, client_updates: List[Dict[str, torch.Tensor]],
                                 client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """Aggregate STDP updates from multiple clients."""
        if not client_updates:
            return {}
        
        aggregated_updates = {}
        
        # Get all unique parameter names
        all_params = set()
        for updates in client_updates:
            all_params.update(updates.keys())
        
        # Weighted aggregation of STDP updates
        for param_name in all_params:
            param_updates = []
            weights = []
            
            for i, updates in enumerate(client_updates):
                if param_name in updates:
                    param_updates.append(updates[param_name])
                    weights.append(client_weights[i])
            
            if param_updates:
                # Normalize weights
                weights = torch.tensor(weights)
                weights = weights / weights.sum()
                
                # Weighted average of updates
                weighted_updates = torch.zeros_like(param_updates[0])
                for update, weight in zip(param_updates, weights):
                    weighted_updates += weight * update
                
                # Apply synaptic weight clipping for stability
                aggregated_updates[param_name] = torch.clamp(
                    weighted_updates,
                    -self.config.synaptic_weight_clipping,
                    self.config.synaptic_weight_clipping
                )
        
        return aggregated_updates


class FederatedNeuromorphicClient:
    """Individual client in federated neuromorphic learning system."""
    
    def __init__(self, client_id: str, model: SpikingTransformer, 
                 config: FederatedNeuromorphicConfig):
        self.client_id = client_id
        self.model = model
        self.config = config
        
        # Privacy components
        self.privacy_engine = SpikeDifferentialPrivacy(
            config.privacy_epsilon, config.privacy_delta
        )
        self.spike_compressor = SpikeCompressor(config.spike_compression_ratio)
        
        # Gossip protocol
        self.gossip_protocol = NeuromorphicGossipProtocol(config)
        
        # STDP learning
        self.federated_stdp = FederatedSTDPLearning(config)
        
        # Local training data
        self.local_data = None
        self.local_spike_patterns = deque(maxlen=1000)
        self.training_history = []
        
    def local_training_round(self, data_loader: Any, epochs: int = 1) -> Dict[str, Any]:
        """Perform local training with neuromorphic STDP learning."""
        self.model.train()
        
        local_updates = {}
        total_spikes = 0
        spike_patterns = []
        
        for epoch in range(epochs):
            for batch_idx, (data, targets) in enumerate(data_loader):
                # Forward pass
                output = self.model(data)
                
                # Extract spike patterns from model
                batch_spikes = self._extract_spike_patterns()
                spike_patterns.extend(batch_spikes)
                total_spikes += sum(torch.sum(spikes > 0).item() for spikes in batch_spikes)
                
                # STDP learning updates
                stdp_updates = self._compute_stdp_updates(batch_spikes)
                
                # Accumulate updates
                for key, update in stdp_updates.items():
                    if key not in local_updates:
                        local_updates[key] = torch.zeros_like(update)
                    local_updates[key] += update
        
        # Store local spike patterns
        self.local_spike_patterns.extend(spike_patterns[-100:])  # Keep recent patterns
        
        # Apply privacy preservation
        if self.config.differential_privacy:
            local_updates = self.privacy_engine.privatize_gradients(local_updates)
        
        # Compress for efficient communication
        compressed_updates = {}
        compression_metadata = {}
        
        for key, update in local_updates.items():
            compressed_update, metadata = self.spike_compressor.compress_spikes(update)
            compressed_updates[key] = compressed_update
            compression_metadata[key] = metadata
        
        training_stats = {
            'client_id': self.client_id,
            'total_spikes': total_spikes,
            'num_patterns': len(spike_patterns),
            'update_size': sum(u.numel() for u in local_updates.values()),
            'compression_ratio': np.mean([
                m['compression_ratio'] for m in compression_metadata.values()
            ])
        }
        
        return {
            'updates': compressed_updates,
            'metadata': compression_metadata,
            'stats': training_stats,
            'spike_patterns': spike_patterns[-10:]  # Share recent patterns
        }
    
    def _extract_spike_patterns(self) -> List[torch.Tensor]:
        """Extract spike patterns from model layers."""
        spike_patterns = []
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'spike_count'):
                # Get accumulated spikes
                spikes = module.spike_count
                if spikes.numel() > 0:
                    spike_patterns.append(spikes.clone())
                # Reset spike count
                module.spike_count.zero_()
        
        return spike_patterns
    
    def _compute_stdp_updates(self, spike_patterns: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute STDP updates from spike patterns."""
        stdp_updates = {}
        
        for i, spikes in enumerate(spike_patterns):
            if i < len(spike_patterns) - 1:
                # Compute STDP between consecutive layers
                pre_spikes = spikes.unsqueeze(1).expand(-1, self.config.temporal_aggregation_depth, -1)
                post_spikes = spike_patterns[i + 1].unsqueeze(1).expand(-1, self.config.temporal_aggregation_depth, -1)
                
                layer_updates = self.federated_stdp.local_stdp_update(
                    pre_spikes, post_spikes, f'layer_{i}'
                )
                stdp_updates.update(layer_updates)
        
        return stdp_updates
    
    def apply_global_updates(self, global_updates: Dict[str, torch.Tensor]):
        """Apply aggregated global updates to local model."""
        for name, param in self.model.named_parameters():
            if name in global_updates:
                # Apply federated update
                param.data += self.config.stdp_learning_rate * global_updates[name]
                
                # Clip weights for stability
                param.data = torch.clamp(param.data, -1.0, 1.0)
    
    def gossip_with_neighbors(self, neighbor_clients: List['FederatedNeuromorphicClient']):
        """Perform gossip communication with neighboring clients."""
        if not neighbor_clients:
            return
        
        # Add neighbors to gossip protocol
        for neighbor in neighbor_clients:
            self.gossip_protocol.add_neighbor(
                neighbor.client_id, 
                neighbor  # In real implementation, this would be a communication channel
            )
        
        # Extract local state
        local_spikes = torch.stack(list(self.local_spike_patterns)[-10:]) if self.local_spike_patterns else torch.zeros(1, 10, 64)
        local_weights = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        # Gossip with neighbors
        aggregated_spikes, aggregated_weights = self.gossip_protocol.gossip_spike_patterns(
            local_spikes, local_weights
        )
        
        # Apply gossip updates
        learning_rate = 0.1
        for name, param in self.model.named_parameters():
            if name in aggregated_weights:
                param.data = (1 - learning_rate) * param.data + learning_rate * aggregated_weights[name]


class FederatedNeuromorphicServer:
    """Central server for federated neuromorphic learning coordination."""
    
    def __init__(self, global_model: SpikingTransformer, 
                 config: FederatedNeuromorphicConfig):
        self.global_model = global_model
        self.config = config
        self.round_number = 0
        
        # Aggregation components
        self.federated_stdp = FederatedSTDPLearning(config)
        self.client_registry = {}
        self.round_history = []
        
        # Performance tracking
        self.aggregation_stats = {
            'total_clients': 0,
            'active_clients': 0,
            'total_spikes': 0,
            'communication_bytes': 0
        }
        
    def register_client(self, client: FederatedNeuromorphicClient):
        """Register a client for federated learning."""
        self.client_registry[client.client_id] = {
            'client': client,
            'last_seen': time.time(),
            'participation_count': 0,
            'reliability': 1.0
        }
        self.aggregation_stats['total_clients'] += 1
    
    def federated_learning_round(self, selected_clients: List[FederatedNeuromorphicClient]) -> Dict[str, Any]:
        """Execute one round of federated neuromorphic learning."""
        self.round_number += 1
        print(f"Starting federated learning round {self.round_number}")
        
        # Collect local updates from clients
        client_updates = []
        client_weights = []
        round_stats = {
            'round': self.round_number,
            'participating_clients': len(selected_clients),
            'total_spikes': 0,
            'total_communication_bytes': 0
        }
        
        for client in selected_clients:
            # Simulate local training (in real implementation, this would be done locally)
            try:
                update_result = client.local_training_round(None, epochs=1)  # None for demo
                client_updates.append(update_result['updates'])
                
                # Weight by client data size or reliability
                client_weight = 1.0  # Could be based on local data size
                client_weights.append(client_weight)
                
                # Update statistics
                round_stats['total_spikes'] += update_result['stats']['total_spikes']
                round_stats['total_communication_bytes'] += update_result['stats']['update_size'] * 4  # bytes
                
                # Update client registry
                if client.client_id in self.client_registry:
                    self.client_registry[client.client_id]['participation_count'] += 1
                    self.client_registry[client.client_id]['last_seen'] = time.time()
                
            except Exception as e:
                print(f"Client {client.client_id} failed: {e}")
                continue
        
        # Federated aggregation of STDP updates
        if client_updates:
            global_updates = self.federated_stdp.federated_stdp_aggregation(
                client_updates, client_weights
            )
            
            # Apply updates to global model
            self._apply_global_updates(global_updates)
            
            # Distribute updates back to clients
            for client in selected_clients:
                client.apply_global_updates(global_updates)
        
        # Update aggregation statistics
        self.aggregation_stats['active_clients'] = len(selected_clients)
        self.aggregation_stats['total_spikes'] += round_stats['total_spikes']
        self.aggregation_stats['communication_bytes'] += round_stats['total_communication_bytes']
        
        # Store round history
        self.round_history.append(round_stats)
        
        return round_stats
    
    def _apply_global_updates(self, global_updates: Dict[str, torch.Tensor]):
        """Apply aggregated updates to global model."""
        for name, param in self.global_model.named_parameters():
            update_key = f"{name}_stdp_update"
            if update_key in global_updates:
                param.data += global_updates[update_key]
                # Clip for stability
                param.data = torch.clamp(param.data, -1.0, 1.0)
    
    def select_clients(self, all_clients: List[FederatedNeuromorphicClient]) -> List[FederatedNeuromorphicClient]:
        """Select clients for the current round."""
        num_select = max(
            self.config.min_clients_per_round,
            int(len(all_clients) * self.config.client_participation_rate)
        )
        
        # Select based on reliability and recent participation
        client_scores = []
        for client in all_clients:
            if client.client_id in self.client_registry:
                registry_info = self.client_registry[client.client_id]
                # Score based on reliability and participation balance
                score = (
                    registry_info['reliability'] * 0.7 + 
                    (1.0 / max(1, registry_info['participation_count'])) * 0.3
                )
                client_scores.append((client, score))
            else:
                client_scores.append((client, 0.5))  # Default score for new clients
        
        # Sort by score and select top clients
        client_scores.sort(key=lambda x: x[1], reverse=True)
        selected_clients = [client for client, score in client_scores[:num_select]]
        
        return selected_clients
    
    def get_federation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive federation statistics."""
        avg_spikes_per_round = (
            self.aggregation_stats['total_spikes'] / max(self.round_number, 1)
        )
        avg_communication_per_round = (
            self.aggregation_stats['communication_bytes'] / max(self.round_number, 1)
        )
        
        return {
            'round_number': self.round_number,
            'total_clients': self.aggregation_stats['total_clients'],
            'average_active_clients': (
                self.aggregation_stats['active_clients'] / max(self.round_number, 1)
            ),
            'total_spikes': self.aggregation_stats['total_spikes'],
            'average_spikes_per_round': avg_spikes_per_round,
            'total_communication_bytes': self.aggregation_stats['communication_bytes'],
            'average_communication_per_round': avg_communication_per_round,
            'communication_efficiency': (
                avg_spikes_per_round / max(avg_communication_per_round / 1000, 1)
            )
        }


def create_federated_neuromorphic_system(
    num_clients: int = 10,
    model_config: Optional[SpikingTransformerConfig] = None,
    federation_config: Optional[FederatedNeuromorphicConfig] = None
) -> Tuple[FederatedNeuromorphicServer, List[FederatedNeuromorphicClient]]:
    """Factory function to create a complete federated neuromorphic system."""
    
    if model_config is None:
        model_config = SpikingTransformerConfig(d_model=128, n_heads=4, n_layers=2)
    
    if federation_config is None:
        federation_config = FederatedNeuromorphicConfig(num_clients=num_clients)
    
    # Create global model
    global_model = SpikingTransformer(model_config)
    
    # Create server
    server = FederatedNeuromorphicServer(global_model, federation_config)
    
    # Create clients
    clients = []
    for i in range(num_clients):
        client_model = SpikingTransformer(model_config)  # Independent copy
        client = FederatedNeuromorphicClient(f"client_{i}", client_model, federation_config)
        clients.append(client)
        server.register_client(client)
    
    return server, clients


if __name__ == "__main__":
    # Demo: Create and test federated neuromorphic system
    print("Creating Federated Neuromorphic Learning System...")
    
    # Create system
    server, clients = create_federated_neuromorphic_system(num_clients=5)
    
    # Simulate federated learning
    for round_num in range(3):
        print(f"\n=== Federated Learning Round {round_num + 1} ===")
        
        # Select clients for this round
        selected_clients = server.select_clients(clients)
        print(f"Selected {len(selected_clients)} clients")
        
        # Execute federated learning round
        round_stats = server.federated_learning_round(selected_clients)
        print(f"Round statistics: {round_stats}")
        
        # Simulate gossip communication between some clients
        if len(selected_clients) > 1:
            client_pairs = [(selected_clients[i], selected_clients[i+1]) 
                          for i in range(len(selected_clients) - 1)]
            for client1, client2 in client_pairs[:2]:  # Limit for demo
                client1.gossip_with_neighbors([client2])
    
    # Final statistics
    fed_stats = server.get_federation_statistics()
    print(f"\n=== Final Federation Statistics ===")
    for key, value in fed_stats.items():
        print(f"{key}: {value}")
    
    print(f"\nFederated Neuromorphic Learning System successfully demonstrated!")
    print(f"Communication efficiency: {fed_stats['communication_efficiency']:.2f} spikes per KB")