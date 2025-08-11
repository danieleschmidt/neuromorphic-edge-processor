"""Advanced research framework for neuromorphic computing experiments."""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

from ..models.spiking_neural_network import SpikingNeuralNetwork
from ..models.liquid_state_machine import LiquidStateMachine
from ..models.reservoir_computer import ReservoirComputer
from ..algorithms.novel_stdp import StabilizedSupervisedSTDP, BatchedSTDP, CompetitiveSTDP
from ..algorithms.spike_processor import SpikeProcessor
from ..utils.metrics import Metrics
from ..monitoring.advanced_monitor import AdvancedMonitor


@dataclass
class ExperimentConfig:
    """Configuration for neuromorphic experiments."""
    experiment_name: str
    description: str
    dataset: str
    model_type: str
    learning_rule: str
    
    # Model parameters
    input_size: int = 784
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])
    output_size: int = 10
    
    # Training parameters
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    sequence_length: int = 100
    
    # Evaluation parameters
    test_batch_size: int = 100
    num_trials: int = 5
    validation_split: float = 0.2
    
    # Experiment parameters
    save_results: bool = True
    plot_results: bool = True
    device: str = "auto"
    seed: int = 42
    
    # Research-specific parameters
    ablation_studies: List[str] = field(default_factory=list)
    hyperparameter_ranges: Dict[str, List] = field(default_factory=dict)
    comparison_baselines: List[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    
    # Performance metrics
    train_accuracy: List[float] = field(default_factory=list)
    test_accuracy: List[float] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    
    # Neuromorphic-specific metrics
    spike_rates: List[float] = field(default_factory=list)
    sparsity: List[float] = field(default_factory=list)
    energy_consumption: List[float] = field(default_factory=list)
    
    # Timing metrics
    training_time: float = 0.0
    inference_time: float = 0.0
    epoch_times: List[float] = field(default_factory=list)
    
    # Additional metrics
    convergence_epoch: int = -1
    final_weights_stats: Dict[str, float] = field(default_factory=dict)
    learning_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    device_used: str = ""
    total_parameters: int = 0


class SyntheticDataGenerator:
    """Generate synthetic datasets for neuromorphic research."""
    
    @staticmethod
    def generate_spike_pattern_dataset(
        num_samples: int = 1000,
        input_size: int = 784,
        sequence_length: int = 100,
        num_classes: int = 10,
        spike_rate: float = 0.1,
        pattern_complexity: float = 0.5,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic spike pattern classification dataset.
        
        Args:
            num_samples: Number of samples to generate
            input_size: Input dimensionality
            sequence_length: Length of spike sequences
            num_classes: Number of output classes
            spike_rate: Base firing rate
            pattern_complexity: Complexity of temporal patterns
            device: Device for tensor placement
            
        Returns:
            Tuple of (data, labels)
        """
        torch.manual_seed(42)
        
        # Generate base patterns for each class
        class_patterns = []
        for c in range(num_classes):
            # Create unique temporal pattern for each class
            pattern = torch.zeros(input_size, sequence_length)
            
            # Add structured spikes
            num_active_neurons = int(input_size * spike_rate * (1 + pattern_complexity))
            active_neurons = torch.randperm(input_size)[:num_active_neurons]
            
            for neuron in active_neurons:
                # Create temporal pattern with class-specific timing
                base_freq = 0.05 + 0.1 * c / num_classes
                phase = torch.rand(1) * 2 * np.pi
                
                for t in range(sequence_length):
                    spike_prob = base_freq * (1 + 0.5 * np.sin(2 * np.pi * t / 20 + phase))
                    if torch.rand(1) < spike_prob:
                        pattern[neuron, t] = 1.0
            
            class_patterns.append(pattern)
        
        # Generate dataset
        data = []
        labels = []
        
        samples_per_class = num_samples // num_classes
        
        for c in range(num_classes):
            base_pattern = class_patterns[c]
            
            for _ in range(samples_per_class):
                # Add noise to base pattern
                noisy_pattern = base_pattern.clone()
                
                # Random noise spikes
                noise_mask = torch.rand_like(base_pattern) < 0.05
                noisy_pattern[noise_mask] = 1.0
                
                # Random spike deletions  
                deletion_mask = torch.rand_like(base_pattern) < 0.1
                noisy_pattern[deletion_mask & (base_pattern == 1)] = 0.0
                
                data.append(noisy_pattern)
                labels.append(c)
        
        # Convert to tensors
        data_tensor = torch.stack(data).to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        
        # Shuffle dataset
        indices = torch.randperm(len(data_tensor))
        data_tensor = data_tensor[indices]
        labels_tensor = labels_tensor[indices]
        
        return data_tensor, labels_tensor
    
    @staticmethod
    def generate_temporal_xor_dataset(
        num_samples: int = 1000,
        sequence_length: int = 50,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate temporal XOR dataset for testing temporal learning.
        
        Args:
            num_samples: Number of samples
            sequence_length: Length of sequences
            device: Device for tensor placement
            
        Returns:
            Tuple of (data, labels)
        """
        torch.manual_seed(42)
        
        data = []
        labels = []
        
        for _ in range(num_samples):
            # Generate two input channels
            seq = torch.zeros(2, sequence_length)
            
            # Random timing for first input
            t1 = torch.randint(5, 25, (1,)).item()
            seq[0, t1] = 1.0
            
            # Random timing for second input
            t2 = torch.randint(25, 45, (1,)).item()
            seq[1, t2] = 1.0
            
            # XOR logic based on relative timing
            if abs(t1 - t2) > 15:
                label = 1  # Far apart = XOR true
            else:
                label = 0  # Close together = XOR false
            
            data.append(seq)
            labels.append(label)
        
        data_tensor = torch.stack(data).to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        
        return data_tensor, labels_tensor


class BaselineModels:
    """Collection of baseline models for comparison."""
    
    @staticmethod
    def create_dense_mlp(input_size: int, hidden_sizes: List[int], 
                        output_size: int, device: str) -> nn.Module:
        """Create dense MLP baseline."""
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        model = nn.Sequential(*layers).to(device)
        return model
    
    @staticmethod
    def create_lstm_baseline(input_size: int, hidden_size: int, 
                           num_layers: int, output_size: int, device: str) -> nn.Module:
        """Create LSTM baseline for temporal data."""
        class LSTMBaseline(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.classifier = nn.Linear(hidden_size, output_size)
                
            def forward(self, x):
                # x: [batch, input_size, seq_len] -> [batch, seq_len, input_size]
                x = x.transpose(1, 2)
                lstm_out, _ = self.lstm(x)
                # Use last output
                return self.classifier(lstm_out[:, -1, :])
        
        return LSTMBaseline().to(device)
    
    @staticmethod
    def create_conv1d_baseline(input_size: int, output_size: int, device: str) -> nn.Module:
        """Create 1D CNN baseline for temporal data."""
        class Conv1DBaseline(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv1d(input_size, 64, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.AdaptiveAvgPool1d(1)
                )
                self.classifier = nn.Linear(128, output_size)
                
            def forward(self, x):
                # x: [batch, input_size, seq_len]
                x = self.conv_layers(x)
                x = x.squeeze(-1)
                return self.classifier(x)
        
        return Conv1DBaseline().to(device)


class ResearchFramework:
    """Main research framework for conducting neuromorphic experiments."""
    
    def __init__(self, experiment_config: ExperimentConfig):
        """Initialize research framework.
        
        Args:
            experiment_config: Experiment configuration
        """
        self.config = experiment_config
        
        # Setup device
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize components
        self.monitor = AdvancedMonitor() if self.device == "cuda" else None
        self.spike_processor = SpikeProcessor()
        self.metrics = Metrics()
        
        # Results storage
        self.results: List[ExperimentResult] = []
        self.baseline_results: Dict[str, ExperimentResult] = {}
        
        # Data generators
        self.data_generator = SyntheticDataGenerator()
        self.baseline_models = BaselineModels()
        
        self.logger.info(f"Research framework initialized on device: {self.device}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup experiment logger."""
        logger = logging.getLogger(f'neuromorphic_research_{self.config.experiment_name}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_model(self, model_type: str, **kwargs) -> nn.Module:
        """Create neuromorphic model based on type.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional model parameters
            
        Returns:
            Created model
        """
        if model_type == "spiking_neural_network":
            layer_sizes = [self.config.input_size] + self.config.hidden_sizes + [self.config.output_size]
            model = SpikingNeuralNetwork(
                layer_sizes=layer_sizes,
                learning_rule=self.config.learning_rule,
                **kwargs
            )
        
        elif model_type == "liquid_state_machine":
            model = LiquidStateMachine(
                input_size=self.config.input_size,
                reservoir_size=self.config.hidden_sizes[0] if self.config.hidden_sizes else 100,
                output_size=self.config.output_size,
                **kwargs
            )
        
        elif model_type == "reservoir_computer":
            model = ReservoirComputer(
                input_size=self.config.input_size,
                reservoir_size=self.config.hidden_sizes[0] if self.config.hidden_sizes else 100,
                output_size=self.config.output_size,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device)
    
    def generate_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate or load dataset for experiments.
        
        Returns:
            Tuple of (train_data, train_labels, test_data, test_labels)
        """
        if self.config.dataset == "spike_patterns":
            data, labels = self.data_generator.generate_spike_pattern_dataset(
                num_samples=2000,
                input_size=self.config.input_size,
                sequence_length=self.config.sequence_length,
                device=self.device
            )
        
        elif self.config.dataset == "temporal_xor":
            data, labels = self.data_generator.generate_temporal_xor_dataset(
                num_samples=1000,
                sequence_length=self.config.sequence_length,
                device=self.device
            )
        
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
        
        # Split into train/test
        num_samples = len(data)
        num_train = int(num_samples * (1 - self.config.validation_split))
        
        indices = torch.randperm(num_samples)
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]
        
        train_data = data[train_indices]
        train_labels = labels[train_indices]
        test_data = data[test_indices]
        test_labels = labels[test_indices]
        
        self.logger.info(f"Dataset loaded: {len(train_data)} train, {len(test_data)} test samples")
        
        return train_data, train_labels, test_data, test_labels
    
    def train_model(self, model: nn.Module, train_data: torch.Tensor, 
                   train_labels: torch.Tensor) -> ExperimentResult:
        """Train neuromorphic model and collect metrics.
        
        Args:
            model: Model to train
            train_data: Training data
            train_labels: Training labels
            
        Returns:
            Experiment result
        """
        result = ExperimentResult(config=self.config, device_used=self.device)
        result.total_parameters = sum(p.numel() for p in model.parameters())
        
        # Setup optimizer (if applicable)
        if hasattr(model, 'parameters') and self.config.learning_rule != "stdp":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            criterion = nn.CrossEntropyLoss()
        else:
            optimizer = None
            criterion = None
        
        # Training loop
        model.train()
        training_start = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            # Create batches
            num_samples = len(train_data)
            batch_size = self.config.batch_size
            
            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                batch_data = train_data[i:batch_end]
                batch_labels = train_labels[i:batch_end]
                
                # Forward pass
                if isinstance(model, (SpikingNeuralNetwork, LiquidStateMachine, ReservoirComputer)):
                    outputs = model(batch_data)
                    
                    # Handle different output formats
                    if outputs.dim() == 3:  # [batch, neurons, time]
                        outputs = outputs.mean(dim=-1)  # Average over time
                else:
                    outputs = model(batch_data)
                
                # Compute loss and accuracy
                if criterion is not None:
                    loss = criterion(outputs, batch_labels)
                    
                    if optimizer is not None:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == batch_labels).float().mean().item()
                epoch_accuracy += accuracy
                
                num_batches += 1
            
            # Average metrics over epoch
            avg_loss = epoch_loss / max(1, num_batches)
            avg_accuracy = epoch_accuracy / max(1, num_batches)
            
            result.train_loss.append(avg_loss)
            result.train_accuracy.append(avg_accuracy)
            
            epoch_time = time.time() - epoch_start
            result.epoch_times.append(epoch_time)
            
            # Log progress
            if epoch % 10 == 0 or epoch == self.config.num_epochs - 1:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs}: "
                    f"Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}, Time={epoch_time:.2f}s"
                )
        
        result.training_time = time.time() - training_start
        
        # Collect neuromorphic-specific metrics
        if hasattr(model, 'get_network_activity'):
            activity_stats = model.get_network_activity()
            result.learning_stats.update(activity_stats)
        
        self.logger.info(f"Training completed in {result.training_time:.2f}s")
        
        return result
    
    def evaluate_model(self, model: nn.Module, test_data: torch.Tensor, 
                      test_labels: torch.Tensor, result: ExperimentResult):
        """Evaluate model performance and update results.
        
        Args:
            model: Trained model
            test_data: Test data
            test_labels: Test labels
            result: Result object to update
        """
        model.eval()
        
        test_loss = 0.0
        test_accuracy = 0.0
        total_spike_rate = 0.0
        total_sparsity = 0.0
        num_batches = 0
        
        inference_start = time.time()
        
        with torch.no_grad():
            batch_size = self.config.test_batch_size
            num_samples = len(test_data)
            
            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                batch_data = test_data[i:batch_end]
                batch_labels = test_labels[i:batch_end]
                
                # Forward pass
                outputs = model(batch_data)
                
                # Handle different output formats
                if outputs.dim() == 3:  # [batch, neurons, time]
                    # Compute neuromorphic metrics
                    spike_rate = outputs.sum() / (outputs.numel() * 0.001)  # Assume 1ms timesteps
                    sparsity = (outputs == 0).float().mean()
                    
                    total_spike_rate += spike_rate.item()
                    total_sparsity += sparsity.item()
                    
                    outputs = outputs.mean(dim=-1)  # Average over time
                
                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == batch_labels).float().mean()
                test_accuracy += accuracy.item()
                
                num_batches += 1
        
        inference_time = time.time() - inference_start
        result.inference_time = inference_time
        
        # Average metrics
        result.test_accuracy = [test_accuracy / max(1, num_batches)]
        result.spike_rates = [total_spike_rate / max(1, num_batches)]
        result.sparsity = [total_sparsity / max(1, num_batches)]
        
        self.logger.info(
            f"Test Results: Acc={result.test_accuracy[0]:.4f}, "
            f"Spike Rate={result.spike_rates[0]:.2f} Hz, "
            f"Sparsity={result.sparsity[0]:.4f}, "
            f"Inference Time={inference_time:.3f}s"
        )
    
    def run_baseline_comparisons(self, train_data: torch.Tensor, train_labels: torch.Tensor,
                               test_data: torch.Tensor, test_labels: torch.Tensor):
        """Run baseline model comparisons.
        
        Args:
            train_data: Training data
            train_labels: Training labels
            test_data: Test data
            test_labels: Test labels
        """
        baselines = {
            "dense_mlp": self.baseline_models.create_dense_mlp(
                input_size=self.config.input_size,
                hidden_sizes=self.config.hidden_sizes,
                output_size=self.config.output_size,
                device=self.device
            ),
            "lstm": self.baseline_models.create_lstm_baseline(
                input_size=self.config.input_size,
                hidden_size=self.config.hidden_sizes[0] if self.config.hidden_sizes else 64,
                num_layers=2,
                output_size=self.config.output_size,
                device=self.device
            )
        }
        
        for baseline_name, baseline_model in baselines.items():
            self.logger.info(f"Training baseline: {baseline_name}")
            
            # Create temporary config for baseline
            baseline_config = ExperimentConfig(
                experiment_name=f"{self.config.experiment_name}_baseline_{baseline_name}",
                description=f"Baseline comparison: {baseline_name}",
                dataset=self.config.dataset,
                model_type=baseline_name,
                learning_rule="backprop",
                num_epochs=self.config.num_epochs // 2,  # Faster training
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate
            )
            
            # Train baseline
            optimizer = torch.optim.Adam(baseline_model.parameters(), lr=self.config.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            baseline_result = ExperimentResult(config=baseline_config, device_used=self.device)
            baseline_result.total_parameters = sum(p.numel() for p in baseline_model.parameters())
            
            baseline_model.train()
            for epoch in range(baseline_config.num_epochs):
                epoch_loss = 0.0
                epoch_accuracy = 0.0
                num_batches = 0
                
                for i in range(0, len(train_data), self.config.batch_size):
                    batch_end = min(i + self.config.batch_size, len(train_data))
                    batch_data = train_data[i:batch_end]
                    batch_labels = train_labels[i:batch_end]
                    
                    # Flatten temporal data for non-temporal baselines
                    if baseline_name == "dense_mlp":
                        batch_data = batch_data.mean(dim=-1)  # Average over time
                    
                    optimizer.zero_grad()
                    outputs = baseline_model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    accuracy = (predicted == batch_labels).float().mean()
                    epoch_accuracy += accuracy.item()
                    num_batches += 1
                
                baseline_result.train_loss.append(epoch_loss / num_batches)
                baseline_result.train_accuracy.append(epoch_accuracy / num_batches)
            
            # Evaluate baseline
            baseline_model.eval()
            test_accuracy = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for i in range(0, len(test_data), self.config.test_batch_size):
                    batch_end = min(i + self.config.test_batch_size, len(test_data))
                    batch_data = test_data[i:batch_end]
                    batch_labels = test_labels[i:batch_end]
                    
                    if baseline_name == "dense_mlp":
                        batch_data = batch_data.mean(dim=-1)
                    
                    outputs = baseline_model(batch_data)
                    _, predicted = torch.max(outputs, 1)
                    accuracy = (predicted == batch_labels).float().mean()
                    test_accuracy += accuracy.item()
                    num_batches += 1
            
            baseline_result.test_accuracy = [test_accuracy / num_batches]
            self.baseline_results[baseline_name] = baseline_result
            
            self.logger.info(f"Baseline {baseline_name}: Test Acc={baseline_result.test_accuracy[0]:.4f}")
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run complete experiment with multiple trials.
        
        Returns:
            Experiment summary results
        """
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        self.logger.info(f"Description: {self.config.description}")
        
        # Generate dataset
        train_data, train_labels, test_data, test_labels = self.generate_dataset()
        
        # Run multiple trials
        for trial in range(self.config.num_trials):
            self.logger.info(f"Running trial {trial + 1}/{self.config.num_trials}")
            
            # Set seed for reproducibility
            torch.manual_seed(self.config.seed + trial)
            np.random.seed(self.config.seed + trial)
            
            # Create model
            model = self.create_model(self.config.model_type)
            
            # Train model
            result = self.train_model(model, train_data, train_labels)
            
            # Evaluate model
            self.evaluate_model(model, test_data, test_labels, result)
            
            # Store result
            self.results.append(result)
        
        # Run baseline comparisons
        if self.config.comparison_baselines:
            self.run_baseline_comparisons(train_data, train_labels, test_data, test_labels)
        
        # Generate summary
        summary = self._generate_experiment_summary()
        
        # Save results if requested
        if self.config.save_results:
            self._save_results(summary)
        
        # Plot results if requested
        if self.config.plot_results:
            self._plot_results()
        
        self.logger.info("Experiment completed successfully")
        
        return summary
    
    def _generate_experiment_summary(self) -> Dict[str, Any]:
        """Generate experiment summary statistics.
        
        Returns:
            Summary statistics dictionary
        """
        if not self.results:
            return {"error": "No results available"}
        
        # Aggregate metrics across trials
        test_accuracies = [r.test_accuracy[0] if r.test_accuracy else 0.0 for r in self.results]
        train_accuracies = [r.train_accuracy[-1] if r.train_accuracy else 0.0 for r in self.results]
        training_times = [r.training_time for r in self.results]
        spike_rates = [r.spike_rates[0] if r.spike_rates else 0.0 for r in self.results]
        sparsities = [r.sparsity[0] if r.sparsity else 0.0 for r in self.results]
        
        summary = {
            "experiment_config": asdict(self.config),
            "num_trials": len(self.results),
            "performance": {
                "test_accuracy": {
                    "mean": np.mean(test_accuracies),
                    "std": np.std(test_accuracies),
                    "min": np.min(test_accuracies),
                    "max": np.max(test_accuracies),
                    "values": test_accuracies
                },
                "train_accuracy": {
                    "mean": np.mean(train_accuracies),
                    "std": np.std(train_accuracies),
                    "values": train_accuracies
                }
            },
            "efficiency": {
                "training_time": {
                    "mean": np.mean(training_times),
                    "std": np.std(training_times),
                    "values": training_times
                },
                "spike_rate": {
                    "mean": np.mean(spike_rates),
                    "std": np.std(spike_rates),
                    "values": spike_rates
                },
                "sparsity": {
                    "mean": np.mean(sparsities),
                    "std": np.std(sparsities),
                    "values": sparsities
                }
            },
            "model_info": {
                "total_parameters": self.results[0].total_parameters,
                "device_used": self.results[0].device_used
            }
        }
        
        # Add baseline comparisons if available
        if self.baseline_results:
            baseline_summary = {}
            for name, result in self.baseline_results.items():
                baseline_summary[name] = {
                    "test_accuracy": result.test_accuracy[0] if result.test_accuracy else 0.0,
                    "train_accuracy": result.train_accuracy[-1] if result.train_accuracy else 0.0,
                    "parameters": result.total_parameters
                }
            
            summary["baselines"] = baseline_summary
        
        return summary
    
    def _save_results(self, summary: Dict[str, Any]):
        """Save experiment results to files.
        
        Args:
            summary: Experiment summary to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(f"experiment_results/{self.config.experiment_name}_{timestamp}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary as JSON
        with open(results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed results as pickle
        with open(results_dir / "detailed_results.pkl", "wb") as f:
            pickle.dump({
                "config": self.config,
                "results": self.results,
                "baseline_results": self.baseline_results,
                "summary": summary
            }, f)
        
        self.logger.info(f"Results saved to {results_dir}")
    
    def _plot_results(self):
        """Generate plots for experiment results."""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Experiment Results: {self.config.experiment_name}", fontsize=16)
        
        # Training curves
        ax1 = axes[0, 0]
        for i, result in enumerate(self.results):
            if result.train_accuracy:
                ax1.plot(result.train_accuracy, alpha=0.7, label=f"Trial {i+1}")
        ax1.set_title("Training Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True)
        
        # Test accuracy distribution
        ax2 = axes[0, 1]
        test_accs = [r.test_accuracy[0] if r.test_accuracy else 0.0 for r in self.results]
        ax2.hist(test_accs, bins=10, alpha=0.7, edgecolor='black')
        ax2.set_title("Test Accuracy Distribution")
        ax2.set_xlabel("Test Accuracy")
        ax2.set_ylabel("Frequency")
        ax2.axvline(np.mean(test_accs), color='red', linestyle='--', label=f'Mean: {np.mean(test_accs):.3f}')
        ax2.legend()
        ax2.grid(True)
        
        # Neuromorphic metrics
        ax3 = axes[1, 0]
        spike_rates = [r.spike_rates[0] if r.spike_rates else 0.0 for r in self.results]
        sparsities = [r.sparsity[0] if r.sparsity else 0.0 for r in self.results]
        
        x = np.arange(len(spike_rates))
        width = 0.35
        ax3.bar(x - width/2, spike_rates, width, label='Spike Rate (Hz)', alpha=0.7)
        ax3_twin = ax3.twinx()
        ax3_twin.bar(x + width/2, sparsities, width, label='Sparsity', alpha=0.7, color='orange')
        
        ax3.set_title("Neuromorphic Metrics by Trial")
        ax3.set_xlabel("Trial")
        ax3.set_ylabel("Spike Rate (Hz)")
        ax3_twin.set_ylabel("Sparsity")
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        
        # Baseline comparison
        ax4 = axes[1, 1]
        if self.baseline_results:
            models = [self.config.model_type] + list(self.baseline_results.keys())
            accuracies = [np.mean([r.test_accuracy[0] if r.test_accuracy else 0.0 for r in self.results])]
            accuracies.extend([result.test_accuracy[0] if result.test_accuracy else 0.0 
                             for result in self.baseline_results.values()])
            
            bars = ax4.bar(models, accuracies, alpha=0.7)
            ax4.set_title("Model Comparison")
            ax4.set_ylabel("Test Accuracy")
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, "No baseline comparisons", ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("Baseline Comparison (Not Available)")
        
        plt.tight_layout()
        plt.show()
        
        self.logger.info("Plots generated successfully")