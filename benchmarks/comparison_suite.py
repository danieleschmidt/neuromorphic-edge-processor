"""Comprehensive comparison suite for neuromorphic vs traditional approaches."""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .performance_benchmarks import PerformanceBenchmark
from .energy_benchmarks import EnergyBenchmark
from .accuracy_benchmarks import AccuracyBenchmark
from ..src.models.spiking_neural_network import SpikingNeuralNetwork
from ..src.models.liquid_state_machine import LiquidStateMachine
from ..src.models.reservoir_computer import ReservoirComputer


@dataclass
class ComparisonResult:
    """Results from neuromorphic vs traditional comparison."""
    task_name: str
    neuromorphic_score: float
    traditional_score: float
    advantage_ratio: float
    metric_type: str
    additional_info: Optional[Dict] = None


class TraditionalMLPModel(torch.nn.Module):
    """Traditional MLP for comparison."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_size, hidden_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.1))
            prev_size = hidden_size
        
        layers.append(torch.nn.Linear(prev_size, output_size))
        
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        if x.dim() == 3:  # Handle sequence data
            x = x.mean(dim=-1)  # Average over time
        return self.network(x)


class TraditionalLSTMModel(torch.nn.Module):
    """Traditional LSTM for temporal sequence comparison."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        self.fc = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.transpose(1, 2)  # [batch, time, features]
        
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Use last time step
        return output


class ComparisonSuite:
    """Comprehensive comparison suite between neuromorphic and traditional approaches."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.performance_benchmark = PerformanceBenchmark(device)
        self.energy_benchmark = EnergyBenchmark()
        self.accuracy_benchmark = AccuracyBenchmark()
        self.comparison_results = []
        
    def compare_inference_speed(
        self,
        neuromorphic_models: Dict[str, torch.nn.Module],
        traditional_models: Dict[str, torch.nn.Module],
        test_data: List[torch.Tensor],
        sparsity_levels: List[float] = [0.1, 0.5, 0.9]
    ) -> List[ComparisonResult]:
        """Compare inference speed between neuromorphic and traditional models."""
        
        results = []
        
        for sparsity in sparsity_levels:
            # Create sparse test data
            sparse_test_data = []
            for data in test_data:
                sparse_data = data.clone()
                mask = torch.rand_like(sparse_data) < (1 - sparsity)
                sparse_data = sparse_data * mask
                sparse_test_data.append(sparse_data)
            
            # Benchmark neuromorphic models
            neuro_times = []
            for name, model in neuromorphic_models.items():
                result = self.performance_benchmark.benchmark_inference_speed(
                    model, sparse_test_data, f"neuro_{name}_sparsity_{sparsity}"
                )
                neuro_times.append(result.execution_time)
            
            # Benchmark traditional models
            trad_times = []
            for name, model in traditional_models.items():
                result = self.performance_benchmark.benchmark_inference_speed(
                    model, sparse_test_data, f"trad_{name}_sparsity_{sparsity}"
                )
                trad_times.append(result.execution_time)
            
            # Compare average performance
            avg_neuro_time = np.mean(neuro_times)
            avg_trad_time = np.mean(trad_times)
            speedup_ratio = avg_trad_time / avg_neuro_time
            
            comparison = ComparisonResult(
                task_name=f"inference_speed_sparsity_{sparsity}",
                neuromorphic_score=1.0 / avg_neuro_time,  # Higher is better (throughput)
                traditional_score=1.0 / avg_trad_time,
                advantage_ratio=speedup_ratio,
                metric_type="speed",
                additional_info={
                    "sparsity_level": sparsity,
                    "neuro_times": neuro_times,
                    "trad_times": trad_times,
                    "neuromorphic_advantage": speedup_ratio > 1.0
                }
            )
            
            results.append(comparison)
            self.comparison_results.append(comparison)
        
        return results
    
    def compare_energy_efficiency(
        self,
        neuromorphic_models: Dict[str, torch.nn.Module],
        traditional_models: Dict[str, torch.nn.Module],
        test_data: List[torch.Tensor],
        activity_levels: List[float] = [0.1, 0.3, 0.5]
    ) -> List[ComparisonResult]:
        """Compare energy efficiency between model types."""
        
        results = []
        
        for activity in activity_levels:
            # Scale input activity
            active_test_data = [data * activity for data in test_data]
            
            # Energy analysis for neuromorphic models
            neuro_energies = []
            for name, model in neuromorphic_models.items():
                if hasattr(model, 'layers') and hasattr(model.layers[0], 'neurons'):
                    # Spiking neural network
                    energy_info = self.energy_benchmark.estimate_spiking_network_energy(
                        model, active_test_data[0]
                    )
                    neuro_energies.append(energy_info["total_energy"])
                elif hasattr(model, 'reservoirs') or hasattr(model, 'reservoir_weights'):
                    # Reservoir computing
                    energy_info = self.energy_benchmark.estimate_reservoir_energy(
                        model, active_test_data[0]
                    )
                    neuro_energies.append(energy_info["total_energy"])
                else:
                    neuro_energies.append(100.0)  # Default estimate
            
            # Energy analysis for traditional models (rough estimates)
            trad_energies = []
            for name, model in traditional_models.items():
                param_count = sum(p.numel() for p in model.parameters())
                # Traditional models use all parameters regardless of sparsity
                energy = param_count * self.energy_benchmark.energy_model.computation_energy
                trad_energies.append(energy)
            
            # Compare average energy efficiency
            avg_neuro_energy = np.mean(neuro_energies)
            avg_trad_energy = np.mean(trad_energies)
            efficiency_ratio = avg_trad_energy / avg_neuro_energy
            
            comparison = ComparisonResult(
                task_name=f"energy_efficiency_activity_{activity}",
                neuromorphic_score=1.0 / avg_neuro_energy,  # Higher is better
                traditional_score=1.0 / avg_trad_energy,
                advantage_ratio=efficiency_ratio,
                metric_type="energy_efficiency",
                additional_info={
                    "activity_level": activity,
                    "neuro_energies": neuro_energies,
                    "trad_energies": trad_energies,
                    "energy_savings": 1.0 - (avg_neuro_energy / avg_trad_energy)
                }
            )
            
            results.append(comparison)
            self.comparison_results.append(comparison)
        
        return results
    
    def compare_accuracy_vs_sparsity(
        self,
        neuromorphic_models: Dict[str, torch.nn.Module],
        traditional_models: Dict[str, torch.nn.Module],
        test_data: List[Tuple[torch.Tensor, torch.Tensor]],
        sparsity_levels: List[float] = [0.0, 0.3, 0.5, 0.7, 0.9]
    ) -> List[ComparisonResult]:
        """Compare how accuracy degrades with input sparsity."""
        
        results = []
        
        # Baseline accuracy on dense data (sparsity = 0)
        baseline_accuracies = {}
        
        for sparsity in sparsity_levels:
            # Create sparse test data
            sparse_test_data = []
            for inputs, targets in test_data:
                sparse_inputs = inputs.clone()
                mask = torch.rand_like(sparse_inputs) < (1 - sparsity)
                sparse_inputs = sparse_inputs * mask
                sparse_test_data.append((sparse_inputs, targets))
            
            # Evaluate neuromorphic models
            neuro_accuracies = []
            for name, model in neuromorphic_models.items():
                result = self.accuracy_benchmark.evaluate_classification(
                    model, sparse_test_data, f"neuro_{name}", num_classes=2
                )
                accuracy = result.accuracy
                neuro_accuracies.append(accuracy)
                
                if sparsity == 0.0:
                    baseline_accuracies[f"neuro_{name}"] = accuracy
            
            # Evaluate traditional models
            trad_accuracies = []
            for name, model in traditional_models.items():
                result = self.accuracy_benchmark.evaluate_classification(
                    model, sparse_test_data, f"trad_{name}", num_classes=2
                )
                accuracy = result.accuracy
                trad_accuracies.append(accuracy)
                
                if sparsity == 0.0:
                    baseline_accuracies[f"trad_{name}"] = accuracy
            
            # Compare robustness to sparsity
            avg_neuro_acc = np.mean(neuro_accuracies)
            avg_trad_acc = np.mean(trad_accuracies)
            
            # Compute accuracy retention relative to dense case
            if sparsity > 0.0:
                baseline_neuro = np.mean([baseline_accuracies[f"neuro_{name}"] for name in neuromorphic_models.keys()])
                baseline_trad = np.mean([baseline_accuracies[f"trad_{name}"] for name in traditional_models.keys()])
                
                neuro_retention = avg_neuro_acc / baseline_neuro if baseline_neuro > 0 else 0
                trad_retention = avg_trad_acc / baseline_trad if baseline_trad > 0 else 0
                robustness_advantage = neuro_retention / trad_retention if trad_retention > 0 else 1.0
            else:
                robustness_advantage = 1.0
            
            comparison = ComparisonResult(
                task_name=f"accuracy_vs_sparsity_{sparsity}",
                neuromorphic_score=avg_neuro_acc,
                traditional_score=avg_trad_acc,
                advantage_ratio=robustness_advantage,
                metric_type="accuracy",
                additional_info={
                    "sparsity_level": sparsity,
                    "neuro_accuracies": neuro_accuracies,
                    "trad_accuracies": trad_accuracies,
                    "accuracy_advantage": avg_neuro_acc / avg_trad_acc if avg_trad_acc > 0 else 1.0
                }
            )
            
            results.append(comparison)
            self.comparison_results.append(comparison)
        
        return results
    
    def compare_temporal_processing(
        self,
        neuromorphic_models: Dict[str, torch.nn.Module],
        traditional_models: Dict[str, torch.nn.Module],
        temporal_sequences: List[Tuple[torch.Tensor, torch.Tensor]],
        sequence_lengths: List[int] = [10, 50, 100, 200]
    ) -> List[ComparisonResult]:
        """Compare temporal sequence processing capabilities."""
        
        results = []
        
        for seq_length in sequence_lengths:
            # Truncate/extend sequences to desired length
            adapted_sequences = []
            for inputs, targets in temporal_sequences:
                if inputs.shape[-1] > seq_length:
                    adapted_inputs = inputs[:, :, :seq_length]
                else:
                    # Pad with zeros
                    padding = seq_length - inputs.shape[-1]
                    adapted_inputs = torch.nn.functional.pad(inputs, (0, padding))
                
                adapted_sequences.append((adapted_inputs, targets))
            
            # Evaluate neuromorphic models
            neuro_scores = []
            for name, model in neuromorphic_models.items():
                result = self.accuracy_benchmark.evaluate_temporal_sequence_prediction(
                    model, adapted_sequences, f"neuro_{name}", prediction_horizon=1
                )
                neuro_scores.append(result.accuracy)
            
            # Evaluate traditional models
            trad_scores = []
            for name, model in traditional_models.items():
                result = self.accuracy_benchmark.evaluate_temporal_sequence_prediction(
                    model, adapted_sequences, f"trad_{name}", prediction_horizon=1
                )
                trad_scores.append(result.accuracy)
            
            # Compare performance
            avg_neuro_score = np.mean(neuro_scores)
            avg_trad_score = np.mean(trad_scores)
            temporal_advantage = avg_neuro_score / avg_trad_score if avg_trad_score > 0 else 1.0
            
            comparison = ComparisonResult(
                task_name=f"temporal_processing_length_{seq_length}",
                neuromorphic_score=avg_neuro_score,
                traditional_score=avg_trad_score,
                advantage_ratio=temporal_advantage,
                metric_type="temporal_accuracy",
                additional_info={
                    "sequence_length": seq_length,
                    "neuro_scores": neuro_scores,
                    "trad_scores": trad_scores,
                    "num_sequences": len(adapted_sequences)
                }
            )
            
            results.append(comparison)
            self.comparison_results.append(comparison)
        
        return results
    
    def compare_memory_capacity(
        self,
        neuromorphic_models: Dict[str, torch.nn.Module],
        traditional_models: Dict[str, torch.nn.Module],
        memory_test_sequences: List[torch.Tensor],
        max_delays: List[int] = [5, 10, 20, 50]
    ) -> List[ComparisonResult]:
        """Compare memory capacity and retention."""
        
        results = []
        
        for max_delay in max_delays:
            neuro_capacities = []
            trad_capacities = []
            
            # Test neuromorphic models
            for name, model in neuromorphic_models.items():
                if hasattr(model, 'analyze_memory_capacity'):
                    # LSM or similar with built-in memory analysis
                    memory_result = model.analyze_memory_capacity(
                        memory_test_sequences[0], max_delay=max_delay
                    )
                    capacity = memory_result.get("total_memory_capacity", 0)
                else:
                    # Estimate memory capacity for spiking networks
                    capacity = self._estimate_memory_capacity(model, memory_test_sequences[0], max_delay)
                
                neuro_capacities.append(capacity)
            
            # Test traditional models (RNNs have implicit memory)
            for name, model in traditional_models.items():
                if hasattr(model, 'lstm'):
                    # LSTM models have explicit memory
                    capacity = self._estimate_lstm_memory_capacity(model, memory_test_sequences[0], max_delay)
                else:
                    # MLP has no temporal memory
                    capacity = 0.0
                
                trad_capacities.append(capacity)
            
            # Compare memory capabilities
            avg_neuro_capacity = np.mean(neuro_capacities)
            avg_trad_capacity = np.mean(trad_capacities)
            memory_advantage = avg_neuro_capacity / avg_trad_capacity if avg_trad_capacity > 0 else float('inf')
            
            comparison = ComparisonResult(
                task_name=f"memory_capacity_delay_{max_delay}",
                neuromorphic_score=avg_neuro_capacity,
                traditional_score=avg_trad_capacity,
                advantage_ratio=memory_advantage,
                metric_type="memory_capacity",
                additional_info={
                    "max_delay": max_delay,
                    "neuro_capacities": neuro_capacities,
                    "trad_capacities": trad_capacities
                }
            )
            
            results.append(comparison)
            self.comparison_results.append(comparison)
        
        return results
    
    def _estimate_memory_capacity(
        self,
        model: torch.nn.Module,
        test_sequence: torch.Tensor,
        max_delay: int
    ) -> float:
        """Estimate memory capacity for arbitrary neuromorphic models."""
        
        if test_sequence.dim() == 2:
            test_sequence = test_sequence.unsqueeze(0)
        
        capacity = 0.0
        
        try:
            with torch.no_grad():
                # Get model response
                if hasattr(model, 'forward') and 'return_spikes' in str(model.forward.__code__.co_varnames):
                    output, states = model(test_sequence, return_spikes=True)
                    model_states = states[-1] if states else output
                else:
                    model_states = model(test_sequence)
                
                # Test memory at different delays
                for delay in range(1, min(max_delay, test_sequence.shape[-1])):
                    if delay < model_states.shape[-1]:
                        # Correlation between input at t and state at t+delay
                        input_signal = test_sequence[0, 0, :-delay]  # First channel
                        state_signal = model_states[0, 0, delay:]   # First neuron/output
                        
                        if len(input_signal) > 1 and len(state_signal) > 1:
                            corr = torch.corrcoef(torch.stack([input_signal, state_signal]))[0, 1]
                            if not torch.isnan(corr):
                                capacity += corr.abs().item() ** 2
        
        except:
            capacity = 0.0
        
        return capacity
    
    def _estimate_lstm_memory_capacity(
        self,
        model: torch.nn.Module,
        test_sequence: torch.Tensor,
        max_delay: int
    ) -> float:
        """Estimate memory capacity for LSTM models."""
        
        if test_sequence.dim() == 2:
            test_sequence = test_sequence.unsqueeze(0)
        
        capacity = 0.0
        
        try:
            with torch.no_grad():
                # LSTM forward pass
                x = test_sequence.transpose(1, 2)  # [batch, time, features]
                lstm_out, _ = model.lstm(x)
                
                # Test memory at different delays
                for delay in range(1, min(max_delay, x.shape[1])):
                    if delay < lstm_out.shape[1]:
                        # Input at time t, LSTM state at time t+delay
                        input_signal = x[0, :-delay, 0]  # First feature
                        state_signal = lstm_out[0, delay:, 0]  # First hidden unit
                        
                        if len(input_signal) > 1 and len(state_signal) > 1:
                            corr = torch.corrcoef(torch.stack([input_signal, state_signal]))[0, 1]
                            if not torch.isnan(corr):
                                capacity += corr.abs().item() ** 2
        
        except:
            capacity = 0.0
        
        return capacity
    
    def run_comprehensive_comparison(
        self,
        neuromorphic_models: Dict[str, torch.nn.Module],
        traditional_models: Dict[str, torch.nn.Module],
        test_datasets: Dict[str, List],
        output_file: str = "comprehensive_comparison.json"
    ) -> Dict:
        """Run comprehensive comparison across all metrics."""
        
        print("Running comprehensive neuromorphic vs traditional comparison...")
        
        # 1. Inference Speed Comparison
        print("1. Comparing inference speed...")
        if "inference_data" in test_datasets:
            speed_results = self.compare_inference_speed(
                neuromorphic_models, traditional_models, test_datasets["inference_data"]
            )
        
        # 2. Energy Efficiency Comparison
        print("2. Comparing energy efficiency...")
        if "energy_data" in test_datasets:
            energy_results = self.compare_energy_efficiency(
                neuromorphic_models, traditional_models, test_datasets["energy_data"]
            )
        
        # 3. Accuracy vs Sparsity
        print("3. Comparing accuracy vs sparsity...")
        if "classification_data" in test_datasets:
            accuracy_results = self.compare_accuracy_vs_sparsity(
                neuromorphic_models, traditional_models, test_datasets["classification_data"]
            )
        
        # 4. Temporal Processing
        print("4. Comparing temporal processing...")
        if "temporal_data" in test_datasets:
            temporal_results = self.compare_temporal_processing(
                neuromorphic_models, traditional_models, test_datasets["temporal_data"]
            )
        
        # 5. Memory Capacity
        print("5. Comparing memory capacity...")
        if "memory_data" in test_datasets:
            memory_results = self.compare_memory_capacity(
                neuromorphic_models, traditional_models, test_datasets["memory_data"]
            )
        
        # Generate comprehensive report
        report = self.generate_comparison_report()
        
        # Save results
        import json
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Comprehensive comparison results saved to {output_file}")
        
        return report
    
    def generate_comparison_report(self) -> Dict:
        """Generate comprehensive comparison report."""
        
        report = {
            "summary": {
                "total_comparisons": len(self.comparison_results),
                "comparison_tasks": list(set(r.task_name for r in self.comparison_results)),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "neuromorphic_advantages": {},
            "traditional_advantages": {},
            "overall_analysis": {}
        }
        
        # Analyze advantages by metric type
        metric_types = set(r.metric_type for r in self.comparison_results)
        
        for metric in metric_types:
            metric_results = [r for r in self.comparison_results if r.metric_type == metric]
            
            # Count where neuromorphic has advantage
            neuro_wins = sum(1 for r in metric_results if r.advantage_ratio > 1.0)
            total_comparisons = len(metric_results)
            
            advantage_ratios = [r.advantage_ratio for r in metric_results]
            
            report["neuromorphic_advantages"][metric] = {
                "wins": neuro_wins,
                "total": total_comparisons,
                "win_rate": neuro_wins / total_comparisons if total_comparisons > 0 else 0,
                "mean_advantage": np.mean(advantage_ratios),
                "max_advantage": max(advantage_ratios),
                "tasks_with_advantage": [
                    r.task_name for r in metric_results if r.advantage_ratio > 1.0
                ]
            }
        
        # Overall analysis
        all_advantages = [r.advantage_ratio for r in self.comparison_results]
        overall_wins = sum(1 for ratio in all_advantages if ratio > 1.0)
        
        report["overall_analysis"] = {
            "neuromorphic_win_rate": overall_wins / len(all_advantages) if all_advantages else 0,
            "mean_advantage_ratio": np.mean(all_advantages) if all_advantages else 1.0,
            "std_advantage_ratio": np.std(all_advantages) if all_advantages else 0.0,
            "best_neuromorphic_tasks": [
                r.task_name for r in sorted(
                    self.comparison_results, 
                    key=lambda x: x.advantage_ratio, 
                    reverse=True
                )[:5]
            ],
            "best_traditional_tasks": [
                r.task_name for r in sorted(
                    self.comparison_results,
                    key=lambda x: x.advantage_ratio
                )[:5]
            ]
        }
        
        return report
    
    def clear_results(self):
        """Clear all comparison results."""
        self.comparison_results.clear()
        self.performance_benchmark.clear_results()
        self.energy_benchmark.results.clear()
        self.accuracy_benchmark.results.clear()