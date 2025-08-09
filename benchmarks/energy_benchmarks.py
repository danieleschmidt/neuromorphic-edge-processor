"""Energy efficiency benchmarking for neuromorphic processors."""

import torch
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ..benchmarks.performance_benchmarks import BenchmarkResult


@dataclass
class EnergyModel:
    """Energy consumption model for different operations."""
    spike_energy: float = 1.0  # Energy per spike (arbitrary units)
    synapse_energy: float = 0.1  # Energy per synaptic operation
    neuron_idle_energy: float = 0.01  # Energy per neuron per time step when idle
    memory_access_energy: float = 0.05  # Energy per memory access
    computation_energy: float = 0.5  # Energy per floating point operation


class EnergyBenchmark:
    """Energy efficiency benchmarking suite."""
    
    def __init__(self, energy_model: Optional[EnergyModel] = None):
        self.energy_model = energy_model or EnergyModel()
        self.results = []
    
    def estimate_spiking_network_energy(
        self,
        model,
        input_data: torch.Tensor,
        duration: float = 100.0
    ) -> Dict:
        """Estimate energy consumption of a spiking neural network."""
        
        # Forward pass to get spike activity
        with torch.no_grad():
            output, spike_trains = model.forward(input_data, duration, return_spikes=True)
        
        total_energy = 0.0
        energy_breakdown = {}
        
        # Count spikes across all layers
        total_spikes = 0
        for i, spikes in enumerate(spike_trains):
            layer_spikes = spikes.sum().item()
            layer_energy = layer_spikes * self.energy_model.spike_energy
            
            total_spikes += layer_spikes
            total_energy += layer_energy
            energy_breakdown[f"layer_{i}_spikes"] = layer_energy
        
        # Estimate synaptic operations
        total_synapses = 0
        for layer in model.layers:
            # Count active synapses (connections * activity)
            active_synapses = layer.connection_mask.sum().item()
            synapse_energy = active_synapses * self.energy_model.synapse_energy
            
            total_synapses += active_synapses
            total_energy += synapse_energy
            energy_breakdown[f"synapses"] = synapse_energy
        
        # Idle neuron energy
        total_neurons = sum(layer.output_size for layer in model.layers)
        time_steps = duration / model.dt
        idle_operations = total_neurons * time_steps - total_spikes
        idle_energy = idle_operations * self.energy_model.neuron_idle_energy
        
        total_energy += idle_energy
        energy_breakdown["idle_neurons"] = idle_energy
        
        # Memory access energy (rough estimate)
        param_count = sum(p.numel() for p in model.parameters())
        memory_energy = param_count * self.energy_model.memory_access_energy
        
        total_energy += memory_energy
        energy_breakdown["memory_access"] = memory_energy
        
        # Energy efficiency metrics
        sparsity = 1.0 - (total_spikes / (total_neurons * time_steps))
        energy_per_spike = total_energy / max(total_spikes, 1)
        
        # Comparison with dense computation
        dense_operations = total_neurons * time_steps
        dense_energy = dense_operations * self.energy_model.computation_energy
        energy_savings = 1.0 - (total_energy / dense_energy)
        
        return {
            "total_energy": total_energy,
            "energy_breakdown": energy_breakdown,
            "total_spikes": int(total_spikes),
            "sparsity": sparsity,
            "energy_per_spike": energy_per_spike,
            "dense_energy_equivalent": dense_energy,
            "energy_savings": energy_savings,
            "energy_efficiency_ratio": dense_energy / total_energy
        }
    
    def estimate_reservoir_energy(
        self,
        reservoir_model,
        input_sequence: torch.Tensor
    ) -> Dict:
        """Estimate energy consumption of reservoir computing models."""
        
        with torch.no_grad():
            output, states = reservoir_model.forward(input_sequence, return_reservoir_states=True)
        
        total_energy = 0.0
        energy_breakdown = {}
        
        # Compute reservoir activity energy
        for i, reservoir_states in enumerate(states):
            # Continuous-valued neurons: energy proportional to activity level
            activity_energy = reservoir_states.abs().sum().item() * self.energy_model.computation_energy
            total_energy += activity_energy
            energy_breakdown[f"reservoir_{i}_activity"] = activity_energy
        
        # Connection energy (based on reservoir connectivity)
        if hasattr(reservoir_model, 'reservoirs'):
            for i, reservoir in enumerate(reservoir_model.reservoirs):
                connection_energy = reservoir.reservoir_weights.abs().sum().item() * self.energy_model.synapse_energy
                total_energy += connection_energy
                energy_breakdown[f"reservoir_{i}_connections"] = connection_energy
        
        # Memory access energy
        param_count = sum(p.numel() for p in reservoir_model.parameters())
        memory_energy = param_count * self.energy_model.memory_access_energy
        total_energy += memory_energy
        energy_breakdown["memory_access"] = memory_energy
        
        # Efficiency metrics
        total_operations = sum(s.numel() for s in states)
        energy_per_operation = total_energy / max(total_operations, 1)
        
        return {
            "total_energy": total_energy,
            "energy_breakdown": energy_breakdown,
            "total_operations": int(total_operations),
            "energy_per_operation": energy_per_operation,
            "reservoir_efficiency": 1.0 / energy_per_operation if energy_per_operation > 0 else 0
        }
    
    def benchmark_energy_vs_accuracy(
        self,
        models: Dict[str, torch.nn.Module],
        test_data: List[Tuple[torch.Tensor, torch.Tensor]],
        accuracy_fn: callable
    ) -> List[BenchmarkResult]:
        """Benchmark energy consumption vs accuracy trade-off."""
        
        results = []
        
        for model_name, model in models.items():
            total_energy = 0.0
            total_accuracy = 0.0
            
            for inputs, targets in test_data:
                # Estimate energy based on model type
                if hasattr(model, 'layers') and hasattr(model.layers[0], 'neurons'):
                    # Spiking neural network
                    energy_info = self.estimate_spiking_network_energy(model, inputs)
                elif hasattr(model, 'reservoirs') or hasattr(model, 'reservoir_weights'):
                    # Reservoir computing
                    energy_info = self.estimate_reservoir_energy(model, inputs)
                else:
                    # Dense network (rough estimate)
                    param_count = sum(p.numel() for p in model.parameters())
                    energy_info = {
                        "total_energy": param_count * self.energy_model.computation_energy,
                        "energy_breakdown": {"computation": param_count * self.energy_model.computation_energy}
                    }
                
                # Compute accuracy
                with torch.no_grad():
                    outputs = model(inputs)
                    accuracy = accuracy_fn(outputs, targets)
                
                total_energy += energy_info["total_energy"]
                total_accuracy += accuracy
            
            avg_energy = total_energy / len(test_data)
            avg_accuracy = total_accuracy / len(test_data)
            
            # Energy efficiency score
            efficiency_score = avg_accuracy / avg_energy if avg_energy > 0 else 0
            
            result = BenchmarkResult(
                model_name=model_name,
                task_name="energy_vs_accuracy",
                execution_time=0,
                throughput=0,
                memory_usage=0,
                accuracy=avg_accuracy,
                energy_efficiency=efficiency_score,
                additional_metrics={
                    "average_energy": avg_energy,
                    "energy_per_sample": avg_energy,
                    "accuracy_per_energy_unit": avg_accuracy / avg_energy if avg_energy > 0 else 0
                }
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def benchmark_sparsity_energy_savings(
        self,
        spiking_model,
        sparsity_levels: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
        test_input: torch.Tensor = None
    ) -> List[BenchmarkResult]:
        """Benchmark energy savings from sparsity."""
        
        if test_input is None:
            test_input = torch.randn(4, 100)  # Default test input
        
        results = []
        
        for sparsity in sparsity_levels:
            # Create sparse input
            sparse_input = test_input.clone()
            mask = torch.rand_like(sparse_input) < (1 - sparsity)
            sparse_input = sparse_input * mask
            
            # Estimate energy
            energy_info = self.estimate_spiking_network_energy(spiking_model, sparse_input)
            
            # Energy savings compared to dense (0% sparsity)
            if sparsity == 0.0:
                baseline_energy = energy_info["total_energy"]
            else:
                energy_savings = 1.0 - (energy_info["total_energy"] / baseline_energy) if 'baseline_energy' in locals() else 0
            
            result = BenchmarkResult(
                model_name=f"spiking_sparsity_{sparsity}",
                task_name="sparsity_energy_savings",
                execution_time=0,
                throughput=0,
                memory_usage=0,
                energy_efficiency=energy_info.get("energy_efficiency_ratio", 1.0),
                additional_metrics={
                    "sparsity_level": sparsity,
                    "total_energy": energy_info["total_energy"],
                    "energy_savings": energy_savings if 'energy_savings' in locals() else 0,
                    "spikes_count": energy_info["total_spikes"],
                    "energy_breakdown": energy_info["energy_breakdown"]
                }
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def benchmark_dynamic_power_scaling(
        self,
        models: Dict[str, torch.nn.Module],
        workload_intensities: List[float] = [0.1, 0.3, 0.5, 0.8, 1.0]
    ) -> List[BenchmarkResult]:
        """Benchmark dynamic power scaling with workload intensity."""
        
        results = []
        
        for model_name, model in models.items():
            for intensity in workload_intensities:
                # Generate workload based on intensity
                batch_size = max(1, int(intensity * 32))
                input_size = 100
                sequence_length = max(10, int(intensity * 100))
                
                test_input = torch.randn(batch_size, input_size, sequence_length) * intensity
                
                # Measure energy
                if hasattr(model, 'layers') and hasattr(model.layers[0], 'neurons'):
                    energy_info = self.estimate_spiking_network_energy(model, test_input)
                elif hasattr(model, 'reservoirs') or hasattr(model, 'reservoir_weights'):
                    energy_info = self.estimate_reservoir_energy(model, test_input)
                else:
                    # Dense model approximation
                    operations = batch_size * input_size * sequence_length
                    energy_info = {"total_energy": operations * self.energy_model.computation_energy}
                
                # Power efficiency at this intensity
                power_efficiency = intensity / energy_info["total_energy"] if energy_info["total_energy"] > 0 else 0
                
                result = BenchmarkResult(
                    model_name=model_name,
                    task_name="dynamic_power_scaling",
                    execution_time=0,
                    throughput=0,
                    memory_usage=0,
                    energy_efficiency=power_efficiency,
                    additional_metrics={
                        "workload_intensity": intensity,
                        "total_energy": energy_info["total_energy"],
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "energy_per_sample": energy_info["total_energy"] / batch_size
                    }
                )
                
                results.append(result)
                self.results.append(result)
        
        return results
    
    def benchmark_idle_vs_active_energy(
        self,
        models: Dict[str, torch.nn.Module],
        idle_duration: float = 100.0,
        active_duration: float = 100.0
    ) -> List[BenchmarkResult]:
        """Benchmark energy consumption during idle vs active periods."""
        
        results = []
        
        for model_name, model in models.items():
            # Idle period (minimal input)
            idle_input = torch.zeros(1, 100, int(idle_duration))
            
            # Active period (normal input)
            active_input = torch.randn(1, 100, int(active_duration))
            
            # Measure idle energy
            if hasattr(model, 'layers') and hasattr(model.layers[0], 'neurons'):
                idle_energy_info = self.estimate_spiking_network_energy(model, idle_input, idle_duration)
                active_energy_info = self.estimate_spiking_network_energy(model, active_input, active_duration)
            elif hasattr(model, 'reservoirs') or hasattr(model, 'reservoir_weights'):
                idle_energy_info = self.estimate_reservoir_energy(model, idle_input)
                active_energy_info = self.estimate_reservoir_energy(model, active_input)
            else:
                # Rough approximation for dense models
                idle_energy_info = {"total_energy": 100 * self.energy_model.neuron_idle_energy}
                active_energy_info = {"total_energy": 1000 * self.energy_model.computation_energy}
            
            idle_energy = idle_energy_info["total_energy"]
            active_energy = active_energy_info["total_energy"]
            
            # Dynamic range (how much energy scales with activity)
            dynamic_range = active_energy / idle_energy if idle_energy > 0 else float('inf')
            
            result = BenchmarkResult(
                model_name=model_name,
                task_name="idle_vs_active_energy",
                execution_time=0,
                throughput=0,
                memory_usage=0,
                energy_efficiency=1.0 / dynamic_range if dynamic_range < float('inf') else 1.0,
                additional_metrics={
                    "idle_energy": idle_energy,
                    "active_energy": active_energy,
                    "dynamic_range": dynamic_range,
                    "energy_ratio": active_energy / idle_energy if idle_energy > 0 else 0,
                    "adaptive_efficiency": (active_energy - idle_energy) / active_energy if active_energy > 0 else 0
                }
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def generate_energy_report(self) -> Dict:
        """Generate comprehensive energy efficiency report."""
        
        report = {
            "energy_model": self.energy_model.__dict__,
            "benchmark_summary": {
                "total_benchmarks": len(self.results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "energy_rankings": {},
            "efficiency_analysis": {}
        }
        
        # Group results by task
        tasks = set(r.task_name for r in self.results)
        
        for task in tasks:
            task_results = [r for r in self.results if r.task_name == task]
            
            # Rank by energy efficiency
            if any(r.energy_efficiency for r in task_results):
                efficiency_ranking = sorted(
                    [r for r in task_results if r.energy_efficiency is not None],
                    key=lambda x: x.energy_efficiency,
                    reverse=True
                )
                
                report["energy_rankings"][task] = [
                    {
                        "model": r.model_name,
                        "energy_efficiency": r.energy_efficiency,
                        "additional_metrics": r.additional_metrics
                    }
                    for r in efficiency_ranking[:5]  # Top 5
                ]
        
        # Overall efficiency analysis
        all_efficiencies = [r.energy_efficiency for r in self.results if r.energy_efficiency is not None]
        if all_efficiencies:
            report["efficiency_analysis"] = {
                "mean_efficiency": np.mean(all_efficiencies),
                "std_efficiency": np.std(all_efficiencies),
                "best_efficiency": max(all_efficiencies),
                "worst_efficiency": min(all_efficiencies)
            }
        
        return report
    
    def save_energy_results(self, filename: str = "energy_benchmark_results.json"):
        """Save energy benchmark results to file."""
        import json
        
        report = self.generate_energy_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Energy benchmark results saved to {filename}")