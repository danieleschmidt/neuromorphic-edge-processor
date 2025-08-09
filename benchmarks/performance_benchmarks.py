"""Performance benchmarking for neuromorphic processors."""

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from ..src.models.spiking_neural_network import SpikingNeuralNetwork
from ..src.models.liquid_state_machine import LiquidStateMachine
from ..src.models.reservoir_computer import ReservoirComputer
from ..src.algorithms.event_processor import EventDrivenProcessor
from ..src.algorithms.spike_processor import SpikeProcessor


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    task_name: str
    execution_time: float
    throughput: float
    memory_usage: float
    accuracy: Optional[float] = None
    energy_efficiency: Optional[float] = None
    additional_metrics: Optional[Dict] = None


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results = []
        
    def benchmark_inference_speed(
        self,
        model: torch.nn.Module,
        test_data: List[torch.Tensor],
        model_name: str,
        warmup_runs: int = 10,
        benchmark_runs: int = 100
    ) -> BenchmarkResult:
        """Benchmark inference speed of a model."""
        
        model.eval()
        model.to(self.device)
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                dummy_input = test_data[0].to(self.device)
                _ = model(dummy_input)
        
        # Benchmark
        execution_times = []
        
        for run in range(benchmark_runs):
            input_idx = run % len(test_data)
            input_tensor = test_data[input_idx].to(self.device)
            
            torch.cuda.synchronize() if self.device.startswith('cuda') else None
            
            start_time = time.time()
            with torch.no_grad():
                output = model(input_tensor)
            
            torch.cuda.synchronize() if self.device.startswith('cuda') else None
            end_time = time.time()
            
            execution_times.append(end_time - start_time)
        
        # Calculate metrics
        mean_time = np.mean(execution_times)
        std_time = np.std(execution_times)
        throughput = len(test_data[0]) / mean_time  # samples per second
        
        # Memory usage
        memory_usage = self._get_memory_usage(model)
        
        result = BenchmarkResult(
            model_name=model_name,
            task_name="inference_speed",
            execution_time=mean_time,
            throughput=throughput,
            memory_usage=memory_usage,
            additional_metrics={
                "std_time": std_time,
                "min_time": min(execution_times),
                "max_time": max(execution_times),
                "p95_time": np.percentile(execution_times, 95),
                "p99_time": np.percentile(execution_times, 99)
            }
        )
        
        self.results.append(result)
        return result
    
    def benchmark_training_speed(
        self,
        model: torch.nn.Module,
        train_data: List[Tuple[torch.Tensor, torch.Tensor]],
        model_name: str,
        optimizer: torch.optim.Optimizer,
        epochs: int = 10
    ) -> BenchmarkResult:
        """Benchmark training speed of a model."""
        
        model.train()
        model.to(self.device)
        
        training_times = []
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            for inputs, targets in train_data:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                if isinstance(model, SpikingNeuralNetwork):
                    loss, _ = model.train_step(inputs, targets)
                else:
                    outputs = model(inputs)
                    loss = torch.nn.functional.mse_loss(outputs.mean(dim=-1), targets)
                
                loss.backward()
                optimizer.step()
            
            epoch_time = time.time() - epoch_start
            training_times.append(epoch_time)
        
        mean_time = np.mean(training_times)
        throughput = len(train_data) / mean_time  # batches per second
        memory_usage = self._get_memory_usage(model)
        
        result = BenchmarkResult(
            model_name=model_name,
            task_name="training_speed",
            execution_time=mean_time,
            throughput=throughput,
            memory_usage=memory_usage,
            additional_metrics={
                "total_training_time": sum(training_times),
                "epochs": epochs
            }
        )
        
        self.results.append(result)
        return result
    
    def benchmark_scalability(
        self,
        model_class: type,
        model_configs: List[Dict],
        test_input_shape: tuple,
        model_name: str
    ) -> List[BenchmarkResult]:
        """Benchmark how performance scales with model size."""
        
        scalability_results = []
        
        for config in model_configs:
            # Create model with this configuration
            model = model_class(**config)
            model.to(self.device)
            
            # Generate test input
            test_input = torch.randn(1, *test_input_shape).to(self.device)
            
            # Benchmark inference
            execution_times = []
            memory_usage = self._get_memory_usage(model)
            
            for _ in range(50):
                start_time = time.time()
                with torch.no_grad():
                    _ = model(test_input)
                end_time = time.time()
                execution_times.append(end_time - start_time)
            
            mean_time = np.mean(execution_times)
            throughput = 1.0 / mean_time
            
            result = BenchmarkResult(
                model_name=f"{model_name}_{config}",
                task_name="scalability",
                execution_time=mean_time,
                throughput=throughput,
                memory_usage=memory_usage,
                additional_metrics={
                    "config": config,
                    "parameter_count": sum(p.numel() for p in model.parameters())
                }
            )
            
            scalability_results.append(result)
            self.results.append(result)
        
        return scalability_results
    
    def benchmark_sparsity_benefits(
        self,
        spiking_model: SpikingNeuralNetwork,
        dense_model: torch.nn.Module,
        test_data: List[torch.Tensor],
        sparsity_levels: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
    ) -> List[BenchmarkResult]:
        """Benchmark performance benefits of sparsity."""
        
        results = []
        
        for sparsity in sparsity_levels:
            # Generate sparse input
            sparse_inputs = []
            for data in test_data:
                sparse_input = data.clone()
                mask = torch.rand_like(sparse_input) < (1 - sparsity)
                sparse_input = sparse_input * mask
                sparse_inputs.append(sparse_input)
            
            # Benchmark spiking model
            spiking_result = self.benchmark_inference_speed(
                spiking_model, sparse_inputs, f"spiking_sparsity_{sparsity}"
            )
            
            # Benchmark dense model
            dense_result = self.benchmark_inference_speed(
                dense_model, sparse_inputs, f"dense_sparsity_{sparsity}"
            )
            
            # Calculate speedup
            speedup = dense_result.execution_time / spiking_result.execution_time
            
            comparison_result = BenchmarkResult(
                model_name=f"sparsity_comparison_{sparsity}",
                task_name="sparsity_benefits",
                execution_time=spiking_result.execution_time,
                throughput=spiking_result.throughput,
                memory_usage=spiking_result.memory_usage,
                additional_metrics={
                    "sparsity_level": sparsity,
                    "speedup": speedup,
                    "spiking_time": spiking_result.execution_time,
                    "dense_time": dense_result.execution_time
                }
            )
            
            results.append(comparison_result)
            self.results.append(comparison_result)
        
        return results
    
    def benchmark_event_driven_processing(
        self,
        processor: EventDrivenProcessor,
        input_patterns: List[List[tuple]],  # List of (neuron_id, current, timestamp)
        comparison_timesteps: int = 1000
    ) -> BenchmarkResult:
        """Benchmark event-driven vs synchronous processing."""
        
        total_speedups = []
        total_events = 0
        
        for pattern in input_patterns:
            # Reset processor
            processor.reset()
            
            # Add events
            for neuron_id, current, timestamp in pattern:
                processor.add_external_input(neuron_id, current, timestamp)
            
            # Benchmark event-driven processing
            max_time = max(timestamp for _, _, timestamp in pattern) + 100
            start_time = time.time()
            processor.process_until_time(max_time)
            event_time = time.time() - start_time
            
            # Get statistics
            stats = processor.get_statistics()
            speedup = stats.get("theoretical_speedup", 1.0)
            total_speedups.append(speedup)
            total_events += stats.get("total_events_processed", 0)
        
        mean_speedup = np.mean(total_speedups)
        
        result = BenchmarkResult(
            model_name="event_driven_processor",
            task_name="event_driven_processing",
            execution_time=0,  # Not directly comparable
            throughput=total_events / len(input_patterns),
            memory_usage=0,  # Would need to measure
            additional_metrics={
                "mean_theoretical_speedup": mean_speedup,
                "total_events_processed": total_events,
                "num_patterns": len(input_patterns)
            }
        )
        
        self.results.append(result)
        return result
    
    def benchmark_memory_efficiency(
        self,
        models: Dict[str, torch.nn.Module],
        input_sizes: List[tuple],
        sequence_lengths: List[int]
    ) -> List[BenchmarkResult]:
        """Benchmark memory efficiency across different input sizes."""
        
        results = []
        
        for model_name, model in models.items():
            model.to(self.device)
            
            for input_size in input_sizes:
                for seq_length in sequence_lengths:
                    # Create test input
                    test_input = torch.randn(1, *input_size, seq_length).to(self.device)
                    
                    # Measure peak memory usage
                    if self.device.startswith('cuda'):
                        torch.cuda.reset_peak_memory_stats()
                    
                    with torch.no_grad():
                        _ = model(test_input)
                    
                    if self.device.startswith('cuda'):
                        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                    else:
                        peak_memory = self._get_memory_usage(model)
                    
                    result = BenchmarkResult(
                        model_name=model_name,
                        task_name="memory_efficiency",
                        execution_time=0,
                        throughput=0,
                        memory_usage=peak_memory,
                        additional_metrics={
                            "input_size": input_size,
                            "sequence_length": seq_length,
                            "memory_per_element": peak_memory / (np.prod(input_size) * seq_length)
                        }
                    )
                    
                    results.append(result)
                    self.results.append(result)
        
        return results
    
    def _get_memory_usage(self, model: torch.nn.Module) -> float:
        """Estimate model memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size
        return total_size / 1024**2  # Convert to MB
    
    def generate_report(self) -> Dict:
        """Generate comprehensive benchmark report."""
        
        report = {
            "summary": {
                "total_benchmarks": len(self.results),
                "device": self.device,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "results_by_task": {},
            "results_by_model": {},
            "performance_rankings": {}
        }
        
        # Group results by task
        for result in self.results:
            task = result.task_name
            if task not in report["results_by_task"]:
                report["results_by_task"][task] = []
            report["results_by_task"][task].append(result.__dict__)
        
        # Group results by model
        for result in self.results:
            model = result.model_name
            if model not in report["results_by_model"]:
                report["results_by_model"][model] = []
            report["results_by_model"][model].append(result.__dict__)
        
        # Performance rankings
        for task in report["results_by_task"]:
            task_results = report["results_by_task"][task]
            
            # Rank by throughput (higher is better)
            if any(r["throughput"] > 0 for r in task_results):
                throughput_ranking = sorted(
                    task_results, 
                    key=lambda x: x["throughput"], 
                    reverse=True
                )
                report["performance_rankings"][f"{task}_throughput"] = [
                    {"model": r["model_name"], "throughput": r["throughput"]}
                    for r in throughput_ranking[:5]  # Top 5
                ]
            
            # Rank by execution time (lower is better)
            if any(r["execution_time"] > 0 for r in task_results):
                time_ranking = sorted(
                    task_results,
                    key=lambda x: x["execution_time"]
                )
                report["performance_rankings"][f"{task}_speed"] = [
                    {"model": r["model_name"], "execution_time": r["execution_time"]}
                    for r in time_ranking[:5]  # Top 5
                ]
        
        return report
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        import json
        
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Benchmark results saved to {filename}")
    
    def clear_results(self):
        """Clear all stored benchmark results."""
        self.results.clear()