"""Command line interface for neuromorphic benchmarking."""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from src.models.spiking_neural_network import SpikingNeuralNetwork
from src.models.liquid_state_machine import LiquidStateMachine
from src.models.reservoir_computer import ReservoirComputer
from src.algorithms.spike_processor import SpikeProcessor
from src.algorithms.event_processor import EventDrivenProcessor
from benchmarks.performance_benchmarks import PerformanceBenchmark
from benchmarks.accuracy_benchmarks import AccuracyBenchmark
from benchmarks.energy_benchmarks import EnergyBenchmark
from benchmarks.comparison_suite import ComparisonSuite


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Neuromorphic Edge Processor Benchmarking Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run performance benchmarks
  neuromorphic-benchmark --benchmark performance --model spiking --device cpu
  
  # Run accuracy benchmarks with custom config
  neuromorphic-benchmark --benchmark accuracy --config my_config.json
  
  # Run full comparison suite
  neuromorphic-benchmark --benchmark comparison --output results/
  
  # Run energy benchmarks on GPU
  neuromorphic-benchmark --benchmark energy --device cuda --save-results
        """
    )
    
    # Main benchmark type
    parser.add_argument(
        "--benchmark", "-b",
        choices=["performance", "accuracy", "energy", "comparison", "all"],
        default="performance",
        help="Type of benchmark to run (default: performance)"
    )
    
    # Model selection
    parser.add_argument(
        "--model", "-m",
        choices=["spiking", "lsm", "reservoir", "all"],
        default="spiking",
        help="Model type to benchmark (default: spiking)"
    )
    
    # Device selection
    parser.add_argument(
        "--device", "-d",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to run benchmarks on (default: auto)"
    )
    
    # Configuration file
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="JSON configuration file path"
    )
    
    # Output directory
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )
    
    # Save results flag
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save benchmark results to files"
    )
    
    # Verbose output
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Quick mode (reduced test sizes)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks with reduced test sizes"
    )
    
    # Custom parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for benchmarks (default: 32)"
    )
    
    parser.add_argument(
        "--input-size",
        type=int,
        default=784,
        help="Input size for models (default: 784)"
    )
    
    parser.add_argument(
        "--reservoir-size",
        type=int,
        default=100,
        help="Reservoir size for LSM/reservoir models (default: 100)"
    )
    
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=100,
        help="Sequence length for temporal data (default: 100)"
    )
    
    parser.add_argument(
        "--num-trials",
        type=int,
        default=10,
        help="Number of benchmark trials (default: 10)"
    )
    
    return parser


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded configuration from {config_path}")
        return config
    except Exception as e:
        print(f"✗ Error loading config {config_path}: {e}")
        sys.exit(1)


def setup_device(device_arg: str) -> str:
    """Setup and validate compute device."""
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    print(f"🔧 Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return device


def create_model(model_type: str, config: Dict[str, Any], device: str):
    """Create model instance based on type and configuration."""
    model_config = config.get('model_config', {})
    
    if model_type == "spiking":
        model = SpikingNeuralNetwork(
            layer_sizes=model_config.get('layer_sizes', [config['input_size'], 128, 10]),
            tau_mem=model_config.get('tau_mem', 20.0),
            tau_syn=model_config.get('tau_syn', 5.0),
            dt=model_config.get('dt', 1.0)
        )
    
    elif model_type == "lsm":
        model = LiquidStateMachine(
            input_size=config['input_size'],
            reservoir_size=config['reservoir_size'],
            output_size=model_config.get('output_size', 10),
            connectivity=model_config.get('connectivity', 0.1),
            spectral_radius=model_config.get('spectral_radius', 0.9)
        )
    
    elif model_type == "reservoir":
        model = ReservoirComputer(
            input_size=config['input_size'],
            reservoir_size=config['reservoir_size'],
            output_size=model_config.get('output_size', 10),
            reservoir_type=model_config.get('reservoir_type', 'esn'),
            sparsity=model_config.get('sparsity', 0.1),
            spectral_radius=model_config.get('spectral_radius', 0.9)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.to(device)
    return model


def generate_test_data(config: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    """Generate test data for benchmarking."""
    batch_size = config['batch_size']
    input_size = config['input_size']
    sequence_length = config['sequence_length']
    
    # Generate spike train data
    spike_data = torch.rand(batch_size, input_size, sequence_length) < 0.1
    spike_data = spike_data.float().to(device)
    
    # Generate continuous data
    continuous_data = torch.randn(batch_size, input_size, sequence_length).to(device)
    
    # Generate labels
    labels = torch.randint(0, 10, (batch_size,)).to(device)
    
    return {
        'spikes': spike_data,
        'continuous': continuous_data,
        'labels': labels
    }


def run_performance_benchmarks(models: Dict[str, Any], test_data: Dict[str, torch.Tensor], 
                             config: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Run performance benchmarks."""
    print("\n🚀 Running Performance Benchmarks")
    print("=" * 50)
    
    benchmark = PerformanceBenchmark(device=device)
    results = {}
    
    for model_name, model in models.items():
        print(f"\n📊 Benchmarking {model_name} model...")
        
        # Inference speed benchmark
        try:
            test_inputs = [test_data['spikes'][:config['batch_size']//4] for _ in range(4)]
            
            result = benchmark.benchmark_inference_speed(
                model=model,
                test_data=test_inputs,
                model_name=model_name,
                warmup_runs=config.get('warmup_runs', 5),
                benchmark_runs=config.get('benchmark_runs', config['num_trials'])
            )
            
            results[f"{model_name}_inference"] = result
            
            print(f"  ✓ Inference time: {result.execution_time*1000:.2f} ms")
            print(f"  ✓ Throughput: {result.throughput:.1f} samples/sec")
            print(f"  ✓ Memory usage: {result.memory_usage:.1f} MB")
            
        except Exception as e:
            print(f"  ✗ Inference benchmark failed: {e}")
        
        # Memory efficiency benchmark
        try:
            memory_results = benchmark.benchmark_memory_efficiency(
                models={model_name: model},
                input_sizes=[(config['input_size'],)],
                sequence_lengths=[config['sequence_length']]
            )
            
            if memory_results:
                results[f"{model_name}_memory"] = memory_results[0]
                print(f"  ✓ Peak memory: {memory_results[0].memory_usage:.1f} MB")
                
        except Exception as e:
            print(f"  ✗ Memory benchmark failed: {e}")
    
    return results


def run_accuracy_benchmarks(models: Dict[str, Any], test_data: Dict[str, torch.Tensor],
                          config: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Run accuracy benchmarks."""
    print("\n🎯 Running Accuracy Benchmarks")
    print("=" * 50)
    
    benchmark = AccuracyBenchmark(device=device)
    results = {}
    
    # Create synthetic datasets for different tasks
    tasks = {
        'classification': {
            'inputs': test_data['spikes'],
            'targets': test_data['labels']
        },
        'pattern_recognition': {
            'inputs': test_data['continuous'],
            'targets': torch.randint(0, 2, (config['batch_size'],)).to(device)
        }
    }
    
    for model_name, model in models.items():
        print(f"\n🎯 Testing {model_name} model accuracy...")
        
        for task_name, task_data in tasks.items():
            try:
                accuracy = benchmark.evaluate_model_accuracy(
                    model=model,
                    test_inputs=task_data['inputs'],
                    test_targets=task_data['targets'],
                    model_name=f"{model_name}_{task_name}"
                )
                
                results[f"{model_name}_{task_name}"] = accuracy
                print(f"  ✓ {task_name} accuracy: {accuracy:.3f}")
                
            except Exception as e:
                print(f"  ✗ {task_name} accuracy test failed: {e}")
    
    return results


def run_energy_benchmarks(models: Dict[str, Any], test_data: Dict[str, torch.Tensor],
                        config: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Run energy efficiency benchmarks."""
    print("\n⚡ Running Energy Benchmarks")  
    print("=" * 50)
    
    benchmark = EnergyBenchmark(device=device)
    results = {}
    
    for model_name, model in models.items():
        print(f"\n⚡ Measuring {model_name} energy consumption...")
        
        try:
            energy_result = benchmark.measure_energy_consumption(
                model=model,
                test_input=test_data['spikes'],
                model_name=model_name,
                num_iterations=config.get('energy_iterations', 100)
            )
            
            results[f"{model_name}_energy"] = energy_result
            
            print(f"  ✓ Energy per inference: {energy_result.energy_per_inference:.6f} J")
            print(f"  ✓ Power consumption: {energy_result.average_power:.3f} W")
            print(f"  ✓ Energy efficiency: {energy_result.efficiency_metric:.1f} inferences/J")
            
        except Exception as e:
            print(f"  ✗ Energy benchmark failed: {e}")
    
    return results


def run_comparison_suite(models: Dict[str, Any], test_data: Dict[str, torch.Tensor],
                       config: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Run comprehensive comparison suite."""
    print("\n📈 Running Comparison Suite")
    print("=" * 50)
    
    comparison = ComparisonSuite(device=device)
    results = {}
    
    try:
        # Cross-model comparison
        comparison_result = comparison.compare_models(
            models=models,
            test_inputs=[test_data['spikes']],
            comparison_name="neuromorphic_models",
            metrics=['performance', 'accuracy', 'energy']
        )
        
        results['model_comparison'] = comparison_result
        
        print("✓ Model comparison completed")
        
        # Display summary
        if 'summary' in comparison_result:
            summary = comparison_result['summary']
            print(f"\n📊 Performance Summary:")
            print(f"  Best throughput: {summary.get('best_throughput', 'N/A')}")
            print(f"  Lowest latency: {summary.get('lowest_latency', 'N/A')}")
            print(f"  Best accuracy: {summary.get('best_accuracy', 'N/A')}")
            print(f"  Most efficient: {summary.get('most_efficient', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Comparison suite failed: {e}")
    
    return results


def save_results(results: Dict[str, Any], output_dir: str, benchmark_type: str):
    """Save benchmark results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_file = output_path / f"{benchmark_type}_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {json_file}")
    
    # Generate summary report
    report_file = output_path / f"{benchmark_type}_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(f"Neuromorphic Benchmarking Report\n")
        f.write(f"{'=' * 40}\n")
        f.write(f"Benchmark Type: {benchmark_type}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Results: {len(results)}\n\n")
        
        for key, result in results.items():
            f.write(f"{key}:\n")
            if hasattr(result, '__dict__'):
                for attr, value in result.__dict__.items():
                    f.write(f"  {attr}: {value}\n")
            else:
                f.write(f"  {result}\n")
            f.write("\n")
    
    print(f"✓ Report saved to {report_file}")


def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    print("🧠 Neuromorphic Edge Processor Benchmarking Suite")
    print("=" * 60)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = {}
    
    # Override config with command line arguments
    config.update({
        'batch_size': args.batch_size,
        'input_size': args.input_size,
        'reservoir_size': args.reservoir_size,
        'sequence_length': args.sequence_length,
        'num_trials': args.num_trials,
        'quick_mode': args.quick
    })
    
    if args.quick:
        print("⚡ Quick mode enabled - reducing test sizes")
        config.update({
            'num_trials': max(3, args.num_trials // 3),
            'batch_size': max(8, args.batch_size // 4),
            'sequence_length': max(50, args.sequence_length // 2)
        })
    
    # Setup device
    device = setup_device(args.device)
    
    # Create models
    models = {}
    model_types = [args.model] if args.model != "all" else ["spiking", "lsm", "reservoir"]
    
    print(f"\n🔧 Creating models: {model_types}")
    for model_type in model_types:
        try:
            model = create_model(model_type, config, device)
            models[model_type] = model
            print(f"  ✓ {model_type} model created")
        except Exception as e:
            print(f"  ✗ Failed to create {model_type} model: {e}")
    
    if not models:
        print("✗ No models created successfully")
        sys.exit(1)
    
    # Generate test data
    print(f"\n📊 Generating test data...")
    test_data = generate_test_data(config, device)
    print(f"  ✓ Generated data shapes: {[(k, v.shape) for k, v in test_data.items()]}")
    
    # Run benchmarks
    all_results = {}
    
    if args.benchmark in ["performance", "all"]:
        perf_results = run_performance_benchmarks(models, test_data, config, device)
        all_results.update(perf_results)
    
    if args.benchmark in ["accuracy", "all"]:
        acc_results = run_accuracy_benchmarks(models, test_data, config, device)
        all_results.update(acc_results)
    
    if args.benchmark in ["energy", "all"]:
        energy_results = run_energy_benchmarks(models, test_data, config, device)
        all_results.update(energy_results)
    
    if args.benchmark in ["comparison", "all"]:
        comp_results = run_comparison_suite(models, test_data, config, device)
        all_results.update(comp_results)
    
    # Save results if requested
    if args.save_results or args.output != "benchmark_results":
        save_results(all_results, args.output, args.benchmark)
    
    print(f"\n✅ Benchmarking completed successfully!")
    print(f"📈 Total results collected: {len(all_results)}")
    
    if args.verbose:
        print(f"\n📊 Detailed Results:")
        print("=" * 40)
        for key, result in all_results.items():
            print(f"\n{key}:")
            if hasattr(result, '__dict__'):
                for attr, value in result.__dict__.items():
                    print(f"  {attr}: {value}")
            else:
                print(f"  {result}")


if __name__ == "__main__":
    main()