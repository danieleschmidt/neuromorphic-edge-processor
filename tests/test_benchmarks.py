"""Tests for benchmark modules."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.performance_benchmarks import PerformanceBenchmark, BenchmarkResult
from benchmarks.energy_benchmarks import EnergyBenchmark, EnergyModel
from benchmarks.accuracy_benchmarks import AccuracyBenchmark
from models.spiking_neural_network import SpikingNeuralNetwork


class SimpleMockModel(torch.nn.Module):
    """Simple mock model for testing."""
    
    def __init__(self, input_size=10, output_size=3):
        super().__init__()
        self.fc = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        if x.dim() == 3:  # Handle sequence data
            x = x.mean(dim=-1)  # Average over time
        return self.fc(x)


class TestPerformanceBenchmark:
    """Test performance benchmarking functionality."""
    
    @pytest.fixture
    def benchmark(self):
        """Create benchmark instance."""
        return PerformanceBenchmark(device="cpu")
    
    @pytest.fixture
    def sample_model(self):
        """Create sample model for testing."""
        return SimpleMockModel(input_size=20, output_size=5)
    
    @pytest.fixture
    def test_data(self):
        """Create test data."""
        return [torch.randn(4, 20) for _ in range(5)]
    
    def test_benchmark_initialization(self, benchmark):
        """Test benchmark initialization."""
        assert benchmark.device == "cpu"
        assert len(benchmark.results) == 0
    
    def test_benchmark_inference_speed(self, benchmark, sample_model, test_data):
        """Test inference speed benchmarking."""
        result = benchmark.benchmark_inference_speed(
            model=sample_model,
            test_data=test_data,
            model_name="test_model",
            warmup_runs=2,
            benchmark_runs=5
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.model_name == "test_model"
        assert result.task_name == "inference_speed"
        assert result.execution_time > 0
        assert result.throughput > 0
        assert result.memory_usage > 0
        assert result.additional_metrics is not None
    
    def test_benchmark_memory_efficiency(self, benchmark):
        """Test memory efficiency benchmarking."""
        models = {
            "small_model": SimpleMockModel(10, 2),
            "large_model": SimpleMockModel(100, 20)
        }
        
        input_sizes = [(10,), (100,)]
        sequence_lengths = [10, 20]
        
        results = benchmark.benchmark_memory_efficiency(
            models=models,
            input_sizes=input_sizes,
            sequence_lengths=sequence_lengths
        )
        
        assert len(results) > 0
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.task_name == "memory_efficiency"
            assert result.memory_usage > 0
    
    def test_benchmark_scalability(self, benchmark):
        """Test scalability benchmarking."""
        model_configs = [
            {"input_size": 10, "output_size": 5},
            {"input_size": 50, "output_size": 10},
            {"input_size": 100, "output_size": 20}
        ]
        
        results = benchmark.benchmark_scalability(
            model_class=SimpleMockModel,
            model_configs=model_configs,
            test_input_shape=(10,),
            model_name="scalability_test"
        )
        
        assert len(results) == len(model_configs)
        
        # Check that larger models generally take more time
        execution_times = [r.execution_time for r in results]
        assert all(t > 0 for t in execution_times)
    
    def test_generate_report(self, benchmark, sample_model, test_data):
        """Test report generation."""
        # Add some benchmark results
        benchmark.benchmark_inference_speed(
            sample_model, test_data, "model1", warmup_runs=1, benchmark_runs=2
        )
        benchmark.benchmark_inference_speed(
            sample_model, test_data, "model2", warmup_runs=1, benchmark_runs=2
        )
        
        report = benchmark.generate_report()
        
        assert "summary" in report
        assert "results_by_task" in report
        assert "results_by_model" in report
        assert "performance_rankings" in report
        
        assert report["summary"]["total_benchmarks"] == 2
        assert report["summary"]["device"] == "cpu"
    
    def test_save_results(self, benchmark, sample_model, test_data):
        """Test saving results to file."""
        benchmark.benchmark_inference_speed(
            sample_model, test_data, "test_model", warmup_runs=1, benchmark_runs=2
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name
        
        try:
            benchmark.save_results(filename)
            
            # Check file was created and contains valid JSON
            with open(filename, 'r') as f:
                saved_data = json.load(f)
            
            assert "summary" in saved_data
            assert "results_by_task" in saved_data
            
        finally:
            Path(filename).unlink()


class TestEnergyBenchmark:
    """Test energy benchmarking functionality."""
    
    @pytest.fixture
    def energy_benchmark(self):
        """Create energy benchmark instance."""
        energy_model = EnergyModel(
            spike_energy=1.0,
            synapse_energy=0.1,
            neuron_idle_energy=0.01
        )
        return EnergyBenchmark(energy_model=energy_model)
    
    @pytest.fixture
    def spiking_model(self):
        """Create spiking neural network for testing."""
        return SpikingNeuralNetwork(
            layer_sizes=[10, 8, 5],
            dt=0.1,
            encoding_method="poisson"
        )
    
    def test_energy_model_initialization(self, energy_benchmark):
        """Test energy model initialization."""
        assert energy_benchmark.energy_model.spike_energy == 1.0
        assert energy_benchmark.energy_model.synapse_energy == 0.1
        assert energy_benchmark.energy_model.neuron_idle_energy == 0.01
    
    def test_spiking_network_energy_estimation(self, energy_benchmark, spiking_model):
        """Test spiking network energy estimation."""
        input_data = torch.rand(2, 10) * 5  # 0-5 Hz rates
        
        energy_info = energy_benchmark.estimate_spiking_network_energy(
            spiking_model, input_data, duration=20.0
        )
        
        required_keys = [
            "total_energy", "energy_breakdown", "total_spikes",
            "sparsity", "energy_per_spike", "energy_savings"
        ]
        
        for key in required_keys:
            assert key in energy_info
        
        assert energy_info["total_energy"] > 0
        assert 0 <= energy_info["sparsity"] <= 1
        assert energy_info["total_spikes"] >= 0
        assert energy_info["energy_per_spike"] > 0
    
    def test_energy_vs_accuracy_benchmark(self, energy_benchmark):
        """Test energy vs accuracy benchmarking."""
        models = {
            "model1": SimpleMockModel(10, 2),
            "model2": SimpleMockModel(20, 2)
        }
        
        test_data = [
            (torch.randn(4, 10), torch.randint(0, 2, (4,)).float()),
            (torch.randn(4, 20), torch.randint(0, 2, (4,)).float())
        ]
        
        def accuracy_fn(outputs, targets):
            predictions = (outputs > 0.5).float().squeeze()
            return (predictions == targets).float().mean().item()
        
        results = energy_benchmark.benchmark_energy_vs_accuracy(
            models=models,
            test_data=test_data,
            accuracy_fn=accuracy_fn
        )
        
        assert len(results) == len(models)
        for result in results:
            assert result.task_name == "energy_vs_accuracy"
            assert result.accuracy is not None
            assert result.energy_efficiency is not None
    
    def test_sparsity_energy_savings(self, energy_benchmark, spiking_model):
        """Test sparsity energy savings analysis."""
        sparsity_levels = [0.1, 0.5, 0.9]
        test_input = torch.rand(2, 10) * 8
        
        results = energy_benchmark.benchmark_sparsity_energy_savings(
            spiking_model=spiking_model,
            sparsity_levels=sparsity_levels,
            test_input=test_input
        )
        
        assert len(results) == len(sparsity_levels)
        
        for result in results:
            assert result.task_name == "sparsity_energy_savings"
            assert "sparsity_level" in result.additional_metrics
            assert "total_energy" in result.additional_metrics
            assert result.energy_efficiency is not None
    
    def test_generate_energy_report(self, energy_benchmark, spiking_model):
        """Test energy report generation."""
        # Add some benchmark results
        test_input = torch.rand(2, 10) * 5
        
        energy_benchmark.benchmark_sparsity_energy_savings(
            spiking_model, sparsity_levels=[0.5], test_input=test_input
        )
        
        report = energy_benchmark.generate_energy_report()
        
        assert "energy_model" in report
        assert "benchmark_summary" in report
        assert "efficiency_analysis" in report
        assert report["benchmark_summary"]["total_benchmarks"] > 0


class TestAccuracyBenchmark:
    """Test accuracy benchmarking functionality."""
    
    @pytest.fixture
    def accuracy_benchmark(self):
        """Create accuracy benchmark instance."""
        return AccuracyBenchmark()
    
    @pytest.fixture
    def classification_model(self):
        """Create classification model for testing."""
        return SimpleMockModel(input_size=20, output_size=3)
    
    @pytest.fixture
    def classification_data(self):
        """Create classification test data."""
        data = []
        for _ in range(5):
            inputs = torch.randn(4, 20)
            targets = torch.randint(0, 3, (4,))
            data.append((inputs, targets))
        return data
    
    def test_evaluate_classification(self, accuracy_benchmark, classification_model, classification_data):
        """Test classification evaluation."""
        result = accuracy_benchmark.evaluate_classification(
            model=classification_model,
            test_data=classification_data,
            model_name="test_classifier",
            num_classes=3
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.task_name == "classification_accuracy"
        assert result.model_name == "test_classifier"
        assert 0 <= result.accuracy <= 1
        
        # Check additional metrics
        assert "precision" in result.additional_metrics
        assert "recall" in result.additional_metrics
        assert "f1_score" in result.additional_metrics
        assert "confusion_matrix" in result.additional_metrics
        assert "num_classes" in result.additional_metrics
        
        assert result.additional_metrics["num_classes"] == 3
    
    def test_evaluate_robustness(self, accuracy_benchmark, classification_model, classification_data):
        """Test robustness evaluation."""
        noise_levels = [0.1, 0.2, 0.3]
        
        results = accuracy_benchmark.evaluate_robustness(
            model=classification_model,
            clean_test_data=classification_data,
            noise_levels=noise_levels,
            model_name="robust_test"
        )
        
        assert len(results) == len(noise_levels)
        
        for i, result in enumerate(results):
            assert result.task_name == "robustness_evaluation"
            assert "noise_level" in result.additional_metrics
            assert result.additional_metrics["noise_level"] == noise_levels[i]
            assert "accuracy_retention" in result.additional_metrics
            assert "robustness_score" in result.additional_metrics
    
    def test_temporal_sequence_prediction(self, accuracy_benchmark):
        """Test temporal sequence prediction evaluation."""
        # Create mock temporal model
        class TemporalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                if x.dim() == 3:
                    return self.fc(x.mean(dim=-1))  # Average over time
                return self.fc(x)
        
        model = TemporalModel()
        
        # Create temporal test data
        test_sequences = []
        for _ in range(3):
            input_seq = torch.randn(2, 10, 20)  # [batch, features, time]
            target_seq = torch.randn(2, 5, 20)  # [batch, features, time]
            test_sequences.append((input_seq, target_seq))
        
        result = accuracy_benchmark.evaluate_temporal_sequence_prediction(
            model=model,
            test_sequences=test_sequences,
            model_name="temporal_model",
            prediction_horizon=5
        )
        
        assert result.task_name == "temporal_prediction"
        assert result.accuracy is not None
        assert "mse" in result.additional_metrics
        assert "mae" in result.additional_metrics
        assert "temporal_correlation" in result.additional_metrics
    
    def test_generate_accuracy_report(self, accuracy_benchmark, classification_model, classification_data):
        """Test accuracy report generation."""
        # Add some evaluation results
        accuracy_benchmark.evaluate_classification(
            classification_model, classification_data, "model1", num_classes=3
        )
        accuracy_benchmark.evaluate_classification(
            classification_model, classification_data, "model2", num_classes=3
        )
        
        report = accuracy_benchmark.generate_accuracy_report()
        
        assert "summary" in report
        assert "accuracy_rankings" in report
        assert "task_analysis" in report
        
        assert report["summary"]["total_evaluations"] == 2
        assert len(report["summary"]["models_evaluated"]) == 2
    
    def test_save_accuracy_results(self, accuracy_benchmark, classification_model, classification_data):
        """Test saving accuracy results."""
        accuracy_benchmark.evaluate_classification(
            classification_model, classification_data, "test_model", num_classes=3
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name
        
        try:
            accuracy_benchmark.save_accuracy_results(filename)
            
            # Check file was created and contains valid JSON
            with open(filename, 'r') as f:
                saved_data = json.load(f)
            
            assert "summary" in saved_data
            assert "accuracy_rankings" in saved_data
            
        finally:
            Path(filename).unlink()


class TestBenchmarkIntegration:
    """Integration tests for benchmark modules."""
    
    def test_end_to_end_benchmarking(self):
        """Test complete benchmarking workflow."""
        # Create models
        simple_model = SimpleMockModel(15, 5)
        snn_model = SpikingNeuralNetwork([15, 10, 5], dt=0.1)
        
        # Create test data
        test_data = [torch.randn(3, 15) for _ in range(3)]
        classification_data = [
            (torch.randn(3, 15), torch.randint(0, 5, (3,))) for _ in range(3)
        ]
        
        # Performance benchmarking
        perf_benchmark = PerformanceBenchmark(device="cpu")
        perf_result = perf_benchmark.benchmark_inference_speed(
            simple_model, test_data, "simple_model", warmup_runs=1, benchmark_runs=2
        )
        
        # Energy benchmarking
        energy_benchmark = EnergyBenchmark()
        energy_result = energy_benchmark.estimate_spiking_network_energy(
            snn_model, torch.rand(2, 15) * 5, duration=15.0
        )
        
        # Accuracy benchmarking
        acc_benchmark = AccuracyBenchmark()
        acc_result = acc_benchmark.evaluate_classification(
            simple_model, classification_data, "simple_model", num_classes=5
        )
        
        # Verify all benchmarks completed successfully
        assert perf_result.execution_time > 0
        assert energy_result["total_energy"] > 0
        assert 0 <= acc_result.accuracy <= 1
    
    def test_benchmark_error_handling(self):
        """Test benchmark error handling."""
        benchmark = PerformanceBenchmark(device="cpu")
        
        # Test with None model (should raise error)
        with pytest.raises((AttributeError, TypeError)):
            benchmark.benchmark_inference_speed(
                None, [torch.randn(2, 10)], "none_model"
            )
        
        # Test with empty test data
        model = SimpleMockModel(10, 5)
        with pytest.raises((IndexError, ValueError)):
            benchmark.benchmark_inference_speed(
                model, [], "empty_data_model"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])