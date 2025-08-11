# 📚 Neuromorphic Edge Processor - API Reference

## Overview

This document provides comprehensive API documentation for the Neuromorphic Edge Processor, including all classes, methods, and functions with detailed examples.

## 🧠 Core Models

### SpikingNeuralNetwork

Multi-layer spiking neural network with temporal processing capabilities.

```python
class SpikingNeuralNetwork(nn.Module):
    def __init__(
        self,
        layer_sizes: List[int],
        neuron_model: str = "lif",
        tau_mem: float = 20.0,
        tau_syn: float = 5.0,
        dt: float = 1.0,
        learning_rule: str = "stdp",
        topology: str = "feedforward"
    )
```

**Parameters:**
- `layer_sizes`: List of layer sizes [input, hidden1, hidden2, ..., output]
- `neuron_model`: Type of neuron model ("lif", "adaptive_lif")
- `tau_mem`: Membrane time constant (ms)
- `tau_syn`: Synaptic time constant (ms)
- `dt`: Simulation time step (ms)
- `learning_rule`: Learning rule ("stdp", "r-stdp", "supervised")
- `topology`: Network topology ("feedforward", "recurrent", "reservoir")

**Methods:**

#### `forward(input_spikes: torch.Tensor) -> torch.Tensor`

Forward pass through the spiking network.

**Parameters:**
- `input_spikes`: Input spike trains [batch_size, input_size, time_steps]

**Returns:**
- Output spike trains [batch_size, output_size, time_steps]

**Example:**
```python
from neuromorphic_edge_processor import SpikingNeuralNetwork
import torch

# Create network
model = SpikingNeuralNetwork(
    layer_sizes=[784, 128, 10],
    learning_rule="stdp"
)

# Generate sample input
input_spikes = torch.rand(32, 784, 100) < 0.1  # 10% spike probability
input_spikes = input_spikes.float()

# Forward pass
output = model(input_spikes)
print(f"Output shape: {output.shape}")  # [32, 10, 100]
```

#### `train_step(input_data: torch.Tensor, targets: torch.Tensor) -> Tuple[float, Dict]`

Perform one training step with spike-based learning.

**Parameters:**
- `input_data`: Input spike trains [batch_size, input_size, time_steps]
- `targets`: Target patterns [batch_size, output_size]

**Returns:**
- Tuple of (loss, metrics_dict)

**Example:**
```python
# Training step
loss, metrics = model.train_step(input_spikes, targets)
print(f"Loss: {loss:.4f}, Firing rate: {metrics['output_firing_rate']:.2f}")
```

#### `get_layer_statistics() -> List[Dict]`

Get statistics for each layer.

**Returns:**
- List of dictionaries containing layer statistics

### LIFNeuron

Leaky Integrate-and-Fire neuron model optimized for edge deployment.

```python
class LIFNeuron(nn.Module):
    def __init__(
        self,
        input_size: int,
        tau_mem: float = 20.0,
        tau_syn: float = 5.0,
        v_thresh: float = 1.0,
        v_reset: float = 0.0,
        v_rest: float = 0.0,
        refractory_period: int = 2,
        dt: float = 1.0,
        learnable: bool = True
    )
```

**Parameters:**
- `input_size`: Number of input connections
- `tau_mem`: Membrane time constant (ms)
- `tau_syn`: Synaptic time constant (ms)
- `v_thresh`: Spike threshold voltage
- `v_reset`: Reset voltage after spike
- `v_rest`: Resting potential
- `refractory_period`: Refractory period (time steps)
- `dt`: Time step size (ms)
- `learnable`: Whether parameters are learnable

**Example:**
```python
from neuromorphic_edge_processor import LIFNeuron

# Create LIF neuron
neuron = LIFNeuron(
    input_size=100,
    tau_mem=20.0,
    v_thresh=1.0
)

# Process spike train
input_spikes = torch.rand(1, 100, 200) < 0.05
output_spikes, membrane_potentials = neuron(input_spikes)
```

### LiquidStateMachine

Liquid State Machine for temporal pattern processing.

```python
class LiquidStateMachine(nn.Module):
    def __init__(
        self,
        input_size: int,
        reservoir_size: int = 100,
        output_size: int = 10,
        connectivity: float = 0.1,
        spectral_radius: float = 0.9,
        input_scaling: float = 1.0,
        dt: float = 1.0,
        readout_type: str = "linear"
    )
```

**Parameters:**
- `input_size`: Number of input features
- `reservoir_size`: Number of neurons in reservoir
- `output_size`: Number of output classes/features
- `connectivity`: Connection probability in reservoir [0,1]
- `spectral_radius`: Spectral radius of reservoir matrix
- `input_scaling`: Scaling factor for input connections
- `dt`: Time step size (ms)
- `readout_type`: Type of readout ("linear", "svm", "ridge")

**Example:**
```python
from neuromorphic_edge_processor import LiquidStateMachine

# Create LSM
lsm = LiquidStateMachine(
    input_size=784,
    reservoir_size=200,
    output_size=10,
    connectivity=0.15
)

# Process temporal data
input_sequence = torch.randn(16, 784, 100)
output = lsm(input_sequence)
```

## 🧮 Algorithms

### SpikeProcessor

Advanced spike processing and analysis for neuromorphic systems.

```python
class SpikeProcessor:
    def __init__(self, sampling_rate: float = 1000.0)
```

**Methods:**

#### `encode_rate_to_spikes(rates: torch.Tensor, duration: float, method: str = "poisson") -> torch.Tensor`

Convert firing rates to spike trains.

**Parameters:**
- `rates`: Firing rates [batch_size, num_neurons] (Hz)
- `duration`: Spike train duration (ms)
- `method`: Encoding method ("poisson", "regular", "temporal")

**Returns:**
- Spike trains [batch_size, num_neurons, time_steps]

**Example:**
```python
from neuromorphic_edge_processor.algorithms import SpikeProcessor

processor = SpikeProcessor(sampling_rate=1000.0)

# Convert rates to spikes
rates = torch.tensor([[10.0, 20.0, 5.0]])  # Hz
spikes = processor.encode_rate_to_spikes(rates, duration=100.0, method="poisson")
print(f"Generated spikes shape: {spikes.shape}")
```

#### `decode_spikes_to_rate(spikes: torch.Tensor, window_size: float = 50.0) -> torch.Tensor`

Decode spike trains to firing rates using sliding window.

#### `compute_spike_train_metrics(spikes: torch.Tensor) -> Dict`

Compute various metrics for spike train analysis.

### Novel STDP Algorithms

#### StabilizedSupervisedSTDP

Stabilized Supervised STDP (S2-STDP) implementation based on 2024 research.

```python
class StabilizedSupervisedSTDP:
    def __init__(self, config: STDPConfig)
```

**Example:**
```python
from neuromorphic_edge_processor.algorithms.novel_stdp import StabilizedSupervisedSTDP, STDPConfig

# Configure STDP
config = STDPConfig(
    tau_plus=20.0,
    tau_minus=20.0,
    a_plus=0.01,
    a_minus=0.01,
    learning_rate=0.001
)

# Create STDP learner
stdp = StabilizedSupervisedSTDP(config)

# Initialize traces
stdp.initialize_traces(batch_size=32, num_pre=100, num_post=10, time_steps=200)
```

#### BatchedSTDP

Samples Temporal Batch STDP for accelerated learning.

```python
class BatchedSTDP:
    def __init__(self, config: STDPConfig, batch_accumulation: int = 10)
```

## 🔧 Utilities and Tools

### Benchmarking

#### PerformanceBenchmark

Comprehensive performance benchmarking suite.

```python
class PerformanceBenchmark:
    def __init__(self, device: str = "cpu")
```

**Methods:**

#### `benchmark_inference_speed(model, test_data, model_name, warmup_runs=10, benchmark_runs=100)`

Benchmark inference speed of a model.

**Example:**
```python
from neuromorphic_edge_processor.benchmarks import PerformanceBenchmark

benchmark = PerformanceBenchmark(device="cuda")

# Benchmark model
result = benchmark.benchmark_inference_speed(
    model=my_model,
    test_data=test_inputs,
    model_name="spiking_nn",
    benchmark_runs=50
)

print(f"Average inference time: {result.execution_time*1000:.2f}ms")
print(f"Throughput: {result.throughput:.1f} samples/sec")
```

### Security

#### SecurityManager

Comprehensive security manager for neuromorphic operations.

```python
class SecurityManager:
    def __init__(self, config: Optional[SecurityConfig] = None)
```

**Methods:**

#### `validate_input(data: torch.Tensor, source: str = "unknown") -> bool`

Validate input data for security compliance.

**Example:**
```python
from neuromorphic_edge_processor.security import SecurityManager, SecurityConfig

# Configure security
config = SecurityConfig(
    max_input_size=10000,
    max_memory_mb=1000.0,
    enable_input_sanitization=True
)

security = SecurityManager(config)

# Validate input
input_data = torch.randn(32, 784, 100)
if security.validate_input(input_data, "inference"):
    print("Input validation passed")
else:
    print("Input validation failed")
```

### Optimization

#### NeuromorphicOptimizer

Main optimization coordinator for neuromorphic computing at scale.

```python
class NeuromorphicOptimizer:
    def __init__(self, config: Optional[OptimizationConfig] = None)
```

**Methods:**

#### `optimize_model(model: nn.Module) -> nn.Module`

Apply optimization techniques to neuromorphic model.

**Example:**
```python
from neuromorphic_edge_processor.optimization import NeuromorphicOptimizer, OptimizationConfig

# Configure optimizer
opt_config = OptimizationConfig(
    enable_caching=True,
    enable_sparsity_optimization=True,
    enable_quantization=True,
    optimization_level="aggressive"
)

optimizer = NeuromorphicOptimizer(opt_config)

# Optimize model
optimized_model = optimizer.optimize_model(original_model)
```

### Monitoring

#### AdvancedMonitor

Advanced monitoring system for neuromorphic edge processing.

```python
class AdvancedMonitor:
    def __init__(
        self,
        monitoring_interval: float = 1.0,
        history_size: int = 10000,
        alert_thresholds: Optional[Dict[str, float]] = None
    )
```

**Methods:**

#### `start_monitoring()`

Start continuous monitoring in background thread.

#### `record_inference(inference_time_ms, input_shape, output_shape=None, accuracy=None)`

Record inference performance metrics.

**Example:**
```python
from neuromorphic_edge_processor.monitoring import AdvancedMonitor

monitor = AdvancedMonitor()
monitor.start_monitoring()

# Record inference
monitor.record_inference(
    inference_time_ms=25.5,
    input_shape=(32, 784, 100),
    accuracy=0.87
)

# Get performance summary
summary = monitor.get_performance_summary(window_minutes=5)
print(f"Average inference time: {summary['inference_time_ms']['mean']:.2f}ms")
```

## 🧪 Research Framework

### ExperimentFramework

Advanced research framework for conducting neuromorphic experiments.

```python
class ResearchFramework:
    def __init__(self, experiment_config: ExperimentConfig)
```

**Example:**
```python
from neuromorphic_edge_processor.experiments import ResearchFramework, ExperimentConfig

# Configure experiment
config = ExperimentConfig(
    experiment_name="stdp_comparison",
    description="Compare STDP variants",
    dataset="spike_patterns",
    model_type="spiking_neural_network",
    learning_rule="s2-stdp",
    num_trials=5
)

# Run experiment
framework = ResearchFramework(config)
results = framework.run_experiment()
```

### StatisticalAnalyzer

Statistical analysis and validation for neuromorphic research.

```python
class StatisticalAnalyzer:
    def __init__(self, alpha: float = 0.05, random_seed: int = 42)
```

**Methods:**

#### `compare_two_groups(group1_data, group2_data, paired=False)`

Compare two groups using appropriate statistical tests.

#### `validate_model_performance(model_results, metrics=["accuracy", "precision", "recall"])`

Validate and compare model performance across metrics.

**Example:**
```python
from neuromorphic_edge_processor.validation import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(alpha=0.05)

# Compare model performance
model_results = {
    "spiking_nn": {"accuracy": [0.85, 0.87, 0.86]},
    "lstm_baseline": {"accuracy": [0.82, 0.83, 0.81]}
}

comparison = analyzer.validate_model_performance(model_results)
print(f"Best model: {comparison.best_model}")
```

## 🛠️ CLI Interface

### neuromorphic-benchmark

Command-line benchmarking tool.

```bash
# Basic usage
neuromorphic-benchmark --benchmark performance --model spiking --device cuda

# Full comparison suite
neuromorphic-benchmark --benchmark comparison --output results/ --save-results

# Custom configuration
neuromorphic-benchmark --config my_config.json --verbose
```

**Options:**
- `--benchmark`: Type of benchmark ("performance", "accuracy", "energy", "comparison", "all")
- `--model`: Model type ("spiking", "lsm", "reservoir", "all")
- `--device`: Device ("cpu", "cuda", "auto")
- `--config`: JSON configuration file
- `--output`: Output directory
- `--save-results`: Save results to files
- `--verbose`: Enable verbose output

## 📊 Data Types and Structures

### Configuration Classes

#### ExperimentConfig

```python
@dataclass
class ExperimentConfig:
    experiment_name: str
    description: str
    dataset: str
    model_type: str
    learning_rule: str
    input_size: int = 784
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])
    output_size: int = 10
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    # ... more fields
```

#### BenchmarkResult

```python
@dataclass
class BenchmarkResult:
    model_name: str
    task_name: str
    execution_time: float
    throughput: float
    memory_usage: float
    accuracy: Optional[float] = None
    energy_efficiency: Optional[float] = None
    additional_metrics: Optional[Dict] = None
```

### Error Types

#### SecurityError

Custom exception for security-related errors.

```python
class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass
```

## 🔗 Integration Examples

### Flask API Integration

```python
from flask import Flask, request, jsonify
from neuromorphic_edge_processor import SpikingNeuralNetwork
from neuromorphic_edge_processor.security import get_security_manager

app = Flask(__name__)
model = SpikingNeuralNetwork([784, 128, 10])
security = get_security_manager()

@app.route('/inference', methods=['POST'])
def inference():
    try:
        # Get input data
        data = torch.tensor(request.json['data'])
        
        # Security validation
        if not security.validate_input(data, "api_inference"):
            return jsonify({"error": "Input validation failed"}), 400
        
        # Run inference
        with torch.no_grad():
            output = model(data)
        
        return jsonify({
            "prediction": output.tolist(),
            "shape": list(output.shape)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

app = FastAPI()

class InferenceRequest(BaseModel):
    data: List[List[float]]
    model_type: str = "spiking"

@app.post("/inference")
async def inference(request: InferenceRequest):
    try:
        input_tensor = torch.tensor(request.data)
        
        # Load appropriate model
        if request.model_type == "spiking":
            model = load_spiking_model()
        elif request.model_type == "lsm":
            model = load_lsm_model()
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        # Run inference
        output = model(input_tensor)
        
        return {
            "prediction": output.tolist(),
            "model_type": request.model_type,
            "inference_time_ms": 25.0  # Would measure actual time
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 🎯 Best Practices

### Model Initialization

```python
# Best practice: Use configuration objects
config = ExperimentConfig(
    experiment_name="production_model",
    model_type="spiking_neural_network",
    input_size=784,
    hidden_sizes=[256, 128],
    output_size=10
)

model = create_model_from_config(config)
```

### Error Handling

```python
# Best practice: Comprehensive error handling
try:
    result = model(input_data)
except SecurityError as e:
    logger.error(f"Security violation: {e}")
    raise HTTPException(status_code=403, detail="Security validation failed")
except torch.cuda.OutOfMemoryError:
    logger.error("GPU out of memory")
    # Fallback to CPU or reduce batch size
    model = model.cpu()
    result = model(input_data.cpu())
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise
```

### Resource Management

```python
# Best practice: Context managers for resources
from neuromorphic_edge_processor.monitoring import AdvancedMonitor

class ModelInference:
    def __init__(self, model):
        self.model = model
        self.monitor = AdvancedMonitor()
    
    def __enter__(self):
        self.monitor.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop_monitoring()
        return False
    
    def infer(self, data):
        start_time = time.time()
        result = self.model(data)
        inference_time = (time.time() - start_time) * 1000
        
        self.monitor.record_inference(
            inference_time_ms=inference_time,
            input_shape=data.shape
        )
        
        return result

# Usage
with ModelInference(model) as inference_engine:
    result = inference_engine.infer(input_data)
```

## 📖 Version History

- **v0.1.0**: Initial release with basic SNN, LSM, and benchmarking
- **v0.2.0**: Added advanced STDP algorithms and optimization framework
- **v0.3.0**: Comprehensive security, monitoring, and research tools

For detailed changelog, see [CHANGELOG.md](CHANGELOG.md).

## 📞 Support

For API questions and support:
- **Documentation**: [GitHub Wiki](https://github.com/danieleschmidt/neuromorphic-edge-processor/wiki)
- **API Issues**: [GitHub Issues](https://github.com/danieleschmidt/neuromorphic-edge-processor/issues) (label: `api`)
- **Examples**: [examples/](examples/) directory

---

*This API reference is auto-generated from docstrings. For the most up-to-date information, refer to the source code.*