"""Basic spiking neural network example."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.spiking_neural_network import SpikingNeuralNetwork
from algorithms.spike_processor import SpikeProcessor
from utils.logging import get_logger, log_performance


@log_performance
def basic_snn_example():
    """Demonstrate basic spiking neural network functionality."""
    
    logger = get_logger("examples.basic_snn")
    logger.info("Starting basic spiking neural network example")
    
    # Model configuration
    layer_sizes = [784, 256, 128, 10]  # MNIST-like architecture
    model = SpikingNeuralNetwork(
        layer_sizes=layer_sizes,
        neuron_params={
            "tau_mem": 20.0,
            "tau_syn": 5.0,  
            "v_thresh": -50.0,
            "v_reset": -70.0,
            "v_rest": -65.0
        },
        connection_prob=0.1,
        dt=0.1,
        encoding_method="poisson"
    )
    
    logger.info(f"Created SNN with architecture: {layer_sizes}")
    
    # Generate sample input data (simulating MNIST)
    batch_size = 4
    input_size = 784
    num_classes = 10
    
    # Random input patterns
    x = torch.randn(batch_size, input_size) * 2 + 5  # Positive rates
    x = torch.clamp(x, 0, 20)  # 0-20 Hz firing rates
    
    # Random targets
    y = torch.randint(0, num_classes, (batch_size,))
    y_one_hot = torch.nn.functional.one_hot(y, num_classes).float()
    
    logger.info(f"Generated test data: {x.shape} -> {y_one_hot.shape}")
    
    # Forward pass
    logger.info("Performing forward pass...")
    with torch.no_grad():
        output, spike_trains = model.forward(x, duration=100.0, return_spikes=True)
    
    logger.info(f"Forward pass complete. Output shape: {output.shape}")
    
    # Analyze spike activity
    total_spikes = sum(spikes.sum().item() for spikes in spike_trains)
    total_neurons = sum(spikes.shape[1] for spikes in spike_trains)
    total_time_steps = spike_trains[0].shape[-1]
    
    sparsity = 1.0 - (total_spikes / (total_neurons * total_time_steps * batch_size))
    
    logger.info(f"Network activity analysis:")
    logger.info(f"  Total spikes: {int(total_spikes)}")
    logger.info(f"  Network sparsity: {sparsity:.3f}")
    logger.info(f"  Spikes per neuron per ms: {total_spikes / (total_neurons * 100):.3f}")
    
    # Energy efficiency analysis
    energy_metrics = model.compute_energy_consumption(spike_trains)
    
    logger.info("Energy efficiency metrics:")
    logger.info(f"  Energy efficiency: {energy_metrics['energy_efficiency']:.3f}")
    logger.info(f"  Theoretical speedup: {energy_metrics['theoretical_speedup']:.2f}x")
    
    # Training step demonstration
    logger.info("Demonstrating training step...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    loss, metrics = model.train_step(x, y_one_hot, duration=100.0)
    
    logger.info(f"Training step complete:")
    logger.info(f"  Loss: {loss:.6f}")
    logger.info(f"  Energy efficiency: {metrics['energy_efficiency']:.3f}")
    
    # Spike pattern analysis
    spike_processor = SpikeProcessor()
    
    # Analyze first layer spikes
    first_layer_spikes = spike_trains[0]
    spike_metrics = spike_processor.compute_spike_train_metrics(first_layer_spikes)
    
    logger.info("First layer spike analysis:")
    logger.info(f"  Mean firing rate: {spike_metrics['mean_firing_rate']:.2f} Hz")
    logger.info(f"  Network sparsity: {spike_metrics['sparsity']:.3f}")
    
    if spike_metrics.get('mean_isi'):
        logger.info(f"  Mean ISI: {spike_metrics['mean_isi']:.2f} ms")
        logger.info(f"  ISI CV: {spike_metrics['cv_isi']:.3f}")
    
    # Visualization (save to file)
    logger.info("Creating visualizations...")
    
    # Raster plot of first layer
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    spike_processor.visualize_spike_trains(
        first_layer_spikes[0:1],  # First sample only
        save_path=str(output_dir / "snn_raster_plot.png"),
        max_neurons=50,
        time_range=(0, 100)
    )
    plt.close()
    
    # Activity over time
    plt.figure(figsize=(10, 6))
    network_activity = first_layer_spikes.sum(dim=1)  # Sum across neurons
    time_axis = torch.arange(network_activity.shape[-1]) * model.dt
    
    for i in range(min(batch_size, 4)):
        plt.plot(time_axis, network_activity[i], label=f'Sample {i+1}')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Network Activity (spikes/ms)')
    plt.title('Network Activity Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "network_activity.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")
    
    # Performance comparison with different sparsity levels
    logger.info("Comparing performance at different sparsity levels...")
    
    sparsity_levels = [0.1, 0.5, 0.9]
    results = []
    
    for sparsity in sparsity_levels:
        # Create sparse input
        sparse_x = x.clone()
        mask = torch.rand_like(sparse_x) < (1 - sparsity)
        sparse_x = sparse_x * mask
        
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            import time
            cpu_start = time.time()
            
            output, spikes = model.forward(sparse_x, duration=100.0, return_spikes=True)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                gpu_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            else:
                gpu_time = time.time() - cpu_start
            
            energy_metrics = model.compute_energy_consumption(spikes)
            
            results.append({
                'sparsity': sparsity,
                'execution_time': gpu_time,
                'total_spikes': sum(s.sum().item() for s in spikes),
                'energy_efficiency': energy_metrics['energy_efficiency'],
                'theoretical_speedup': energy_metrics['theoretical_speedup']
            })
    
    logger.info("Sparsity comparison results:")
    for result in results:
        logger.info(f"  Sparsity {result['sparsity']:.1f}: "
                   f"Time {result['execution_time']:.4f}s, "
                   f"Spikes {result['total_spikes']}, "
                   f"Speedup {result['theoretical_speedup']:.2f}x")
    
    logger.info("Basic spiking neural network example completed successfully!")
    
    return {
        'model': model,
        'results': results,
        'spike_trains': spike_trains,
        'output': output
    }


def main():
    """Run the basic SNN example."""
    try:
        results = basic_snn_example()
        print("\n=== Example completed successfully! ===")
        print(f"Generated {len(results['spike_trains'])} layer spike trains")
        print(f"Final output shape: {results['output'].shape}")
        print("Check 'outputs/' directory for visualizations")
        
    except Exception as e:
        print(f"Example failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()