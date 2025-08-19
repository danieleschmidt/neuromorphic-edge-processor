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
        membrane_tau=20.0,
        threshold=1.0,
        learning_rate=0.001,
        stdp_enabled=True,
        adaptive_threshold=True
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
        output, network_states = model.forward(x)
        spike_trains = network_states['layer_outputs'] if isinstance(network_states['layer_outputs'], list) else [network_states['layer_outputs']]
    
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
    energy_consumption = network_states.get('energy_consumption', 0.0)
    
    logger.info("Energy efficiency metrics:")
    logger.info(f"  Total energy consumption: {energy_consumption:.3f} mJ")
    logger.info(f"  Network sparsity: {network_states.get('sparsity', 0.0):.3f}")
    
    # Training step demonstration
    logger.info("Demonstrating training step...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Simple training step simulation
    model.train()
    optimizer.zero_grad()
    output, states = model.forward(x)
    
    # Simple MSE loss for demonstration
    loss = torch.nn.functional.mse_loss(output, y_one_hot)
    loss.backward()
    optimizer.step()
    
    logger.info(f"Training step complete:")
    logger.info(f"  Loss: {loss.item():.6f}")
    logger.info(f"  Output spikes: {output.sum().item():.1f}")
    
    # Spike pattern analysis
    spike_processor = SpikeProcessor()
    
    # Analyze first layer spikes
    first_layer_spikes = spike_trains[0] if spike_trains else output
    
    logger.info("First layer spike analysis:")
    logger.info(f"  Mean spike rate: {first_layer_spikes.mean().item():.4f}")
    logger.info(f"  Max spike value: {first_layer_spikes.max().item():.4f}")
    logger.info(f"  Sparsity level: {(first_layer_spikes == 0).float().mean().item():.3f}")
    
    # Visualization (save to file)
    logger.info("Creating visualizations...")
    
    # Create simple spike visualization
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    # Visualize spike pattern as heatmap
    if first_layer_spikes.dim() >= 2:
        spike_data = first_layer_spikes[:4].detach().cpu().numpy()  # First 4 samples
        plt.imshow(spike_data, aspect='auto', cmap='hot', interpolation='nearest')
        plt.colorbar(label='Spike Value')
        plt.title('Spike Activity Pattern')
        plt.xlabel('Neuron Index')
        plt.ylabel('Sample Index')
    plt.savefig(output_dir / "snn_spike_pattern.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Activity summary plot
    plt.figure(figsize=(10, 6))
    if first_layer_spikes.dim() >= 2:
        activity_per_sample = first_layer_spikes.sum(dim=1).detach().cpu().numpy()
        sample_indices = range(len(activity_per_sample))
        
        plt.bar(sample_indices, activity_per_sample, alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Total Spike Activity')
        plt.title('Network Activity per Sample')
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
            
            output, states = model.forward(sparse_x)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                gpu_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            else:
                gpu_time = time.time() - cpu_start
            
            total_spikes = output.sum().item()
            energy = states.get('energy_consumption', 0.0)
            
            results.append({
                'sparsity': sparsity,
                'execution_time': gpu_time,
                'total_spikes': total_spikes,
                'energy_consumption': energy,
                'network_sparsity': states.get('sparsity', 0.0)
            })
    
    logger.info("Sparsity comparison results:")
    for result in results:
        logger.info(f"  Sparsity {result['sparsity']:.1f}: "
                   f"Time {result['execution_time']:.4f}s, "
                   f"Spikes {result['total_spikes']:.1f}, "
                   f"Energy {result['energy_consumption']:.3f}mJ")
    
    logger.info("Basic spiking neural network example completed successfully!")
    
    return {
        'model': model,
        'results': results,
        'network_states': network_states,
        'output': output
    }


def main():
    """Run the basic SNN example."""
    try:
        results = basic_snn_example()
        print("\n=== Example completed successfully! ===")
        print(f"Network processed successfully with {len(results['results'])} sparsity tests")
        print(f"Final output shape: {results['output'].shape}")
        print("Check 'outputs/' directory for visualizations")
        
    except Exception as e:
        print(f"Example failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()