"""Liquid State Machine demonstration and analysis."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.liquid_state_machine import LiquidStateMachine
from algorithms.spike_processor import SpikeProcessor
from utils.logging import get_logger, log_performance


@log_performance
def lsm_demo():
    """Demonstrate Liquid State Machine capabilities."""
    
    logger = get_logger("examples.lsm_demo")
    logger.info("Starting Liquid State Machine demonstration")
    
    # LSM Configuration
    lsm = LiquidStateMachine(
        input_size=50,
        reservoir_size=1000,
        output_size=10,
        connection_prob=0.1,
        spectral_radius=0.9,
        input_scaling=1.0,
        dt=0.1
    )
    
    logger.info("Created LSM with 1000 reservoir neurons")
    
    # Generate test temporal sequences
    def generate_temporal_pattern(seq_length: int, pattern_type: str) -> torch.Tensor:
        """Generate different types of temporal patterns."""
        
        if pattern_type == "sine_wave":
            t = torch.linspace(0, 4*np.pi, seq_length)
            pattern = torch.sin(t).unsqueeze(0).expand(50, -1)
            # Add some noise
            pattern += torch.randn_like(pattern) * 0.1
            
        elif pattern_type == "chirp":
            t = torch.linspace(0, 1, seq_length)
            freq = 1 + 10 * t  # Frequency increases over time
            pattern = torch.sin(2 * np.pi * freq * t)
            pattern = pattern.unsqueeze(0).expand(50, -1)
            
        elif pattern_type == "pulse_train":
            pattern = torch.zeros(50, seq_length)
            pulse_times = torch.arange(10, seq_length, 20)  # Every 20 time steps
            for t in pulse_times:
                if t < seq_length:
                    pattern[:, t] = torch.randn(50)
                    
        elif pattern_type == "random_walk":
            pattern = torch.cumsum(torch.randn(50, seq_length) * 0.1, dim=1)
            
        else:
            # Random pattern
            pattern = torch.randn(50, seq_length)
        
        # Convert to spike trains (rate-based encoding)
        spike_processor = SpikeProcessor()
        rates = torch.clamp((pattern + 1) * 10, 0, 20)  # 0-20 Hz
        spikes = spike_processor.encode_rate_to_spikes(
            rates.unsqueeze(0), seq_length * 0.1, method="poisson"
        )
        
        return spikes[0]  # Remove batch dimension
    
    # Generate different temporal patterns
    patterns = {}
    pattern_types = ["sine_wave", "chirp", "pulse_train", "random_walk"]
    seq_length = 200
    
    logger.info(f"Generating {len(pattern_types)} temporal patterns...")
    
    for pattern_type in pattern_types:
        patterns[pattern_type] = generate_temporal_pattern(seq_length, pattern_type)
        logger.info(f"  Generated {pattern_type} pattern: {patterns[pattern_type].shape}")
    
    # Test LSM responses to different patterns
    logger.info("Testing LSM responses to temporal patterns...")
    
    responses = {}
    for pattern_name, pattern in patterns.items():
        logger.info(f"  Processing {pattern_name}...")
        
        with torch.no_grad():
            output, states = lsm.forward(pattern.unsqueeze(0), return_states=True)
            
            responses[pattern_name] = {
                'output': output,
                'states': states,
                'input_pattern': pattern
            }
    
    # Analyze reservoir dynamics
    logger.info("Analyzing reservoir dynamics...")
    
    for pattern_name, response in responses.items():
        states = response['states']
        dynamics = lsm.compute_reservoir_dynamics(states)
        
        logger.info(f"  {pattern_name} dynamics:")
        logger.info(f"    Mean activity: {dynamics['mean_reservoir_activity']:.4f}")
        logger.info(f"    Activity variance: {dynamics['activity_variance']:.4f}")
        logger.info(f"    Effective dimensionality: {dynamics.get('effective_dimensionality', 0):.2f}")
        
        if 'mean_separation' in dynamics:
            logger.info(f"    Mean separation: {dynamics['mean_separation']:.4f}")
    
    # Memory capacity analysis
    logger.info("Analyzing memory capacity...")
    
    memory_test_pattern = patterns["sine_wave"]
    memory_analysis = lsm.analyze_memory_capacity(
        memory_test_pattern.unsqueeze(0), max_delay=50
    )
    
    logger.info("Memory capacity analysis:")
    logger.info(f"  Total memory capacity: {memory_analysis['total_memory_capacity']:.3f}")
    logger.info(f"  Max effective delay: {memory_analysis['max_effective_delay']} time steps")
    
    # Computational task benchmarking
    logger.info("Benchmarking computational tasks...")
    
    input_sequences = [pattern.unsqueeze(0) for pattern in patterns.values()]
    task_results = lsm.benchmark_computational_tasks(input_sequences)
    
    for task, results in task_results.items():
        logger.info(f"  {task}: {results['mean_performance']:.4f} ± {results['std_performance']:.4f}")
    
    # Readout training demonstration
    logger.info("Demonstrating readout training...")
    
    # Generate training data for classification task
    train_sequences = []
    train_targets = []
    
    for i, (pattern_name, pattern) in enumerate(patterns.items()):
        # Create multiple variations of each pattern
        for _ in range(5):
            noise_pattern = pattern + torch.randn_like(pattern) * 0.1
            train_sequences.append(noise_pattern.unsqueeze(0))
            
            # One-hot target
            target = torch.zeros(len(patterns))
            target[i] = 1.0
            train_targets.append(target.unsqueeze(0))
    
    logger.info(f"Generated {len(train_sequences)} training sequences")
    
    # Train readout
    training_history = lsm.train_readout(
        train_sequences, train_targets, 
        learning_rate=0.01, epochs=50
    )
    
    final_loss = training_history['training_losses'][-1]
    logger.info(f"Readout training completed. Final loss: {final_loss:.6f}")
    
    # Test trained LSM
    logger.info("Testing trained LSM...")
    
    test_accuracy = 0.0
    for i, (pattern_name, pattern) in enumerate(patterns.items()):
        with torch.no_grad():
            output = lsm.forward(pattern.unsqueeze(0))
            predicted_class = torch.argmax(output, dim=1).item()
            
            correct = (predicted_class == i)
            test_accuracy += correct
            
            logger.info(f"  {pattern_name}: Predicted class {predicted_class}, "
                       f"True class {i}, {'Correct' if correct else 'Wrong'}")
    
    test_accuracy /= len(patterns)
    logger.info(f"Test accuracy: {test_accuracy:.2f}")
    
    # Visualizations
    logger.info("Creating visualizations...")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Plot reservoir states for different patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (pattern_name, response) in enumerate(responses.items()):
        if i >= 4:
            break
        
        states = response['states'][0]  # First batch
        time_axis = torch.arange(states.shape[-1]) * lsm.dt
        
        # Plot a subset of reservoir neurons
        neuron_subset = states[:20]  # First 20 neurons
        
        axes[i].imshow(neuron_subset.cpu().numpy(), aspect='auto', cmap='viridis')
        axes[i].set_title(f'Reservoir Activity: {pattern_name}')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Neuron ID')
    
    plt.tight_layout()
    plt.savefig(output_dir / "lsm_reservoir_activity.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot memory capacity
    if memory_analysis['delay_capacities']:
        delays, capacities = zip(*memory_analysis['delay_capacities'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(delays, capacities, 'o-', linewidth=2, markersize=6)
        plt.xlabel('Delay (time steps)')
        plt.ylabel('Memory Capacity')
        plt.title('LSM Memory Capacity vs Delay')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "lsm_memory_capacity.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot input patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (pattern_name, pattern) in enumerate(patterns.items()):
        if i >= 4:
            break
        
        # Show first 10 input channels
        input_subset = pattern[:10]
        time_axis = torch.arange(pattern.shape[-1]) * lsm.dt
        
        axes[i].imshow(input_subset.cpu().numpy(), aspect='auto', cmap='RdBu_r')
        axes[i].set_title(f'Input Pattern: {pattern_name}')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Input Channel')
    
    plt.tight_layout()
    plt.savefig(output_dir / "lsm_input_patterns.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Training loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(training_history['training_losses'])
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('LSM Readout Training Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "lsm_training_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")
    
    # Performance analysis
    logger.info("Performance analysis summary:")
    logger.info(f"  Patterns processed: {len(patterns)}")
    logger.info(f"  Reservoir neurons: {lsm.reservoir_size}")
    logger.info(f"  Memory capacity: {memory_analysis['total_memory_capacity']:.3f}")
    logger.info(f"  Classification accuracy: {test_accuracy:.2%}")
    logger.info(f"  Training convergence: {len(training_history['training_losses'])} epochs")
    
    logger.info("Liquid State Machine demonstration completed successfully!")
    
    return {
        'lsm': lsm,
        'patterns': patterns,
        'responses': responses,
        'memory_analysis': memory_analysis,
        'task_results': task_results,
        'test_accuracy': test_accuracy
    }


def main():
    """Run the LSM demonstration."""
    try:
        results = lsm_demo()
        print("\n=== LSM demonstration completed successfully! ===")
        print(f"Processed {len(results['patterns'])} temporal patterns")
        print(f"Memory capacity: {results['memory_analysis']['total_memory_capacity']:.3f}")
        print(f"Classification accuracy: {results['test_accuracy']:.2%}")
        print("Check 'outputs/' directory for visualizations")
        
    except Exception as e:
        print(f"LSM demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()