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
        spectral_radius=0.9,
        input_scaling=1.0,
        leak_rate=0.1,
        connectivity=0.1
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
        
        # Convert to spike patterns (simple threshold encoding)
        # Normalize pattern to [0, 1] range for spike probability
        pattern_norm = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
        
        # Generate spikes based on pattern values as probabilities
        spikes = torch.rand_like(pattern_norm) < pattern_norm
        
        return spikes.float()
    
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
            # Reshape pattern for LSM: [seq_len, batch_size, input_size]
            pattern_input = pattern.T.unsqueeze(1)  # [seq_len, 1, input_size]
            output, info = lsm.forward(pattern_input)
            
            responses[pattern_name] = {
                'output': output,
                'info': info,
                'input_pattern': pattern
            }
    
    # Analyze reservoir dynamics
    logger.info("Analyzing reservoir dynamics...")
    
    for pattern_name, response in responses.items():
        info = response['info']
        
        logger.info(f"  {pattern_name} dynamics:")
        logger.info(f"    Reservoir activity: {info['reservoir_activity']:.4f}")
        logger.info(f"    State norm: {info['state_norm']:.4f}")
        logger.info(f"    Sparsity: {info['sparsity']:.3f}")
        logger.info(f"    Complexity: {info['complexity']:.4f}")
        logger.info(f"    Memory capacity: {info['memory_capacity']:.4f}")
    
    # Memory capacity analysis
    logger.info("Analyzing memory capacity...")
    
    memory_test_pattern = patterns["sine_wave"]
    # Get average memory capacity from responses
    avg_memory_capacity = sum(r['info']['memory_capacity'] for r in responses.values()) / len(responses)
    
    logger.info("Memory capacity analysis:")
    logger.info(f"  Average memory capacity: {avg_memory_capacity:.3f}")
    logger.info(f"  Reservoir complexity: {responses['sine_wave']['info']['complexity']:.3f}")
    
    # Simple computational analysis
    logger.info("Analyzing computational capabilities...")
    
    for pattern_name, response in responses.items():
        output_variance = response['output'].var().item()
        output_mean = response['output'].mean().item()
        logger.info(f"  {pattern_name} output: mean={output_mean:.4f}, var={output_variance:.4f}")
    
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
    
    # Train readout (simplified for demonstration)
    optimizer = torch.optim.Adam(lsm.readout.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    training_losses = []
    for epoch in range(50):
        total_loss = 0.0
        for seq, target in zip(train_sequences, train_targets):
            seq_input = seq.transpose(0, 2).transpose(1, 2)  # Reshape to [seq, batch, input]
            output, _ = lsm.forward(seq_input, reset_state=True)
            final_output = output[-1]  # Last timestep output
            
            loss = criterion(final_output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        training_losses.append(total_loss / len(train_sequences))
        if epoch % 10 == 0:
            logger.info(f"  Epoch {epoch}: Loss = {training_losses[-1]:.6f}")
    
    logger.info(f"Readout training completed. Final loss: {training_losses[-1]:.6f}")
    
    # Test trained LSM
    logger.info("Testing trained LSM...")
    
    test_accuracy = 0.0
    for i, (pattern_name, pattern) in enumerate(patterns.items()):
        with torch.no_grad():
            pattern_input = pattern.T.unsqueeze(1)
            output, _ = lsm.forward(pattern_input, reset_state=True)
            final_output = output[-1]  # Last timestep
            predicted_class = torch.argmax(final_output, dim=1).item()
            
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
        
        # Use final reservoir state for visualization
        final_state = response['info']['final_state']
        state_2d = final_state.view(20, -1)  # Reshape for visualization
        
        axes[i].imshow(state_2d.detach().cpu().numpy(), aspect='auto', cmap='viridis')
        axes[i].set_title(f'Reservoir State: {pattern_name}')
        axes[i].set_xlabel('State Dimension')
        axes[i].set_ylabel('Neuron Group')
    
    plt.tight_layout()
    plt.savefig(output_dir / "lsm_reservoir_activity.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot memory capacity comparison
    memory_capacities = [r['info']['memory_capacity'] for r in responses.values()]
    pattern_names = list(responses.keys())
    
    plt.figure(figsize=(10, 6))
    plt.bar(pattern_names, memory_capacities, alpha=0.7)
    plt.xlabel('Pattern Type')
    plt.ylabel('Memory Capacity')
    plt.title('LSM Memory Capacity by Pattern Type')
    plt.xticks(rotation=45)
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
        
        axes[i].imshow(input_subset.cpu().numpy(), aspect='auto', cmap='RdBu_r')
        axes[i].set_title(f'Input Pattern: {pattern_name}')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Input Channel')
    
    plt.tight_layout()
    plt.savefig(output_dir / "lsm_input_patterns.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Training loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses)
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
    logger.info(f"  Average memory capacity: {avg_memory_capacity:.3f}")
    logger.info(f"  Classification accuracy: {test_accuracy:.2%}")
    logger.info(f"  Training epochs: {len(training_losses)} epochs")
    
    logger.info("Liquid State Machine demonstration completed successfully!")
    
    return {
        'lsm': lsm,
        'patterns': patterns,
        'responses': responses,
        'avg_memory_capacity': avg_memory_capacity,
        'training_losses': training_losses,
        'test_accuracy': test_accuracy
    }


def main():
    """Run the LSM demonstration."""
    try:
        results = lsm_demo()
        print("\n=== LSM demonstration completed successfully! ===")
        print(f"Processed {len(results['patterns'])} temporal patterns")
        print(f"Average memory capacity: {results['avg_memory_capacity']:.3f}")
        print(f"Classification accuracy: {results['test_accuracy']:.2%}")
        print("Check 'outputs/' directory for visualizations")
        
    except Exception as e:
        print(f"LSM demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()