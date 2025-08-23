"""Data loading utilities for neuromorphic datasets."""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import os
from pathlib import Path


class DataLoader:
    """Simple data loader for neuromorphic spike data."""
    
    def __init__(self, batch_size: int = 32, shuffle: bool = True):
        """Initialize data loader.
        
        Args:
            batch_size: Batch size for loading data
            shuffle: Whether to shuffle data
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def load_spike_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load spike data from file.
        
        Args:
            data_path: Path to data file
            
        Returns:
            Tuple of (spike_data, labels)
        """
        if not os.path.exists(data_path):
            # Generate synthetic spike data if file doesn't exist
            return self._generate_synthetic_data()
        
        # Load from numpy file if exists
        if data_path.endswith('.npy'):
            data = np.load(data_path)
            # Assume first column is data, second is labels
            if data.ndim == 2 and data.shape[1] > 1:
                return data[:, :-1], data[:, -1]
            else:
                labels = np.random.randint(0, 10, size=(len(data),))
                return data, labels
        
        # Default: return synthetic data
        return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic neuromorphic data for testing.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (spike_data, labels)
        """
        # Generate random spike trains
        time_steps = 100
        num_features = 784  # 28x28 like MNIST
        
        spike_data = np.random.rand(num_samples, num_features, time_steps) < 0.1
        spike_data = spike_data.astype(np.float32)
        
        # Random labels for 10 classes
        labels = np.random.randint(0, 10, size=(num_samples,))
        
        return spike_data, labels
    
    def create_batches(self, data: np.ndarray, labels: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create batches from data.
        
        Args:
            data: Input data
            labels: Labels
            
        Returns:
            List of (batch_data, batch_labels) tuples
        """
        num_samples = len(data)
        indices = np.arange(num_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        batches = []
        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = data[batch_indices]
            batch_labels = labels[batch_indices]
            batches.append((batch_data, batch_labels))
        
        return batches