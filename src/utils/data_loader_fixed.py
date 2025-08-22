"""Data loading utilities for neuromorphic computing."""

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
# import h5py  # Optional dependency


class NeuromorphicDataset(Dataset):
    """Dataset class for neuromorphic data (spikes, events, DVS)."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        data_type: str = "spikes",
        time_window: float = 1000.0,
        spatial_resolution: Tuple[int, int] = (128, 128),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        num_classes: int = 10,
        samples_per_class: int = 100
    ):
        """Initialize neuromorphic dataset.
        
        Args:
            data_path: Path to dataset
            data_type: Type of data ('spikes', 'events', 'dvs')
            time_window: Time window in ms
            spatial_resolution: Spatial resolution (height, width)
            transform: Optional transform for data
            target_transform: Optional transform for targets
            num_classes: Number of classes for synthetic data
            samples_per_class: Samples per class for synthetic data
        """
        self.data_path = Path(data_path)
        self.data_type = data_type
        self.time_window = time_window
        self.spatial_resolution = spatial_resolution
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        
        # Load or generate data
        if self.data_path.exists():
            self.data, self.targets = self._load_data()
        else:
            # Generate synthetic data
            self.data, self.targets = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> Tuple[List, List]:
        """Generate synthetic neuromorphic data for testing."""
        data = []
        targets = []
        
        for class_id in range(self.num_classes):
            for _ in range(self.samples_per_class):
                if self.data_type == "spikes":
                    # Generate spike trains
                    num_neurons = self.spatial_resolution[0] if len(self.spatial_resolution) == 2 else 100
                    time_steps = int(self.time_window)
                    
                    # Base firing rate varies by class
                    base_rate = 0.01 + (class_id * 0.005)  # 1-6% spike probability
                    
                    # Generate random spikes
                    random_spikes = torch.rand(num_neurons, time_steps) < base_rate
                    spikes = random_spikes.float()
                    
                    # Add class-specific patterns
                    pattern_neurons = torch.randperm(num_neurons)[:num_neurons//4]
                    pattern_times = torch.randperm(time_steps)[:time_steps//4]
                    
                    for neuron in pattern_neurons:
                        for t in pattern_times:
                            if torch.rand(1) < 0.3:  # Pattern probability
                                spikes[neuron, t] = 1.0
                    
                    data.append(spikes)
                    
                elif self.data_type == "events":
                    # Generate events (x, y, t, polarity)
                    num_events = torch.randint(100, 1000, (1,)).item()
                    events = []
                    
                    for _ in range(num_events):
                        x = torch.randint(0, self.spatial_resolution[0], (1,)).item()
                        y = torch.randint(0, self.spatial_resolution[1], (1,)).item()
                        t = torch.rand(1).item() * self.time_window
                        polarity = torch.randint(0, 2, (1,)).item()
                        events.append([x, y, t, polarity])
                    
                    data.append(torch.tensor(events))
                
                targets.append(class_id)
        
        return data, targets
    
    def _load_data(self) -> Tuple[List, List]:
        """Load data from files."""
        if self.data_type == "spikes":
            return self._load_spike_data()
        elif self.data_type in ["events", "dvs"]:
            return self._load_event_data()
        else:
            return self._generate_synthetic_data()
    
    def _load_spike_data(self) -> Tuple[List, List]:
        """Load spike train data from files."""
        data = []
        targets = []
        
        # Look for data files
        spike_files = list(self.data_path.glob("*.npz"))
        
        for file_path in spike_files:
            loaded = np.load(file_path)
            if 'spikes' in loaded and 'labels' in loaded:
                spikes = torch.tensor(loaded['spikes'])
                labels = torch.tensor(loaded['labels'])
                data.extend(spikes)
                targets.extend(labels)
        
        if not data:
            # No files found, generate synthetic
            return self._generate_synthetic_data()
        
        return data, targets
    
    def _load_event_data(self) -> Tuple[List, List]:
        """Load event-based data."""
        # Placeholder - would implement DVS/event format parsers
        return self._generate_synthetic_data()
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from dataset."""
        sample = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return sample, torch.tensor(target, dtype=torch.long)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            "num_samples": len(self.data),
            "data_type": self.data_type,
            "time_window": self.time_window,
            "spatial_resolution": self.spatial_resolution,
            "num_classes": len(set(self.targets))
        }
        
        if self.data_type == "spikes" and len(self.data) > 0:
            # Compute spike statistics
            sample_data = self.data[:min(10, len(self.data))]  # Sample first 10
            total_spikes = sum(sample.sum().item() for sample in sample_data)
            avg_spikes_per_sample = total_spikes / len(sample_data)
            stats["avg_spikes_per_sample"] = avg_spikes_per_sample
            
            sample_shape = self.data[0].shape
            stats["sample_shape"] = sample_shape
            sparsity = 1.0 - (total_spikes / (len(sample_data) * np.prod(sample_shape)))
            stats["sparsity"] = sparsity
        
        return stats


class DataLoader:
    """Enhanced data loader for neuromorphic computing experiments."""
    
    def __init__(self):
        self.datasets = {}
        self.data_loaders = {}
    
    def create_dataset(
        self,
        name: str,
        data_path: str,
        data_type: str = "spikes",
        **kwargs
    ) -> NeuromorphicDataset:
        """Create and register a neuromorphic dataset."""
        dataset = NeuromorphicDataset(data_path, data_type, **kwargs)
        self.datasets[name] = dataset
        return dataset
    
    def create_data_loader(
        self,
        dataset_name: str,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> TorchDataLoader:
        """Create PyTorch DataLoader for registered dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset = self.datasets[dataset_name]
        
        data_loader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            **kwargs
        )
        
        self.data_loaders[dataset_name] = data_loader
        return data_loader
    
    def _collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom collate function for variable-length sequences."""
        samples, targets = zip(*batch)
        
        # Simple stacking for now
        try:
            samples_tensor = torch.stack(samples)
        except:
            # Handle variable sizes by padding
            max_shape = max(s.shape for s in samples)
            padded_samples = []
            for sample in samples:
                if sample.shape != max_shape:
                    # Pad to max shape
                    pad_dims = []
                    for i in range(len(max_shape)):
                        pad_dims.extend([0, max_shape[i] - sample.shape[i]])
                    padded = torch.nn.functional.pad(sample, pad_dims)
                    padded_samples.append(padded)
                else:
                    padded_samples.append(sample)
            samples_tensor = torch.stack(padded_samples)
        
        targets_tensor = torch.stack(targets)
        return samples_tensor, targets_tensor
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get information about a dataset."""
        if dataset_name not in self.datasets:
            return {"error": f"Dataset '{dataset_name}' not found"}
        
        dataset = self.datasets[dataset_name]
        info = {
            "name": dataset_name,
            "size": len(dataset),
            "statistics": dataset.get_statistics()
        }
        
        return info