"""Data loading utilities for neuromorphic datasets."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import numpy as np
import h5py
from typing import List, Dict, Optional, Tuple, Union
import os
from pathlib import Path
import json


class NeuromorphicDataset(Dataset):
    """Dataset class for neuromorphic data with spike trains and events.
    
    Supports multiple neuromorphic data formats including:
    - DVS (Dynamic Vision Sensor) events
    - N-MNIST spike data
    - Custom spike train datasets
    """
    
    def __init__(
        self,
        data_path: str,
        data_type: str = "spikes",
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        time_window: float = 100.0,  # ms
        spatial_resolution: Tuple[int, int] = (28, 28)
    ):
        """Initialize neuromorphic dataset.
        
        Args:
            data_path: Path to dataset
            data_type: Type of data ('spikes', 'events', 'dvs')
            transform: Transform to apply to inputs
            target_transform: Transform to apply to targets
            time_window: Time window for spike accumulation (ms)
            spatial_resolution: Spatial resolution for event data
        """
        self.data_path = Path(data_path)
        self.data_type = data_type
        self.transform = transform
        self.target_transform = target_transform
        self.time_window = time_window
        self.spatial_resolution = spatial_resolution
        
        # Load dataset
        self.data, self.targets = self._load_dataset()
        
    def _load_dataset(self) -> Tuple[List, List]:
        """Load dataset from specified path."""
        if not self.data_path.exists():
            # Generate synthetic data for testing
            return self._generate_synthetic_data()
        
        if self.data_type == "spikes":
            return self._load_spike_data()
        elif self.data_type == "events":
            return self._load_event_data()
        elif self.data_type == "dvs":
            return self._load_dvs_data()
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
    
    def _generate_synthetic_data(self, num_samples: int = 1000) -> Tuple[List, List]:
        """Generate synthetic neuromorphic data for testing."""
        print(f"Generating {num_samples} synthetic samples...")
        
        data = []
        targets = []
        
        for i in range(num_samples):
            # Generate random spike pattern
            num_neurons = self.spatial_resolution[0] * self.spatial_resolution[1]
            time_steps = int(self.time_window)  # 1ms resolution
            
            # Create patterns with different classes
            class_id = i % 10  # 10 classes
            
            if self.data_type == "spikes":
                # Spike train format
                spikes = torch.zeros(num_neurons, time_steps)
                
                # Add class-specific patterns
                base_rate = 0.02 + class_id * 0.005  # Vary base rate by class
                
                # Random Poisson spikes
                random_spikes = torch.rand(num_neurons, time_steps) < base_rate
                spikes = random_spikes.float()
                
                # Add structured patterns for each class
                pattern_neurons = torch.randperm(num_neurons)[:num_neurons//4]\n                pattern_times = torch.randperm(time_steps)[:time_steps//4]\n                \n                for neuron in pattern_neurons:\n                    for t in pattern_times:\n                        if torch.rand(1) < 0.3:  # Pattern probability\n                            spikes[neuron, t] = 1.0\n                \n                data.append(spikes)\n                \n            elif self.data_type == \"events\":\n                # Event-based format: (x, y, t, polarity)\n                events = []\n                \n                num_events = torch.randint(100, 1000, (1,)).item()\n                \n                for _ in range(num_events):\n                    x = torch.randint(0, self.spatial_resolution[0], (1,)).item()\n                    y = torch.randint(0, self.spatial_resolution[1], (1,)).item()\n                    t = torch.rand(1).item() * self.time_window\n                    polarity = torch.randint(0, 2, (1,)).item()\n                    \n                    events.append([x, y, t, polarity])\n                \n                data.append(torch.tensor(events))\n            \n            targets.append(class_id)\n        \n        return data, targets\n    \n    def _load_spike_data(self) -> Tuple[List, List]:\n        \"\"\"Load spike train data from files.\"\"\"\n        data = []\n        targets = []\n        \n        # Look for .h5 or .npz files\n        spike_files = list(self.data_path.glob(\"*.h5\")) + list(self.data_path.glob(\"*.npz\"))\n        \n        for file_path in spike_files:\n            if file_path.suffix == \".h5\":\n                with h5py.File(file_path, 'r') as f:\n                    spikes = torch.tensor(f['spikes'][:])\n                    labels = torch.tensor(f['labels'][:])\n            else:  # .npz\n                loaded = np.load(file_path)\n                spikes = torch.tensor(loaded['spikes'])\n                labels = torch.tensor(loaded['labels'])\n            \n            data.extend(spikes)\n            targets.extend(labels)\n        \n        return data, targets\n    \n    def _load_event_data(self) -> Tuple[List, List]:\n        \"\"\"Load event-based data (e.g., DVS format).\"\"\"\n        # Placeholder for event data loading\n        # Would implement specific DVS/event format parsers here\n        return self._generate_synthetic_data()\n    \n    def _load_dvs_data(self) -> Tuple[List, List]:\n        \"\"\"Load DVS camera data.\"\"\"\n        # Placeholder for DVS data loading\n        # Would implement DVS-specific data loaders here\n        return self._generate_synthetic_data()\n    \n    def __len__(self) -> int:\n        \"\"\"Return dataset length.\"\"\"\n        return len(self.data)\n    \n    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:\n        \"\"\"Get item from dataset.\"\"\"\n        sample = self.data[idx]\n        target = self.targets[idx]\n        \n        if self.transform:\n            sample = self.transform(sample)\n        \n        if self.target_transform:\n            target = self.target_transform(target)\n        \n        return sample, torch.tensor(target, dtype=torch.long)\n    \n    def get_statistics(self) -> Dict:\n        \"\"\"Get dataset statistics.\"\"\"\n        stats = {\n            \"num_samples\": len(self.data),\n            \"data_type\": self.data_type,\n            \"time_window\": self.time_window,\n            \"spatial_resolution\": self.spatial_resolution,\n            \"num_classes\": len(set(self.targets))\n        }\n        \n        if self.data_type == \"spikes\":\n            # Compute spike statistics\n            total_spikes = sum(sample.sum().item() for sample in self.data[:100])  # Sample first 100\n            avg_spikes_per_sample = total_spikes / min(100, len(self.data))\n            stats[\"avg_spikes_per_sample\"] = avg_spikes_per_sample\n            \n            if len(self.data) > 0:\n                sample_shape = self.data[0].shape\n                stats[\"sample_shape\"] = sample_shape\n                sparsity = 1.0 - (total_spikes / (min(100, len(self.data)) * np.prod(sample_shape)))\n                stats[\"sparsity\"] = sparsity\n        \n        return stats\n\n\nclass DataLoader:\n    \"\"\"Enhanced data loader for neuromorphic computing experiments.\"\"\"\n    \n    def __init__(self):\n        self.datasets = {}\n        self.data_loaders = {}\n    \n    def create_dataset(\n        self,\n        name: str,\n        data_path: str,\n        data_type: str = \"spikes\",\n        **kwargs\n    ) -> NeuromorphicDataset:\n        \"\"\"Create and register a neuromorphic dataset.\n        \n        Args:\n            name: Dataset name\n            data_path: Path to dataset\n            data_type: Type of neuromorphic data\n            **kwargs: Additional arguments for NeuromorphicDataset\n            \n        Returns:\n            Created dataset\n        \"\"\"\n        dataset = NeuromorphicDataset(data_path, data_type, **kwargs)\n        self.datasets[name] = dataset\n        return dataset\n    \n    def create_data_loader(\n        self,\n        dataset_name: str,\n        batch_size: int = 32,\n        shuffle: bool = True,\n        num_workers: int = 0,\n        **kwargs\n    ) -> TorchDataLoader:\n        \"\"\"Create PyTorch DataLoader for registered dataset.\n        \n        Args:\n            dataset_name: Name of registered dataset\n            batch_size: Batch size\n            shuffle: Whether to shuffle data\n            num_workers: Number of worker processes\n            **kwargs: Additional DataLoader arguments\n            \n        Returns:\n            PyTorch DataLoader\n        \"\"\"\n        if dataset_name not in self.datasets:\n            raise ValueError(f\"Dataset '{dataset_name}' not found. Available: {list(self.datasets.keys())}\")\n        \n        dataset = self.datasets[dataset_name]\n        \n        data_loader = TorchDataLoader(\n            dataset,\n            batch_size=batch_size,\n            shuffle=shuffle,\n            num_workers=num_workers,\n            collate_fn=self._collate_fn,\n            **kwargs\n        )\n        \n        self.data_loaders[dataset_name] = data_loader\n        return data_loader\n    \n    def _collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:\n        \"\"\"Custom collate function for variable-length sequences.\"\"\"\n        samples, targets = zip(*batch)\n        \n        # Handle different data formats\n        if isinstance(samples[0], torch.Tensor):\n            if samples[0].dim() == 2:  # [neurons, time]\n                # Pad sequences to same length\n                max_time = max(s.size(1) for s in samples)\n                padded_samples = []\n                \n                for sample in samples:\n                    if sample.size(1) < max_time:\n                        padding = torch.zeros(sample.size(0), max_time - sample.size(1))\n                        padded_sample = torch.cat([sample, padding], dim=1)\n                    else:\n                        padded_sample = sample\n                    padded_samples.append(padded_sample)\n                \n                samples_tensor = torch.stack(padded_samples)\n            else:\n                samples_tensor = torch.stack(samples)\n        else:\n            # Handle other formats (e.g., events)\n            samples_tensor = torch.stack([torch.tensor(s) for s in samples])\n        \n        targets_tensor = torch.stack(targets)\n        \n        return samples_tensor, targets_tensor\n    \n    def get_benchmark_datasets(self) -> Dict[str, NeuromorphicDataset]:\n        \"\"\"Get standard benchmark datasets for neuromorphic computing.\"\"\"\n        benchmark_datasets = {}\n        \n        # Create standard benchmark datasets\n        benchmarks = {\n            \"n_mnist\": {\n                \"data_path\": \"data/n_mnist\",\n                \"data_type\": \"spikes\",\n                \"spatial_resolution\": (28, 28),\n                \"time_window\": 300.0\n            },\n            \"dvs_gesture\": {\n                \"data_path\": \"data/dvs_gesture\", \n                \"data_type\": \"events\",\n                \"spatial_resolution\": (128, 128),\n                \"time_window\": 1000.0\n            },\n            \"shd\": {  # Spiking Heidelberg Dataset\n                \"data_path\": \"data/shd\",\n                \"data_type\": \"spikes\",\n                \"spatial_resolution\": (700, 1),  # 700 input channels\n                \"time_window\": 1000.0\n            }\n        }\n        \n        for name, params in benchmarks.items():\n            try:\n                dataset = self.create_dataset(name, **params)\n                benchmark_datasets[name] = dataset\n                print(f\"Loaded benchmark dataset: {name} ({len(dataset)} samples)\")\n            except Exception as e:\n                print(f\"Could not load benchmark dataset {name}: {e}\")\n                # Create synthetic version\n                dataset = self.create_dataset(f\"synthetic_{name}\", \"nonexistent\", **params)\n                benchmark_datasets[f\"synthetic_{name}\"] = dataset\n        \n        return benchmark_datasets\n    \n    def create_train_val_split(\n        self,\n        dataset_name: str,\n        val_ratio: float = 0.2,\n        random_seed: int = 42\n    ) -> Tuple[TorchDataLoader, TorchDataLoader]:\n        \"\"\"Create train/validation split for dataset.\n        \n        Args:\n            dataset_name: Name of dataset to split\n            val_ratio: Fraction of data for validation\n            random_seed: Random seed for reproducibility\n            \n        Returns:\n            Tuple of (train_loader, val_loader)\n        \"\"\"\n        if dataset_name not in self.datasets:\n            raise ValueError(f\"Dataset '{dataset_name}' not found\")\n        \n        dataset = self.datasets[dataset_name]\n        dataset_size = len(dataset)\n        val_size = int(val_ratio * dataset_size)\n        train_size = dataset_size - val_size\n        \n        generator = torch.Generator().manual_seed(random_seed)\n        train_dataset, val_dataset = torch.utils.data.random_split(\n            dataset, [train_size, val_size], generator=generator\n        )\n        \n        train_loader = TorchDataLoader(\n            train_dataset,\n            batch_size=32,\n            shuffle=True,\n            collate_fn=self._collate_fn\n        )\n        \n        val_loader = TorchDataLoader(\n            val_dataset,\n            batch_size=32,\n            shuffle=False,\n            collate_fn=self._collate_fn\n        )\n        \n        return train_loader, val_loader\n    \n    def get_dataset_info(self, dataset_name: str) -> Dict:\n        \"\"\"Get information about a dataset.\"\"\"\n        if dataset_name not in self.datasets:\n            return {\"error\": f\"Dataset '{dataset_name}' not found\"}\n        \n        dataset = self.datasets[dataset_name]\n        info = {\n            \"name\": dataset_name,\n            \"size\": len(dataset),\n            \"statistics\": dataset.get_statistics()\n        }\n        \n        return info"