"""Temporal coding algorithms for spiking neural networks with 2024-2025 enhancements."""

import numpy as np
from typing import Dict, List, Optional, Tuple
import math


class TemporalEncoder:
    """Encode continuous values using temporal spike patterns."""
    
    def __init__(self, dt: float = 1.0, encoding_window: float = 50.0):
        """Initialize temporal encoder.
        
        Args:
            dt: Time step (ms)
            encoding_window: Time window for encoding (ms)
        """
        self.dt = dt
        self.encoding_window = encoding_window
        self.time_steps = int(encoding_window / dt)
    
    def encode_latency(self, values: np.ndarray, min_latency: float = 5.0, max_latency: float = 45.0) -> np.ndarray:
        """Encode values as spike latencies.
        
        Higher values produce earlier spikes (lower latency).
        
        Args:
            values: Input values [batch_size, num_features] normalized to [0,1]
            min_latency: Minimum spike latency (ms)
            max_latency: Maximum spike latency (ms)
            
        Returns:
            Spike trains [batch_size, num_features, time_steps]
        """
        batch_size, num_features = values.shape
        spikes = np.zeros((batch_size, num_features, self.time_steps))
        
        # Map values to latencies (higher value = lower latency)
        latencies = max_latency - values * (max_latency - min_latency)
        
        # Generate spikes at computed latencies
        for b in range(batch_size):
            for f in range(num_features):
                if values[b, f] > 0:  # Only create spike if value > 0
                    spike_time = int(latencies[b, f] / self.dt)
                    if 0 <= spike_time < self.time_steps:
                        spikes[b, f, spike_time] = 1.0
        
        return spikes


class TemporalDecoder:
    """Decode temporal spike patterns back to continuous values."""
    
    def __init__(self, dt: float = 1.0, decoding_window: float = 50.0):
        """Initialize temporal decoder.
        
        Args:
            dt: Time step (ms)
            decoding_window: Time window for decoding (ms)
        """
        self.dt = dt
        self.decoding_window = decoding_window
        self.time_steps = int(decoding_window / dt)
    
    def decode_latency(self, spikes: np.ndarray, min_latency: float = 5.0, max_latency: float = 45.0) -> np.ndarray:
        """Decode spike latencies to continuous values.
        
        Args:
            spikes: Spike trains [batch_size, num_features, time_steps]
            min_latency: Minimum spike latency (ms)
            max_latency: Maximum spike latency (ms)
            
        Returns:
            Decoded values [batch_size, num_features]
        """
        batch_size, num_features, _ = spikes.shape
        values = np.zeros((batch_size, num_features))
        
        # Find first spike time for each feature
        for b in range(batch_size):
            for f in range(num_features):
                spike_times = np.where(spikes[b, f] > 0)[0]
                if len(spike_times) > 0:
                    first_spike_time = spike_times[0] * self.dt
                    # Map latency back to value
                    value = 1.0 - (first_spike_time - min_latency) / (max_latency - min_latency)
                    values[b, f] = np.clip(value, 0, 1)
        
        return values


class MemoryEfficientTemporalDecoder:
    """Memory-efficient temporal decoder with compression.
    
    Based on 2024-2025 research on edge-optimized neuromorphic processing.
    Implements delta encoding and circular buffers to minimize memory usage.
    """
    
    def __init__(self, circular_buffer_size: int = 1000, compression_ratio: float = 0.1):
        """Initialize memory-efficient decoder.
        
        Args:
            circular_buffer_size: Size of circular buffer for temporal data
            compression_ratio: Compression ratio for delta encoding
        """
        self.circular_buffer_size = circular_buffer_size
        self.compression_ratio = compression_ratio
        self.temporal_buffer = None
        self.buffer_index = 0
        self.compression_stats = {"original_size": 0, "compressed_size": 0}
    
    def initialize_buffer(self, num_features: int):
        """Initialize circular buffer for temporal data."""
        self.temporal_buffer = np.zeros((self.circular_buffer_size, num_features))
        self.buffer_index = 0
    
    def compress_spike_times(self, spike_times: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Compress spike times using delta encoding.
        
        Args:
            spike_times: Array of spike times
            
        Returns:
            Compressed spike times and compression metadata
        """
        if len(spike_times) == 0:
            return spike_times, {"compression_ratio": 1.0}
        
        # Delta encoding: store differences between consecutive spike times
        deltas = np.diff(spike_times)
        
        # Quantize deltas to reduce precision
        quantization_factor = 1.0 / self.compression_ratio
        quantized_deltas = np.round(deltas * quantization_factor) / quantization_factor
        
        # Store first spike time and deltas
        compressed = np.concatenate([[spike_times[0]], quantized_deltas])
        
        # Update compression statistics
        original_size = len(spike_times) * 8  # 8 bytes per float64
        compressed_size = len(compressed) * 4  # 4 bytes per compressed value
        
        self.compression_stats["original_size"] += original_size
        self.compression_stats["compressed_size"] += compressed_size
        
        compression_metadata = {
            "compression_ratio": compressed_size / original_size if original_size > 0 else 1.0,
            "original_length": len(spike_times),
            "compressed_length": len(compressed)
        }
        
        return compressed, compression_metadata
    
    def decompress_spike_times(self, compressed_data: np.ndarray) -> np.ndarray:
        """Decompress delta-encoded spike times.
        
        Args:
            compressed_data: Compressed spike time data
            
        Returns:
            Decompressed spike times
        """
        if len(compressed_data) == 0:
            return compressed_data
        
        # Reconstruct from deltas
        first_time = compressed_data[0]
        deltas = compressed_data[1:]
        
        # Cumulative sum to get original spike times
        spike_times = np.concatenate([[first_time], first_time + np.cumsum(deltas)])
        
        return spike_times
    
    def decode_with_compression(self, spikes: np.ndarray) -> np.ndarray:
        """Decode spikes using compressed temporal representations.
        
        Args:
            spikes: Spike trains [batch_size, num_features, time_steps]
            
        Returns:
            Decoded values [batch_size, num_features]
        """
        batch_size, num_features, time_steps = spikes.shape
        
        # Initialize buffer if needed
        if self.temporal_buffer is None:
            self.initialize_buffer(num_features)
        
        values = np.zeros((batch_size, num_features))
        
        for b in range(batch_size):
            for f in range(num_features):
                # Extract spike times
                spike_indices = np.where(spikes[b, f] > 0)[0]
                
                if len(spike_indices) > 0:
                    # Compress spike times
                    compressed_times, metadata = self.compress_spike_times(spike_indices.astype(float))
                    
                    # Use circular buffer for temporal context
                    buffer_idx = self.buffer_index % self.circular_buffer_size
                    
                    # Store in buffer (using first spike as representative)
                    self.temporal_buffer[buffer_idx, f] = compressed_times[0] if len(compressed_times) > 0 else 0
                    
                    # Decode value based on temporal pattern and history
                    if len(compressed_times) > 0:
                        # Early spikes = higher values
                        primary_spike_time = compressed_times[0]
                        normalized_time = primary_spike_time / time_steps
                        
                        # Incorporate temporal context from buffer
                        if buffer_idx > 0:
                            recent_history = self.temporal_buffer[max(0, buffer_idx-5):buffer_idx, f]
                            context_weight = np.mean(recent_history) if len(recent_history) > 0 else 0
                            temporal_context = 0.1 * context_weight / time_steps
                        else:
                            temporal_context = 0
                        
                        # Combine temporal information
                        base_value = 1.0 - normalized_time  # Earlier = higher value
                        values[b, f] = np.clip(base_value + temporal_context, 0, 1)
        
        # Update buffer index
        self.buffer_index += batch_size
        
        return values
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression efficiency statistics."""
        total_original = self.compression_stats["original_size"]
        total_compressed = self.compression_stats["compressed_size"]
        
        if total_original > 0:
            overall_ratio = total_compressed / total_original
            space_saved = 1.0 - overall_ratio
        else:
            overall_ratio = 1.0
            space_saved = 0.0
        
        return {
            "overall_compression_ratio": overall_ratio,
            "space_saved_percentage": space_saved * 100,
            "total_original_bytes": total_original,
            "total_compressed_bytes": total_compressed,
            "buffer_utilization": min(1.0, self.buffer_index / self.circular_buffer_size)
        }