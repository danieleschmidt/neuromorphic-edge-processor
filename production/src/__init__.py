"""Neuromorphic Edge Processor - Brain-inspired ultra-low power computing at the edge.

This package implements novel neuromorphic computing algorithms optimized for edge devices,
focusing on spiking neural networks and event-driven processing architectures.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

from .models import SpikingNeuralNetwork, LIFNeuron
from .algorithms import EventDrivenProcessor, SpikeProcessor
from .utils import DataLoader

__all__ = [
    "SpikingNeuralNetwork",
    "LIFNeuron",
    "EventDrivenProcessor", 
    "SpikeProcessor",
    "DataLoader",
]