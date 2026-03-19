"""
Neuromorphic Edge Processor Simulator

Models a brain-inspired event-driven processor architecture.
Core abstractions mirror Intel Loihi / IBM TrueNorth hardware design.
"""

from .event_stream import EventStream, SpikeEvent
from .core import NeuromorphicCore, LIFNeuron
from .router import SynapticRouter
from .processor import NeuromorphicProcessor
from .energy import EnergyProfiler

__all__ = [
    "EventStream",
    "SpikeEvent",
    "NeuromorphicCore",
    "LIFNeuron",
    "SynapticRouter",
    "NeuromorphicProcessor",
    "EnergyProfiler",
]

__version__ = "1.0.0"
