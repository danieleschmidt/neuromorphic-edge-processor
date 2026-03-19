# Neuromorphic Edge Processor

A simulator for brain-inspired, event-driven processor architecture — modelling the computational style of Intel Loihi, IBM TrueNorth, and Dynamic Vision Sensors (DVS).

---

## What This Is

Conventional processors compute continuously, burning energy even when nothing changes. Neuromorphic processors take a different approach: **communicate only via spikes, compute only when events arrive**. The result is orders-of-magnitude better energy efficiency for sparse, event-driven workloads like sensor processing, anomaly detection, and real-time inference at the edge.

This simulator models the core abstractions of neuromorphic hardware:

| Component | Hardware Analog |
|-----------|-----------------|
| `EventStream` | DVS/DAVIS event camera output; AER bus |
| `NeuromorphicCore` | Loihi core (1024 compartments), TrueNorth crossbar (256 neurons) |
| `SynapticRouter` | Loihi mesh network; TrueNorth torus interconnect |
| `NeuromorphicProcessor` | Full chip — multi-core with inter-core routing |
| `EnergyProfiler` | SOP-based energy estimation (23.6 pJ/SOP @ Loihi) |

---

## Core Concepts

### Synaptic Operations (SOPs)

The fundamental compute unit of neuromorphic hardware. One SOP = one input spike crossing one synapse. Unlike dense matrix multiply, SOPs only happen when a neuron fires — so sparse activity = low energy.

| Platform | Energy/operation |
|----------|-----------------|
| Intel Loihi | 23.6 pJ/SOP |
| IBM TrueNorth | 26 pJ/SOP |
| GPU (MAC) | ~1,000 pJ/MAC |
| CPU (MAC) | ~5,000 pJ/MAC |

**Typical advantage**: 40–100× less energy than GPU for sparse event-driven workloads.

### Leaky Integrate-and-Fire (LIF) Neurons

Each core simulates a population of LIF neurons — the workhorse model of neuromorphic silicon:

```
V[t] = V[t-1] · decay + I_syn[t]
if V[t] ≥ V_threshold:
    emit spike, V[t] = V_reset
```

Activity is sparse: at biological firing rates (~5 Hz mean), >95% of neurons are silent at any given moment.

### Address Event Representation (AER)

Events are sparse structs: `(timestamp_µs, neuron_id, polarity)`. This is exactly the protocol used by DVS cameras and the AER bus in real chips. No dense frames — only changes.

---

## Installation

```bash
git clone https://github.com/danieleschmidt/neuromorphic-edge-processor
cd neuromorphic-edge-processor
pip install -e .
```

Requires Python 3.9+ and NumPy. No other dependencies.

---

## Quick Start

```python
from neuromorphic import (
    EventStream,
    NeuromorphicProcessor,
    EnergyProfiler,
)

# 1. Create an event stream (rate-coded encoding)
import numpy as np
rates = np.array([50.0, 100.0, 25.0])   # Hz per neuron
stream = EventStream.from_rate_coding(rates, duration_us=100_000)

# 2. Build a 2-core processor
proc = NeuromorphicProcessor(n_cores=2, neurons_per_core=64)
proc.router.connect_pipeline([0, 1])
proc.core(0).init_weights(n_inputs=3,  connectivity=0.8)
proc.core(1).init_weights(n_inputs=64, connectivity=0.3)

# 3. Run simulation
results = proc.run(stream, duration_us=100_000)

print(f"Total SOPs:   {results['total_sops']:,}")
print(f"Energy:       {results['energy_nj']:.4f} nJ")
print(f"Total spikes: {results['total_spikes']}")
```

---

## Edge Detection Demo

The demo implements a neuromorphic vision pipeline on a synthetic event camera stream:

```
DVS Camera (16×16 synthetic sensor)
    ↓ temporal contrast events
Core 0: Spatial filter  (256 neurons, sparse Gabor-like weights)
    ↓ spikes
Core 1: Edge integrator (256 neurons, recurrent)
    ↓ spikes
Core 2: Readout layer   (256 neurons)
```

```bash
python examples/edge_detection_demo.py
```

Sample output:
```
Total events:  7,809  (ON: 4,017, OFF: 3,792)
Sparsity:      61.9% of (pixel,frame) slots silent
Total SOPs:    287,361
Energy:        6,781.7 nJ  (6.78 µJ)
vs GPU:        42.4× less energy
```

---

## Architecture

```
src/neuromorphic/
├── event_stream.py   # EventStream — sparse AER spike encoding
├── core.py           # NeuromorphicCore — LIF neuron population
├── router.py         # SynapticRouter — inter-core mesh routing
├── processor.py      # NeuromorphicProcessor — top-level orchestration
└── energy.py         # EnergyProfiler — SOP-based energy estimation

tests/
├── test_event_stream.py
├── test_core.py
├── test_router.py
├── test_processor.py
└── test_energy.py

examples/
└── edge_detection_demo.py  # DVS edge detection pipeline
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

56 tests, all passing.

---

## Key Design Decisions

**Event-driven, not clock-driven**: Neurons only update when they receive input. Silent network = zero compute cost (modelling Loihi's event-triggered core wakeup).

**Structured NumPy arrays for events**: 13 bytes/event vs 24 bytes for 3×int64. Matches AER wire protocol compactness.

**SOP counting as primary metric**: Not FLOPS, not inference latency — SOPs, because energy in neuromorphic hardware scales linearly with SOP count. This is the right abstraction.

**Configurable topology**: Pipeline, grid (TrueNorth torus-style), all-to-all, or custom — `SynapticRouter` handles all cases.

---

## References

- Davies et al. (2018). [Loihi: A neuromorphic manycore processor with on-chip learning.](https://doi.org/10.1109/MM.2018.112130359) *IEEE Micro.*
- Merolla et al. (2014). [A million spiking-neuron integrated circuit...](https://doi.org/10.1126/science.1254642) *Science, 345(6197).*
- Lichtsteiner et al. (2008). [A 128×128 120 dB 15 µs latency asynchronous temporal contrast vision sensor.](https://doi.org/10.1109/JSSC.2007.914337) *IEEE JSSC.*
- Mahowald & Douglas (1991). A silicon neuron. *Nature, 354.*

---

## License

MIT
