"""
Edge Detection Demo — Neuromorphic Event Camera Pipeline

This demo simulates a classic neuromorphic vision pipeline:

  DVS Camera (simulated)
       ↓ temporal contrast events
  Core 0: Spatial filter layer (ON/OFF polarity separation + Gabor-like weights)
       ↓ spikes
  Core 1: Edge integration layer (accumulates evidence across orientations)
       ↓ spikes
  Core 2: Readout layer (output edge map)

The input mimics a Dynamic Vision Sensor (DVS) observing a moving edge —
a bright vertical bar sweeping across a 16×16 pixel grid.

Key metrics reported:
  - Synaptic Operations (SOPs) — the neuromorphic compute unit
  - Energy estimate (pJ) — at Loihi's 23.6 pJ/SOP
  - Sparsity of activity
  - Energy advantage vs equivalent GPU MACs
"""

import sys
import os
import numpy as np

# Make sure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neuromorphic import (
    EventStream,
    NeuromorphicProcessor,
    EnergyProfiler,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Synthetic DVS Event Camera
# ──────────────────────────────────────────────────────────────────────────────

def generate_moving_bar_events(
    width: int = 16,
    height: int = 16,
    n_frames: int = 50,
    bar_width: int = 2,
    speed: float = 0.5,           # pixels per frame
    threshold: float = 0.3,
    dt_us: float = 1000.0,        # 1ms per frame (1 kHz, typical DVS rate)
    rng: np.random.Generator = None,
) -> EventStream:
    """
    Simulate a DVS camera observing a bright vertical bar moving left to right.

    The DVS fires ON events (+1) where luminance increases,
    OFF events (-1) where it decreases — only at scene changes.
    Static regions generate no events (the hallmark of DVS sparsity).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_pixels = width * height
    stream = EventStream(capacity=n_frames * n_pixels)

    # Luminance field: shape (height, width)
    lum = np.zeros((height, width), dtype=float)
    log_lum_prev = np.log(lum + 1e-9)

    for frame in range(n_frames):
        t_us = int(frame * dt_us)

        # Update bar position
        bar_x = int((frame * speed) % width)

        # New luminance: bar is bright (1.0), background dark (0.05) + noise
        new_lum = np.full((height, width), 0.05) + rng.normal(0, 0.01, (height, width))
        for bx in range(bar_x, min(bar_x + bar_width, width)):
            new_lum[:, bx] = 1.0 + rng.normal(0, 0.02, height)
        new_lum = np.clip(new_lum, 1e-9, None)

        # Temporal contrast: DVS fires on log luminance change
        log_lum = np.log(new_lum)
        diff = log_lum - log_lum_prev

        on_mask  = diff >=  threshold
        off_mask = diff <= -threshold

        on_ids  = np.where(on_mask.ravel())[0].astype(np.uint32)
        off_ids = np.where(off_mask.ravel())[0].astype(np.uint32)

        if len(on_ids):
            stream.append_batch(
                np.full(len(on_ids),  t_us, dtype=np.uint64),
                on_ids,
                np.ones(len(on_ids), dtype=np.int8),
            )
        if len(off_ids):
            stream.append_batch(
                np.full(len(off_ids), t_us, dtype=np.uint64),
                off_ids,
                np.full(len(off_ids), -1, dtype=np.int8),
            )

        log_lum_prev = log_lum
        lum = new_lum

    return stream, n_pixels


# ──────────────────────────────────────────────────────────────────────────────
# 2. Build the processor
# ──────────────────────────────────────────────────────────────────────────────

def build_edge_processor(n_pixels: int, rng: np.random.Generator) -> NeuromorphicProcessor:
    """
    3-core pipeline:
      Core 0: 256-neuron spatial filter (ON/OFF polarity → oriented edges)
      Core 1: 128-neuron edge integration
      Core 2: 64-neuron readout
    """
    proc = NeuromorphicProcessor(
        n_cores=3,
        neurons_per_core=256,
        dt_us=1000.0,
        tau_m_ms=10.0,       # fast membrane (edge detection needs temporal precision)
        v_threshold=0.8,
        energy_pj_per_sop=23.6,  # Loihi measured value
        rng=rng,
    )

    # ── Core 0: Spatial filter (ON polarity → core 0 neurons) ──
    # Gabor-like sparse connectivity: each input pixel connects to ~10% of neurons
    core0 = proc.core(0)
    W0 = np.zeros((n_pixels, 256))
    for i in range(n_pixels):
        # Each pixel fans out to ~25 neurons (10% of 256)
        targets = rng.choice(256, size=25, replace=False)
        weights = rng.exponential(0.08, size=25)
        W0[i, targets] = weights
    core0.set_weights(W0)

    # ── Core 1: Edge integration ──
    core1 = proc.core(1)
    W1 = np.zeros((256, 256))
    # Local recurrent + feedforward (30% connectivity)
    mask = rng.random((256, 256)) < 0.3
    W1[mask] = rng.exponential(0.06, size=mask.sum())
    core1.set_weights(W1)

    # ── Core 2: Readout ──
    core2 = proc.core(2)
    W2 = np.zeros((256, 256))
    mask = rng.random((256, 256)) < 0.2
    W2[mask] = rng.exponential(0.05, size=mask.sum())
    core2.set_weights(W2)

    # Pipeline connectivity: 0 → 1 → 2
    proc.router.connect_pipeline([0, 1, 2])

    return proc


# ──────────────────────────────────────────────────────────────────────────────
# 3. Run and Report
# ──────────────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(42)

    print("=" * 60)
    print("  Neuromorphic Edge Detection Demo")
    print("  Event Camera → 3-Core LIF Pipeline")
    print("=" * 60)

    # Generate synthetic DVS stream
    print("\n[1] Generating synthetic DVS event stream...")
    n_frames    = 80
    dt_us       = 1000.0
    duration_us = n_frames * dt_us

    event_stream, n_pixels = generate_moving_bar_events(
        width=16, height=16,
        n_frames=n_frames,
        bar_width=2,
        speed=0.4,
        threshold=0.25,
        dt_us=dt_us,
        rng=rng,
    )

    d = event_stream.data
    n_on  = int((d["polarity"] > 0).sum())
    n_off = int((d["polarity"] < 0).sum())
    sparsity = event_stream.sparsity(n_pixels, duration_us, bin_us=dt_us)

    print(f"  Grid:          16×16 pixels ({n_pixels} neurons)")
    print(f"  Duration:      {duration_us/1000:.0f} ms ({n_frames} frames @ 1kHz)")
    print(f"  Total events:  {len(event_stream):,}  (ON: {n_on:,}, OFF: {n_off:,})")
    print(f"  Sparsity:      {sparsity*100:.1f}% of (pixel,frame) slots silent")

    # Build processor
    print("\n[2] Building 3-core neuromorphic processor...")
    proc = build_edge_processor(n_pixels, rng)
    print(f"  {proc}")
    print(f"  Routing: 0 → 1 → 2 (pipeline)")

    # Run simulation
    print("\n[3] Running simulation...")
    results = proc.run(
        input_stream=event_stream,
        duration_us=duration_us,
        input_core_ids=[0],
        output_core_ids=[2],
    )

    # ── Results ──
    print("\n" + "=" * 60)
    print("  SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Steps simulated:     {results['n_steps']:,}")
    print(f"  Total spikes:        {results['total_spikes']:,}")
    print(f"  Mean spikes/step:    {results['mean_spikes_per_step']:.1f}")
    print(f"  Wall-clock time:     {results['wall_time_s']*1000:.1f} ms")
    print(f"  Speedup vs realtime: {results['speedup_vs_realtime']:.0f}×")

    print()
    print("  ── Core breakdown ──")
    for cs in results["core_stats"]:
        print(f"  Core {cs['core_id']}: {cs['total_spikes_out']:5} spikes out, "
              f"{cs['total_sops']:6} SOPs")

    print()
    print("  ── Energy Profile ──")
    profiler = proc.profiler
    total_sops   = results["total_sops"]
    energy_pj    = profiler.total_energy_pj
    energy_nj    = profiler.total_energy_nj

    print(f"  Total SOPs:          {total_sops:,}")
    print(f"  Energy (Loihi est.): {energy_pj:.1f} pJ  = {energy_nj:.4f} nJ")

    if duration_us > 0:
        power_uw = profiler.power_uw(duration_us)
        print(f"  Avg power:           {power_uw:.4f} µW")

    cmp = profiler.compare_to_gpu(n_equivalent_macs=total_sops)
    print()
    print("  ── vs GPU (dense MAC) ──")
    print(f"  Neuromorphic:        {cmp['neuromorphic_pj']:.1f} pJ")
    print(f"  Equivalent GPU:      {cmp['gpu_pj']:.1f} pJ  (1 nJ/MAC)")
    print(f"  Energy advantage:    {cmp['speedup_vs_gpu']:.1f}× less energy")

    # Output spikes from readout core
    output = proc.output_streams[2]
    print()
    print(f"  Readout (Core 2) output spikes: {len(output):,}")
    if len(output) > 0:
        od = output.data
        active_neurons = np.unique(od["neuron_id"])
        print(f"  Active output neurons: {len(active_neurons)} / 256")

    print()
    print(profiler.report(duration_us))
    print("\n[Done] ✓")


if __name__ == "__main__":
    main()
