# ATA Photonics Engine

**ATA Photonics Engine is a simulation and benchmarking platform for photonic AI accelerator architectures, starting with MZI-based neural networks.**

Built by Meryem Atayeva / ATA Photonics.

---

## What this is

A clean Python platform for simulating, training, and benchmarking photonic neural networks — networks where matrix-vector multiplications are performed by light passing through Mach-Zehnder Interferometer (MZI) meshes on a silicon photonic chip, rather than by digital transistors.

This is not yet a fabricated chip. It is the computational foundation for one.

The first milestone question is:

> Can a photonic neural network match a small electronic model on standard classification tasks, while demonstrating a theoretical energy and latency advantage?

v0.4 answers that question with a systematic benchmark.

---

## Repository structure

```
ata-photonics-engine/
  README.md
  requirements.txt
  src/
    activations.py          # MZI cos²(φ/2) and electronic activations
    layers.py               # PhotonicLayer, ElectronicLayer, ReadoutLayer
    models.py               # PhotonicNN, ElectronicNN
    datasets.py             # XOR, Two-Moons, MNIST 0v1
    train.py                # Shared training runner
    benchmark.py            # v0.4 benchmark engine (main entry point)
    hardware.py             # Hardware realism constants and estimators
    legacy_v03_hardware_ready.py  # Frozen v0.3: Clements + noise + GDS
  results/                  # Benchmark outputs (generated)
  layouts/                  # GDS chip layouts (generated)
```

---

## Quickstart

```bash
pip install -r requirements.txt
cd src
python benchmark.py
```

Outputs written to `results/`:
- `benchmark_table.txt` — ASCII comparison table
- `benchmark_plots.png` — training curves, accuracy bars, decision surfaces
- `benchmark_results.npy` — raw results dict

---

## Architecture

### Photonic layer (optical domain)

Each neuron is a programmable phase shifter feeding a Mach-Zehnder Interferometer:

```
input x ──► [ phase shifter: φ = W·x + b ] ──► [ MZI: I = cos²(φ/2) ] ──► intensity
```

The weight `W` is encoded as a voltage on a thermo-optic phase shifter. Inference is performed by light — at the speed of photon propagation through silicon waveguides (~200 µm at 1550 nm), not the CPU clock.

### Electronic readout

Optical signals are detected by on-chip photodetectors, converted to current by transimpedance amplifiers, and passed through a sigmoid output layer. This is the standard hybrid architecture used in published photonic AI chips (Shen et al. 2017, Hamerly et al. 2019).

### Hardware realism model

| Parameter | Value | Notes |
|---|---|---|
| Insertion loss | 0.5 dB/MZI | Si photonics, 220 nm SOI |
| Thermal tuning | 20 mW/π | Thermo-optic phase shifter |
| Detector noise σ | 0.01 | Normalised photocurrent |
| Modulation bandwidth | 10 GHz | Thermo-optic limited |
| MZI footprint | 500 µm² | 50 × 10 µm |
| Wavelength | 1550 nm | Telecom C-band |
| Energy per MAC (photonic) | ~0.01 pJ | Theoretical |
| Energy per MAC (electronic) | ~1.0 pJ | 8-bit digital |

### Clements decomposition (v0.3+)

Trained weight matrices are decomposed into per-MZI phase angles `(θ, φ)` using the Clements mesh architecture (Clements et al., Optica 2016). These angles are the direct programming inputs to the physical chip's electro-optic drivers.

---

## Benchmark datasets

| Dataset | Task | Input dim | Samples |
|---|---|---|---|
| XOR | Binary classification | 2 | 4 |
| Two-Moons | Binary classification | 2 | 400 |
| MNIST 0v1 | Binary classification | 16 (PCA) | 500 |

---

## Version history

| Version | Description |
|---|---|
| v0.1 | Initial XOR demo, MZI activation |
| v0.2 | Trainable network, decision surface plots |
| v0.3 | Clements decomposition, noise-aware training, GDSFactory layout |
| v0.4 | Modular architecture, benchmark engine, 3 datasets, hardware realism |

---

## Product direction

ATA Photonics Engine is being developed as the simulation layer for a physical photonic AI accelerator chip. The roadmap:

1. **v0.4** (current) — benchmark platform, prove accuracy/energy tradeoff
2. **v0.5** — multi-layer photonic networks, coherent detection
3. **v0.6** — full Clements mesh training with backpropagation through phase decomposition
4. **v0.7** — tape-out candidate: GDS layout + foundry design rule check (DRC)
5. **v1.0** — first fabricated chip measurement vs simulation

---

## References

- Shen et al., "Deep learning with coherent nanophotonic circuits," *Nature Photonics* (2017)
- Hamerly et al., "Experimental investigation of performance differences between coherent and incoherent MZI networks," *Advanced Photonics* (2019)
- Clements et al., "An optimal design for universal multiport interferometers," *Optica* (2016)
- Cheng et al., "Silicon photonics codesign for deep learning," *IEEE* (2020)

---

*ATA Photonics — simulation → silicon*
