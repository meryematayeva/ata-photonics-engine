"""
hardware.py
Hardware realism model for photonic AI accelerators.

All values are based on published silicon photonics process data
(220 nm SOI, thermo-optic phase shifters, 1550 nm wavelength).

References
----------
- Shen et al., "Deep learning with coherent nanophotonic circuits," Nature Photonics 2017
- Hamerly et al., "Experimental investigation of performance differences between
  coherent and incoherent MZI networks," Advanced Photonics 2019
- Cheng et al., "Silicon photonics codesign for deep learning," IEEE 2020
"""

import numpy as np


# ── Physical constants ────────────────────────────────────────────────────────

INSERTION_LOSS_DB_PER_MZI       = 0.5     # dB — per MZI, including connectors
THERMAL_TUNING_MW_PER_PHASE     = 20.0    # mW — per phase shifter at π phase
DETECTOR_NOISE_STD              = 0.01    # normalised photocurrent noise σ
MODULATION_BANDWIDTH_GHZ        = 10.0    # GHz — thermo-optic limited
WAVELENGTH_NM                   = 1550.0  # nm
MZI_FOOTPRINT_UM2               = 50 * 10 # µm²  (50 µm long, 10 µm wide)
ELECTRONIC_OP_ENERGY_PJ         = 1.0     # pJ per MAC (8-bit digital)
OPTICAL_OP_ENERGY_PJ            = 0.01    # pJ per MAC (theoretical photonic)
SPEED_OF_LIGHT_UM_PS            = 300.0   # µm/ps  (c in waveguide ≈ c/n_eff)
WAVEGUIDE_LENGTH_PER_MZI_UM     = 200.0   # µm end-to-end


# ── Derived estimators ────────────────────────────────────────────────────────

def insertion_loss_linear(n_mzis: int) -> float:
    """Total insertion loss as a linear factor (not dB)."""
    total_db = INSERTION_LOSS_DB_PER_MZI * n_mzis
    return 10 ** (-total_db / 10)


def tuning_power_mw(n_phase_shifters: int, mean_phase_fraction: float = 0.5) -> float:
    """
    Estimated static tuning power.

    Parameters
    ----------
    n_phase_shifters : int
        Total number of independently tunable phase shifters.
    mean_phase_fraction : float
        Average phase setting as a fraction of π (0=off, 1=full π).
        Defaults to 0.5 (uniform distribution assumption).
    """
    return THERMAL_TUNING_MW_PER_PHASE * n_phase_shifters * mean_phase_fraction


def inference_latency_ps(n_mzis_depth: int) -> float:
    """
    Optical propagation latency through the MZI mesh.
    Pure photonic latency — does not include electrical readout.

    Parameters
    ----------
    n_mzis_depth : int
        Number of MZI layers in the critical path (mesh depth).
    """
    total_length_um = WAVEGUIDE_LENGTH_PER_MZI_UM * n_mzis_depth
    return total_length_um / SPEED_OF_LIGHT_UM_PS


def energy_per_inference_pj(n_macs: int, mode: str = "photonic") -> float:
    """
    Estimated energy per forward pass.

    Parameters
    ----------
    n_macs : int  Multiply-accumulate operations in the forward pass.
    mode   : str  'photonic' or 'electronic'
    """
    pj_per_op = OPTICAL_OP_ENERGY_PJ if mode == "photonic" else ELECTRONIC_OP_ENERGY_PJ
    return n_macs * pj_per_op


def count_macs(layer_sizes: list) -> int:
    """
    Count multiply-accumulate operations in a dense feedforward network.
    layer_sizes = [input_dim, hidden1, hidden2, ..., output_dim]
    """
    total = 0
    for i in range(len(layer_sizes) - 1):
        total += layer_sizes[i] * layer_sizes[i + 1]
    return total


def count_phase_shifters(W: np.ndarray) -> int:
    """
    Number of MZI phase shifters needed to implement weight matrix W
    via the Clements decomposition.

    For an N×M matrix via SVD (U Σ Vt):
      - U is N×N unitary → N(N-1)/2 MZIs → 2 phase shifters each
      - Vt is M×M unitary → M(M-1)/2 MZIs → 2 phase shifters each
      - Σ: N attenuators (1 tunable coupler each)
    """
    n, m = W.shape
    k    = min(n, m)
    mzis_U  = n * (n - 1) // 2
    mzis_Vt = m * (m - 1) // 2
    return 2 * (mzis_U + mzis_Vt) + k   # ×2 because each MZI has θ and φ


def add_detector_noise(signal: np.ndarray, rng=None) -> np.ndarray:
    """Inject photodetector shot/thermal noise onto an optical signal."""
    if rng is None:
        rng = np.random.default_rng()
    return signal + rng.normal(0, DETECTOR_NOISE_STD, signal.shape)


def hardware_summary(layer_sizes: list, n_mzis_depth: int) -> dict:
    """
    Return a dict of key hardware metrics for a given network topology.

    Parameters
    ----------
    layer_sizes   : list  e.g. [2, 6, 1]
    n_mzis_depth  : int   critical-path MZI depth (≈ hidden layer width)
    """
    n_macs = count_macs(layer_sizes)
    # count phase shifters across all weight matrices
    n_ps   = sum(
        count_phase_shifters(np.zeros((layer_sizes[i], layer_sizes[i+1])))
        for i in range(len(layer_sizes) - 1)
    )
    latency_ps = inference_latency_ps(n_mzis_depth)
    return {
        "n_macs":               n_macs,
        "n_phase_shifters":     n_ps,
        "latency_ps":           round(latency_ps, 2),
        "latency_ns":           round(latency_ps / 1000, 4),
        "energy_photonic_pj":   round(energy_per_inference_pj(n_macs, "photonic"), 4),
        "energy_electronic_pj": round(energy_per_inference_pj(n_macs, "electronic"), 2),
        "tuning_power_mw":      round(tuning_power_mw(n_ps), 2),
        "chip_area_um2":        MZI_FOOTPRINT_UM2 * (n_ps // 2),
        "bandwidth_ghz":        MODULATION_BANDWIDTH_GHZ,
    }
