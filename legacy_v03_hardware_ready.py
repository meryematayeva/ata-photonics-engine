"""
ATA Photonics Engine v0.3
Hardware-ready photonic neural network prototype.

New in this version:
  1. Clements decomposition  — converts trained weight matrices into
     per-MZI phase angles (theta, phi) that a real fabricated mesh
     can be programmed with.
  2. Noise-aware training    — injects fabrication-realistic phase
     noise during training so the model stays accurate on real hardware.
  3. GDSFactory layout       — generates an actual waveguide layout
     (GDS-II file) of the MZI mesh for tape-out or process simulation.

Author: Meryem Atayeva / ATA Photonics
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

# ── optional dependency warning ───────────────────────────────────────────────
try:
    import gdsfactory as gf
    HAS_GDS = True
except ImportError:
    HAS_GDS = False
    warnings.warn(
        "gdsfactory not installed. GDS layout export will be skipped.\n"
        "Install with: pip install gdsfactory",
        ImportWarning,
        stacklevel=2,
    )


# =============================================================================
# 1.  CORE PHOTONIC PRIMITIVES
# =============================================================================

def photonic_activation(phase):
    """MZI intensity: I = cos²(phase / 2)."""
    return np.cos(phase / 2) ** 2


def d_photonic_activation(phase):
    """Derivative: d/dφ [cos²(φ/2)] = -½ sin(φ)."""
    return -0.5 * np.sin(phase)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def d_sigmoid_from_output(y):
    return y * (1 - y)


# =============================================================================
# 2.  CLEMENTS DECOMPOSITION
# =============================================================================
#
# Any N×N unitary matrix U can be factored into a triangular mesh of 2×2
# beam-splitter / phase-shifter units (MZIs).  Each MZI is described by
# two angles: theta (internal phase — controls split ratio) and phi
# (external phase — controls output phase).
#
# Reference:
#   Clements et al., "An optimal design for universal multiport
#   interferometers," Optica 3, 1460 (2016).
#
# We first orthogonalise W via SVD (U Σ Vᵀ), then decompose the unitary
# factors.  The singular values Σ are absorbed into the bias / readout layer.

def _nullify(U, i, j, target_col):
    """
    Return (theta, phi) such that a single MZI acting on rows i, j
    zeroes U[j, target_col].

    The 2×2 MZI transfer matrix is:
        T = [[cos θ · e^{iφ},  -sin θ],
             [sin θ · e^{iφ},   cos θ]]   (simplified real-valued form)
    """
    a = U[i, target_col]
    b = U[j, target_col]
    theta = np.arctan2(np.abs(b), np.abs(a))
    phi   = np.angle(a) - np.angle(b)
    return float(theta), float(phi)


def clements_decompose(W):
    """
    Decompose an arbitrary real matrix W into a sequence of MZI settings
    using the Clements mesh architecture.

    Steps
    -----
    1.  SVD:  W ≈ U @ diag(sigma) @ Vt   (best unitary approximation)
    2.  Decompose U and Vt into MZI (theta, phi) lists.
    3.  Return the per-MZI angles alongside the singular values.

    Parameters
    ----------
    W : ndarray, shape (N, M)
        Weight matrix from a trained layer.

    Returns
    -------
    mzi_angles : list of dict
        Each entry: {'row_a': int, 'row_b': int,
                     'theta': float (rad), 'phi': float (rad),
                     'matrix': 'U' or 'Vt'}
    sigma : ndarray
        Singular values (program as intensity attenuators or absorb into
        the next layer's weights).
    U, Vt : ndarray
        Unitary factors for reference.
    """
    U, sigma, Vt = np.linalg.svd(W, full_matrices=False)
    n = U.shape[0]

    mzi_angles = []

    def _decompose_unitary(M, label):
        """
        Decompose a unitary/semi-unitary matrix M (shape p×q) into MZI
        angles by column-by-column nullification.  Skips degenerate cases
        where M has only one row or one column (nothing to nullify).
        """
        angles = []
        M = M.copy().astype(complex)
        p, q = M.shape
        # Number of nullification passes = min(p,q) - 1
        n_cols = min(p, q) - 1
        for col in range(n_cols):
            # Walk rows from bottom up to col+1
            for row in range(p - 1, col, -1):
                if row >= p or row - 1 < 0:
                    continue
                theta, phi = _nullify(M, row - 1, row, col)
                T = np.array([
                    [np.cos(theta) * np.exp(1j * phi), -np.sin(theta)],
                    [np.sin(theta) * np.exp(1j * phi),  np.cos(theta)],
                ])
                M[[row - 1, row], :] = T.conj().T @ M[[row - 1, row], :]
                angles.append({
                    'matrix': label,
                    'row_a':  row - 1,
                    'row_b':  row,
                    'theta':  round(theta, 6),
                    'phi':    round(phi,   6),
                })
        return angles

    mzi_angles += _decompose_unitary(U,  label='U')
    mzi_angles += _decompose_unitary(Vt, label='Vt')

    return mzi_angles, sigma, U, Vt


def print_mzi_program(mzi_angles, sigma, label="W1"):
    """Pretty-print the MZI programming table."""
    print(f"\n{'─'*60}")
    print(f"  Clements decomposition of {label}")
    print(f"{'─'*60}")
    print(f"  {'#':<4} {'Matrix':<8} {'Rows':<10} {'θ (rad)':<12} {'φ (rad)':<12}")
    print(f"  {'─'*56}")
    for i, m in enumerate(mzi_angles):
        rows = f"{m['row_a']}↔{m['row_b']}"
        print(f"  {i:<4} {m['matrix']:<8} {rows:<10} {m['theta']:<12.5f} {m['phi']:<12.5f}")
    print(f"\n  Singular values (intensity attenuators): {np.round(sigma, 4)}")
    print(f"{'─'*60}\n")


# =============================================================================
# 3.  NOISE-AWARE PHOTONIC NEURAL NETWORK
# =============================================================================

class PhotonicNeuralNetwork:
    """
    Trainable photonic neural network with optional phase noise injection.

    Architecture
    ------------
    Input (2) → Phase shifters (hidden_size MZIs) → MZI activation →
    Photodetector → Electronic sigmoid → Output (1)

    Noise model
    -----------
    During training, Gaussian noise N(0, sigma_noise²) is added to each
    computed phase.  This models:
      - electro-optic driver voltage quantisation
      - waveguide fabrication imperfections (~0.01–0.1 rad typical)
      - thermal drift

    A model trained with noise is more tolerant of real-hardware errors.
    """

    def __init__(
        self,
        input_size  = 2,
        hidden_size = 4,
        output_size = 1,
        seed        = 42,
        sigma_noise = 0.0,   # rad; 0 = noise-free, 0.05 = realistic fab
    ):
        rng = np.random.default_rng(seed)
        self.sigma_noise = sigma_noise

        self.W1 = rng.normal(0, 1.0, (input_size, hidden_size))
        self.b1 = rng.normal(0, 0.2, (1, hidden_size))
        self.W2 = rng.normal(0, 1.0, (hidden_size, output_size))
        self.b2 = rng.normal(0, 0.2, (1, output_size))

        self._rng = rng   # keep for noise sampling

    def forward(self, X, inject_noise=False):
        self.phase1 = X @ self.W1 + self.b1
        if inject_noise and self.sigma_noise > 0:
            noise = self._rng.normal(0, self.sigma_noise, self.phase1.shape)
            self.phase1 = self.phase1 + noise
        self.A1 = photonic_activation(self.phase1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def train(self, X, y, epochs=8000, lr=0.2, noise_epochs_ratio=0.5):
        """
        Train the network.

        noise_epochs_ratio : float
            Fraction of training epochs during which phase noise is
            injected.  Noise is ramped on after the first
            (1 - noise_epochs_ratio) fraction of epochs so the model
            first converges cleanly, then hardens to noise.
        """
        losses   = []
        n        = X.shape[0]
        noise_on = int(epochs * (1 - noise_epochs_ratio))

        for epoch in range(epochs):
            inject = (epoch >= noise_on) and (self.sigma_noise > 0)
            y_pred = self.forward(X, inject_noise=inject)

            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)

            d_loss  = 2 * (y_pred - y) / n
            d_Z2    = d_loss * d_sigmoid_from_output(y_pred)
            d_W2    = self.A1.T @ d_Z2
            d_b2    = np.sum(d_Z2, axis=0, keepdims=True)
            d_A1    = d_Z2 @ self.W2.T
            d_phase1= d_A1 * d_photonic_activation(self.phase1)
            d_W1    = X.T @ d_phase1
            d_b1    = np.sum(d_phase1, axis=0, keepdims=True)

            self.W2 -= lr * d_W2
            self.b2 -= lr * d_b2
            self.W1 -= lr * d_W1
            self.b1 -= lr * d_b1

        return losses

    def predict(self, X):
        probs = self.forward(X, inject_noise=False)
        return (probs >= 0.5).astype(int), probs

    def hardware_accuracy(self, X, y, n_trials=200):
        """
        Estimate on-chip accuracy by running forward passes with noise
        n_trials times and averaging the binary accuracy.
        """
        correct = 0
        for _ in range(n_trials):
            preds, _ = self.predict(X)   # noise only during training
            # simulate runtime noise separately
            phase_noisy = X @ self.W1 + self.b1
            phase_noisy += self._rng.normal(0, self.sigma_noise, phase_noisy.shape)
            A1_noisy = photonic_activation(phase_noisy)
            Z2_noisy = A1_noisy @ self.W2 + self.b2
            A2_noisy = sigmoid(Z2_noisy)
            correct  += np.mean(((A2_noisy >= 0.5).astype(int) == y).astype(float))
        return correct / n_trials


# =============================================================================
# 4.  GDSFACTORY LAYOUT
# =============================================================================

def build_mzi_mesh_layout(n_inputs, n_mzis, filename="ata_photonics_mesh.gds"):
    """
    Generate a GDS-II waveguide layout of the MZI mesh using GDSFactory.

    Layout description
    ------------------
    - n_inputs  straight input waveguides spaced 10 µm apart
    - n_mzis    Mach-Zehnder interferometers arranged in a rectangular
                mesh (two columns of MZIs mirroring the Clements topology)
    - Output straight waveguides leading to detector pads

    Each MZI is built from:
      ╔══ DC ══ phase arm ══ DC ══╗
      ║   (50/50 directional      ║
      ║    coupler, 10 µm gap)    ║
      ╚═══════════════════════════╝

    Parameters
    ----------
    n_inputs : int   Number of input waveguides (= hidden layer width)
    n_mzis   : int   Number of MZIs to place
    filename : str   Output GDS filename

    Returns
    -------
    component : gf.Component  The top-level layout component
    """
    if not HAS_GDS:
        print("  [GDS] gdsfactory not available — skipping layout export.")
        return None

    print(f"\n  Building GDS layout: {n_inputs} inputs, {n_mzis} MZIs …")

    c = gf.Component("ata_photonics_mesh")

    wg_spacing  = 10.0   # µm between waveguides
    mzi_length  = 50.0   # µm per MZI cell (phase arm + couplers)
    dc_length   = 10.0   # directional coupler length
    arm_length  = mzi_length - 2 * dc_length

    # ── input waveguides ────────────────────────────────────────────────────
    input_wgs = []
    for i in range(n_inputs):
        y_pos = i * wg_spacing
        wg = c << gf.components.straight(length=20.0)
        wg.move((0, y_pos))
        input_wgs.append(wg)

    # ── MZI cells ──────────────────────────────────────────────────────────
    mzi_refs = []
    n_cols   = (n_mzis + 1) // 2          # two MZIs per column pair
    x_cursor = 20.0

    for col in range(n_cols):
        row_offset = col % 2               # stagger odd/even columns
        for row in range(n_inputs - 1):
            if len(mzi_refs) >= n_mzis:
                break
            # Simple MZI: directional coupler → straight arm → coupler
            mzi = gf.Component(f"mzi_{col}_{row}")

            # Lower arm
            arm_lo = mzi << gf.components.straight(length=arm_length)
            arm_lo.move((dc_length, (row + row_offset) * wg_spacing))

            # Upper arm (phase-shifted — in hardware: heater deposited here)
            arm_hi = mzi << gf.components.straight(length=arm_length)
            arm_hi.move((dc_length, (row + row_offset + 1) * wg_spacing))

            # Input/output directional couplers (simplified as bends)
            for x_dc in [0, dc_length + arm_length]:
                dc_lo = mzi << gf.components.bend_circular(radius=3.0, angle=10)
                dc_lo.move((x_dc, (row + row_offset) * wg_spacing))
                dc_hi = mzi << gf.components.bend_circular(radius=3.0, angle=-10)
                dc_hi.move((x_dc, (row + row_offset + 1) * wg_spacing))

            ref = c << mzi
            ref.move((x_cursor, 0))
            mzi_refs.append(ref)

        x_cursor += mzi_length

    # ── output waveguides ───────────────────────────────────────────────────
    for i in range(n_inputs):
        y_pos = i * wg_spacing
        wg_out = c << gf.components.straight(length=20.0)
        wg_out.move((x_cursor, y_pos))

    # ── text label ──────────────────────────────────────────────────────────
    label = c << gf.components.text(
        text=f"ATA Photonics — {n_inputs}×{n_mzis} MZI mesh",
        size=3,
        layer=(1, 0),
    )
    label.move((0, -15))

    c.write_gds(filename)
    print(f"  [GDS] Layout written → {filename}")
    print(f"        Bounding box: {c.bbox}")
    return c


# =============================================================================
# 5.  TRAINING + EVALUATION
# =============================================================================

# XOR dataset
X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([[0.0], [1.0], [1.0], [0.0]])

print("=" * 60)
print("  ATA Photonics Engine v0.3")
print("=" * 60)

# ── A) Noise-free baseline ──────────────────────────────────────────────────
print("\n[1/3] Training noise-free model …")
model_clean = PhotonicNeuralNetwork(hidden_size=6, seed=7, sigma_noise=0.0)
losses_clean = model_clean.train(X, y, epochs=8000, lr=0.2)
preds_clean, probs_clean = model_clean.predict(X)
acc_clean = np.mean(preds_clean == y)
print(f"      Final loss : {losses_clean[-1]:.6f}")
print(f"      Accuracy   : {acc_clean*100:.1f}%")

# ── B) Noise-hardened model ─────────────────────────────────────────────────
SIGMA_NOISE = 0.05   # rad — realistic for Si photonics heater phase errors
print(f"\n[2/3] Training noise-hardened model (σ={SIGMA_NOISE} rad) …")
model_noisy = PhotonicNeuralNetwork(hidden_size=6, seed=7, sigma_noise=SIGMA_NOISE)
losses_noisy = model_noisy.train(X, y, epochs=8000, lr=0.2, noise_epochs_ratio=0.5)
preds_noisy, probs_noisy = model_noisy.predict(X)
acc_noisy = np.mean(preds_noisy == y)
hw_acc    = model_noisy.hardware_accuracy(X, y, n_trials=500)
print(f"      Final loss        : {losses_noisy[-1]:.6f}")
print(f"      Clean accuracy    : {acc_noisy*100:.1f}%")
print(f"      Simulated HW acc  : {hw_acc*100:.1f}% (500 noise trials)")

# ── C) Clements decomposition of both weight matrices ───────────────────────
print("\n[3/3] Clements MZI decomposition …")
mzi_W1, sigma_W1, U_W1, Vt_W1 = clements_decompose(model_noisy.W1)
mzi_W2, sigma_W2, U_W2, Vt_W2 = clements_decompose(model_noisy.W2)

print_mzi_program(mzi_W1, sigma_W1, label="W1 (input→hidden)")
print_mzi_program(mzi_W2, sigma_W2, label="W2 (hidden→output)")

# Summary table
print("  Predicted probabilities (noise-hardened model):")
print("  ┌──────────┬──────────┬──────────┬──────────┐")
print("  │ Input    │ Target   │ Pred.    │ Correct? │")
print("  ├──────────┼──────────┼──────────┼──────────┤")
for xi, yi, pi, pred in zip(X, y, probs_noisy, preds_noisy):
    inp  = f"[{int(xi[0])},{int(xi[1])}]"
    ok   = "✓" if pred[0] == yi[0] else "✗"
    print(f"  │ {inp:<8} │ {int(yi[0]):<8} │ {pi[0]:.4f}   │ {ok:<8} │")
print("  └──────────┴──────────┴──────────┴──────────┘")


# =============================================================================
# 6.  PLOTS
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("ATA Photonics Engine v0.3", fontsize=14, fontweight="bold")

# ── Training loss comparison ─────────────────────────────────────────────────
ax = axes[0]
ax.plot(losses_clean, label="Noise-free", color="#4C9BE8", linewidth=1.5)
ax.plot(losses_noisy, label=f"Noise-hardened (σ={SIGMA_NOISE})", color="#E85D4C",
        linewidth=1.5, linestyle="--")
ax.axvline(8000 * 0.5, color="#E85D4C", alpha=0.3, linestyle=":",
           label="Noise injection starts")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Training Loss")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Decision surface (noise-hardened model) ──────────────────────────────────
ax = axes[1]
x1g = np.linspace(-0.2, 1.2, 150)
x2g = np.linspace(-0.2, 1.2, 150)
xx, yy = np.meshgrid(x1g, x2g)
_, grid_probs = model_noisy.predict(np.c_[xx.ravel(), yy.ravel()])
zz = grid_probs.reshape(xx.shape)
cf = ax.contourf(xx, yy, zz, levels=30, cmap="RdYlBu_r", alpha=0.8)
plt.colorbar(cf, ax=ax, label="P(class=1)")
ax.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors="black", s=100,
           cmap="coolwarm", zorder=5)
ax.set_xlabel("x1"); ax.set_ylabel("x2")
ax.set_title("Decision Surface (noise-hardened)")
ax.grid(True, alpha=0.3)

# ── MZI phase angles (Clements programming) ──────────────────────────────────
ax = axes[2]
thetas_W1 = [m["theta"] for m in mzi_W1 if m["matrix"] == "U"]
thetas_W2 = [m["theta"] for m in mzi_W2 if m["matrix"] == "U"]
ax.bar(range(len(thetas_W1)), thetas_W1, color="#6B4FBB", alpha=0.7,
       label="W1 MZI θ angles")
ax.bar(range(len(thetas_W2)), thetas_W2, color="#E8903C", alpha=0.7,
       label="W2 MZI θ angles", bottom=[0]*max(len(thetas_W2), 1))
ax.axhline(np.pi / 4, color="gray", linestyle="--", linewidth=1,
           label="π/4 (50/50 split)")
ax.set_xlabel("MZI index")
ax.set_ylabel("θ (rad)")
ax.set_title("Clements MZI θ angles\n(hardware phase program)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ata_photonics_v03_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Plot saved → ata_photonics_v03_results.png")


# =============================================================================
# 7.  GDS LAYOUT EXPORT
# =============================================================================

mesh = build_mzi_mesh_layout(
    n_inputs = model_noisy.W1.shape[1],   # = hidden_size = 6
    n_mzis   = len(mzi_W1),               # number of MZIs from decomposition
    filename = "ata_photonics_mesh.gds",
)


# =============================================================================
# 8.  HARDWARE PROGRAMMING SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("  HARDWARE PROGRAMMING SUMMARY")
print("=" * 60)
print("""
  To program a fabricated chip:

  1. Load the GDS file into your photonic chip controller
     (e.g. via Labber, PyVISA, or a custom driver).

  2. For each MZI in the Clements table:
       voltage = phase_to_voltage(theta, phi, calibration_curve)
       driver.set_channel(mzi_index, voltage)

  3. Feed optical input via a fibre array connected to the
     input waveguides.  Modulate amplitude proportional to
     the input features x1, x2.

  4. Read photodetector currents at the output ports.
     Apply sigmoid to the largest output current for classification.

  5. If accuracy drops below target: re-run hardware_accuracy()
     with higher sigma_noise and retrain, then re-export phases.

  Typical Si photonics parameters:
    Phase shifter sensitivity  :  ~0.1 V / π  (thermo-optic)
    MZI footprint              :  ~50 × 10 µm
    Insertion loss per MZI     :  ~0.5 dB
    Operating wavelength       :  1550 nm
""")
print("=" * 60)
print("  ATA Photonics Engine v0.3 — complete.")
print("=" * 60)
