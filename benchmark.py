"""
benchmark.py
ATA Photonics Engine v0.4 — Benchmark Engine

Compares PhotonicNN vs ElectronicNN across three datasets:
  - XOR
  - Two-Moons
  - MNIST 0 vs 1 (PCA-reduced)

Metrics reported per model per dataset:
  accuracy, final loss, inference time, estimated energy,
  estimated latency, parameter count, phase shifter count

Outputs
-------
  results/benchmark_table.txt   — human-readable ASCII table
  results/benchmark_plots.png   — 3-panel comparison figure
  results/benchmark_results.npy — raw results dict (numpy archive)

Run
---
  cd src && python benchmark.py
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── path setup (run from src/ or repo root) ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from datasets  import xor, two_moons, mnist_01, train_test_split
from models    import PhotonicNN, ElectronicNN
from train     import train_and_evaluate
from hardware  import (
    hardware_summary, count_phase_shifters,
    energy_per_inference_pj, inference_latency_ps,
    count_macs, INSERTION_LOSS_DB_PER_MZI,
    THERMAL_TUNING_MW_PER_PHASE, DETECTOR_NOISE_STD,
    MODULATION_BANDWIDTH_GHZ,
)

# ── configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SIGMA_NOISE = 0.05   # rad — realistic Si photonics phase error

DATASET_CONFIGS = {
    "XOR": {
        "loader":       xor,
        "loader_kwargs":{},
        "epochs":       8000,
        "lr":           0.2,
        "hidden_size":  6,
        "test_ratio":   None,   # XOR uses full set for both train/test
    },
    "Two-Moons": {
        "loader":       two_moons,
        "loader_kwargs":{"n_samples": 400, "noise": 0.15},
        "epochs":       6000,
        "lr":           0.15,
        "hidden_size":  8,
        "test_ratio":   0.2,
    },
    "MNIST 0v1": {
        "loader":       mnist_01,
        "loader_kwargs":{"n_components": 16, "n_samples": 500},
        "epochs":       6000,
        "lr":           0.1,
        "hidden_size":  16,
        "test_ratio":   0.2,
    },
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _hw_metrics(model, layer_sizes: list) -> dict:
    """Compute hardware cost metrics for a trained model."""
    hidden_size = layer_sizes[1]
    n_macs      = count_macs(layer_sizes)

    if hasattr(model, "n_phase_shifters"):
        n_ps = model.n_phase_shifters
        e_pj = energy_per_inference_pj(n_macs, "photonic")
        lat  = inference_latency_ps(hidden_size) / 1e3   # → ns
    else:
        n_ps = 0
        e_pj = energy_per_inference_pj(n_macs, "electronic")
        lat  = None   # electronic latency dominated by clock cycles, not propagation

    return {
        "n_params":         model.n_params,
        "n_phase_shifters": n_ps,
        "energy_pj":        round(e_pj, 4),
        "latency_ns":       round(lat, 4) if lat else "N/A",
    }


def _bar_color(model_type: str) -> str:
    return "#4C7BE8" if model_type == "Electronic" else "#E86A4C"


# ── main benchmark loop ───────────────────────────────────────────────────────

def run_benchmark() -> dict:
    all_results = {}

    print("\n" + "=" * 72)
    print("  ATA Photonics Engine v0.4 — Benchmark")
    print(f"  Hardware model: insertion_loss={INSERTION_LOSS_DB_PER_MZI} dB/MZI, "
          f"thermal={THERMAL_TUNING_MW_PER_PHASE} mW/π,")
    print(f"                  detector_noise_σ={DETECTOR_NOISE_STD}, "
          f"bandwidth={MODULATION_BANDWIDTH_GHZ} GHz")
    print("=" * 72)

    for ds_name, cfg in DATASET_CONFIGS.items():
        print(f"\n▶ Dataset: {ds_name}")
        print(f"  epochs={cfg['epochs']}, lr={cfg['lr']}, "
              f"hidden={cfg['hidden_size']}, σ_noise={SIGMA_NOISE}")

        # load data
        X, y, label = cfg["loader"](**cfg["loader_kwargs"])
        if cfg["test_ratio"]:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_ratio=cfg["test_ratio"]
            )
        else:
            X_tr = X_te = X
            y_tr = y_te = y

        in_dim  = X.shape[1]
        hidden  = cfg["hidden_size"]
        sizes   = [in_dim, hidden, 1]

        # ── Electronic ────────────────────────────────────────────────────────
        e_model = ElectronicNN(sizes, seed=42)
        e_res   = train_and_evaluate(e_model, X_tr, X_te, y_tr, y_te,
                                     epochs=cfg["epochs"], lr=cfg["lr"])
        e_hw    = _hw_metrics(e_model, sizes)

        # ── Photonic ──────────────────────────────────────────────────────────
        p_model = PhotonicNN(sizes, seed=42, sigma_noise=SIGMA_NOISE)
        p_res   = train_and_evaluate(p_model, X_tr, X_te, y_tr, y_te,
                                     epochs=cfg["epochs"], lr=cfg["lr"])
        p_hw    = _hw_metrics(p_model, sizes)

        all_results[ds_name] = {
            "label":      label,
            "sizes":      sizes,
            "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
            "electronic": {**e_res, **e_hw},
            "photonic":   {**p_res, **p_hw},
        }

        # quick progress line
        print(f"  Electronic → test acc {e_res['test_acc']*100:.1f}%  "
              f"loss {e_res['final_loss']:.5f}  "
              f"{e_res['inference_time_ms']:.3f} ms/inf")
        print(f"  Photonic   → test acc {p_res['test_acc']*100:.1f}%  "
              f"loss {p_res['final_loss']:.5f}  "
              f"{p_res['inference_time_ms']:.3f} ms/inf  "
              f"{p_hw['n_phase_shifters']} phase shifters")

    return all_results


# ── table formatter ───────────────────────────────────────────────────────────

def print_table(all_results: dict) -> str:
    col = {
        "Model":        14,
        "Dataset":      12,
        "Accuracy":     10,
        "Loss":         10,
        "Inf. (ms)":    11,
        "Energy (pJ)":  12,
        "Latency (ns)": 13,
        "Params":       8,
        "Phase shifters": 16,
    }
    sep  = "─" * (sum(col.values()) + len(col) + 1)
    hdr  = "│" + "│".join(k.center(v) for k, v in col.items()) + "│"
    rows = [sep, hdr, sep]

    for ds_name, res in all_results.items():
        for mtype in ("Electronic", "Photonic"):
            r = res[mtype.lower()]
            ps = str(r["n_phase_shifters"]) if r["n_phase_shifters"] else "—"
            lat = str(r["latency_ns"]) if r["latency_ns"] != "N/A" else "—"
            cells = [
                mtype.center(col["Model"]),
                ds_name.center(col["Dataset"]),
                f"{r['test_acc']*100:.1f}%".center(col["Accuracy"]),
                f"{r['final_loss']:.5f}".center(col["Loss"]),
                f"{r['inference_time_ms']:.3f}".center(col["Inf. (ms)"]),
                f"{r['energy_pj']:.4f}".center(col["Energy (pJ)"]),
                lat.center(col["Latency (ns)"]),
                str(r["n_params"]).center(col["Params"]),
                ps.center(col["Phase shifters"]),
            ]
            rows.append("│" + "│".join(cells) + "│")
        rows.append(sep)

    table = "\n".join(rows)
    print("\n" + table)
    return table


# ── plots ─────────────────────────────────────────────────────────────────────

def make_plots(all_results: dict, out_path: str):
    ds_names  = list(all_results.keys())
    n_ds      = len(ds_names)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "ATA Photonics Engine v0.4 — Photonic vs Electronic Benchmark",
        fontsize=15, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, n_ds, figure=fig, hspace=0.45, wspace=0.35)

    for col_i, ds_name in enumerate(ds_names):
        res   = all_results[ds_name]
        e     = res["electronic"]
        p     = res["photonic"]

        # ── row 0: training loss curves ──────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col_i])
        ax0.plot(e["losses"], color="#4C7BE8", linewidth=1.2, label="Electronic")
        ax0.plot(p["losses"], color="#E86A4C", linewidth=1.2,
                 linestyle="--", label="Photonic")
        ax0.set_title(f"{ds_name}\nTraining Loss", fontsize=10)
        ax0.set_xlabel("Epoch", fontsize=8)
        ax0.set_ylabel("MSE", fontsize=8)
        ax0.legend(fontsize=7)
        ax0.grid(True, alpha=0.3)

        # ── row 1: accuracy & energy bar chart ───────────────────────────────
        ax1 = fig.add_subplot(gs[1, col_i])
        labels   = ["Electronic", "Photonic"]
        accs     = [e["test_acc"] * 100, p["test_acc"] * 100]
        energies = [e["energy_pj"], p["energy_pj"]]
        colors   = ["#4C7BE8", "#E86A4C"]

        x = np.arange(2)
        bars = ax1.bar(x, accs, color=colors, alpha=0.85, width=0.5, zorder=3)
        ax1.set_ylim(0, 115)
        ax1.set_ylabel("Test Accuracy (%)", fontsize=8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=8)
        ax1.set_title("Accuracy", fontsize=10)
        ax1.grid(True, axis="y", alpha=0.3, zorder=0)
        for bar, acc in zip(bars, accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{acc:.1f}%", ha="center", va="bottom", fontsize=8)

        # energy as secondary axis
        ax1b = ax1.twinx()
        ax1b.plot(x, energies, "D--", color="#444", markersize=6, linewidth=1,
                  label="Energy (pJ)")
        ax1b.set_ylabel("Est. Energy (pJ)", fontsize=7, color="#444")
        ax1b.tick_params(axis="y", labelcolor="#444", labelsize=7)
        ax1b.legend(fontsize=7, loc="upper right")

        # ── row 2: decision surface (2D datasets) or param table (MNIST) ──
        ax2 = fig.add_subplot(gs[2, col_i])

        if res["X_tr"].shape[1] == 2:
            # decision surface for 2D datasets
            x1g = np.linspace(-0.05, 1.05, 120)
            x2g = np.linspace(-0.05, 1.05, 120)
            xx, yy = np.meshgrid(x1g, x2g)
            grid  = np.c_[xx.ravel(), yy.ravel()]
            _, gp = p_model_ref[ds_name].predict(grid) if False else \
                    _get_probs(all_results, ds_name, grid)
            zz = gp.reshape(xx.shape)
            cf = ax2.contourf(xx, yy, zz, levels=20, cmap="RdYlBu_r", alpha=0.75)
            plt.colorbar(cf, ax=ax2, label="P(class=1)", pad=0.02)
            X_all = np.vstack([res["X_tr"], res["X_te"]])
            y_all = np.vstack([res["y_tr"], res["y_te"]])
            ax2.scatter(X_all[:, 0], X_all[:, 1], c=y_all.flatten(),
                        cmap="coolwarm", edgecolors="k", s=18, linewidths=0.4,
                        zorder=5)
            ax2.set_title("Photonic decision surface", fontsize=10)
            ax2.set_xlabel("x1", fontsize=8); ax2.set_ylabel("x2", fontsize=8)
        else:
            # for MNIST (16-D) draw a metric comparison table
            ax2.axis("off")
            metrics = [
                ["Metric", "Electronic", "Photonic"],
                ["Accuracy",  f"{e['test_acc']*100:.1f}%",  f"{p['test_acc']*100:.1f}%"],
                ["Loss",      f"{e['final_loss']:.4f}",      f"{p['final_loss']:.4f}"],
                ["Inf (ms)",  f"{e['inference_time_ms']:.3f}",f"{p['inference_time_ms']:.3f}"],
                ["Energy pJ", f"{e['energy_pj']:.4f}",      f"{p['energy_pj']:.4f}"],
                ["Params",    str(e["n_params"]),             str(p["n_params"])],
                ["PhaseShifters", "—",                       str(p["n_phase_shifters"])],
            ]
            tbl = ax2.table(cellText=metrics[1:], colLabels=metrics[0],
                            cellLoc="center", loc="center",
                            bbox=[0.0, 0.0, 1.0, 1.0])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            for (r, c), cell in tbl.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#DDEEFF")
                    cell.set_text_props(weight="bold")
                cell.set_edgecolor("#CCCCCC")
            ax2.set_title("Metric summary", fontsize=10)

    path = os.path.join(RESULTS_DIR, "benchmark_plots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved → {path}")


def _get_probs(all_results, ds_name, grid):
    """Re-run photonic model forward pass for decision surface grid."""
    res    = all_results[ds_name]
    sizes  = res["sizes"]
    cfg    = DATASET_CONFIGS[ds_name]
    model  = PhotonicNN(sizes, seed=42, sigma_noise=SIGMA_NOISE)
    X_tr   = res["X_tr"]; y_tr = res["y_tr"]
    model.train(X_tr, y_tr, epochs=cfg["epochs"], lr=cfg["lr"])
    return model.predict(grid)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_benchmark()

    table_str = print_table(results)

    table_path = os.path.join(RESULTS_DIR, "benchmark_table.txt")
    with open(table_path, "w") as f:
        f.write("ATA Photonics Engine v0.4 — Benchmark Results\n\n")
        f.write(table_str)
        f.write("\n\nHardware realism parameters:\n")
        f.write(f"  insertion_loss_db_per_mzi       = {INSERTION_LOSS_DB_PER_MZI}\n")
        f.write(f"  thermal_tuning_mw_per_phase_shifter = {THERMAL_TUNING_MW_PER_PHASE}\n")
        f.write(f"  detector_noise_std              = {DETECTOR_NOISE_STD}\n")
        f.write(f"  modulation_bandwidth_ghz        = {MODULATION_BANDWIDTH_GHZ}\n")
    print(f"  Table saved → {table_path}")

    try:
        make_plots(results, RESULTS_DIR)
    except Exception as e:
        print(f"  [plots] Skipped: {e}")

    np.save(os.path.join(RESULTS_DIR, "benchmark_results.npy"),
            results, allow_pickle=True)
    print(f"  Raw results → {RESULTS_DIR}/benchmark_results.npy")

    print("\n" + "=" * 72)
    print("  v0.4 benchmark complete.")
    print("=" * 72 + "\n")
