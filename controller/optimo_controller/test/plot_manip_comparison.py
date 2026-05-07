#!/usr/bin/env python3
"""Plot singularity handler ON vs OFF comparison.

Shows DLS+Handler ON vs NoDLS+Handler OFF to demonstrate the handler effect.
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/manip_comparison"

TRAJECTORY_LABELS = {
    "plato_deep": "15-DOF Arm+Hand\nFingertip Extension",
    "plato_sweep": "15-DOF Arm+Hand\nFingertip Lateral Sweep",
    "plato_progressive": "15-DOF Arm+Hand\nFingertip Reach Limit",
    "plato_circle": "15-DOF Arm+Hand\nFingertip Circle",
}

# Only plot these trajectories (skip 7-DOF Optimo)
INCLUDE_ONLY = set(TRAJECTORY_LABELS.keys())

# 2 variants to show: (filename_prefix, display_label, color, linestyle)
VARIANTS = [
    ("dls_handler_on",    "Singularity Handler ON",  "tab:blue",  "-"),
    ("nodls_handler_off", "Singularity Handler OFF", "tab:red",   "--"),
]


def compute_error(d):
    return np.sqrt(
        (d["ee_x"] - d["ee_x_des"]) ** 2
        + (d["ee_y"] - d["ee_y_des"]) ** 2
        + (d["ee_z"] - d["ee_z_des"]) ** 2
    ) * 1000


def load_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        reader = csv.DictReader(f)
        data = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k, v in row.items():
                try:
                    data[k].append(float(v))
                except ValueError:
                    data[k].append(v)
    for k in data:
        try:
            data[k] = np.array(data[k], dtype=float)
        except (ValueError, TypeError):
            pass
    return data


def has_column(d, col):
    return col in d and isinstance(d[col], np.ndarray)


def plot_all(traj_data, out_dir):
    """One figure: 3 rows (σ_min, task error, qdot) × N trajectories, 2 variants overlaid."""
    n = len(traj_data)
    nrows = 3  # σ_min, task error, qdot_raw
    fig, axes = plt.subplots(nrows, n, figsize=(5 * n, 3.2 * nrows), squeeze=False)
    fig.suptitle("Singularity Handler ON vs OFF", fontsize=14, fontweight="bold", y=1.01)

    for col, (name, label, variants) in enumerate(traj_data):
        # Row 0: σ_min
        ax = axes[0, col]
        for vname, vlabel, vdata, vcolor, vls in variants:
            ax.plot(vdata["time"], vdata["sigma_min"], color=vcolor, linestyle=vls,
                    linewidth=1.5, label=vlabel)
        ax.axhline(y=0.08, color="orange", linestyle=":", linewidth=1, label="threshold")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("σ_min", fontsize=10)
            ax.legend(fontsize=7, loc="lower left", ncol=1)
        mins = {vl: vd["sigma_min"].min() for _, vl, vd, _, _ in variants}
        txt = "\n".join(f"{k}: {v:.4f}" for k, v in mins.items())
        ax.text(0.97, 0.05, txt, transform=ax.transAxes, fontsize=7,
                va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # Row 1: Task space error
        ax = axes[1, col]
        for vname, vlabel, vdata, vcolor, vls in variants:
            err = compute_error(vdata)
            ax.plot(vdata["time"], err, color=vcolor, linestyle=vls, linewidth=1.5)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Task Space Error [mm]", fontsize=10)
        maxes = {vl: compute_error(vd).max() for _, vl, vd, _, _ in variants}
        txt = "\n".join(f"{k}: {v:.1f}" for k, v in maxes.items())
        ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=7,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # Row 2: Raw joint velocity norm (before clamp)
        ax = axes[2, col]
        for vname, vlabel, vdata, vcolor, vls in variants:
            key = "qdot_raw_norm" if has_column(vdata, "qdot_raw_norm") else "qdot_norm"
            if has_column(vdata, key):
                ax.plot(vdata["time"], np.clip(vdata[key], 0, 10),
                        color=vcolor, linestyle=vls, linewidth=1.2)
        ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, label="vel clamp")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("||qdot_raw||\n(capped 10)", fontsize=10)
            ax.legend(fontsize=7)
        ax.set_xlabel("Time [s]", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "comparison_handler.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def print_stats(traj_data):
    print("\n" + "=" * 110)
    print(f"{'Trajectory':<22} {'Variant':<28} {'σ_min range':<22} {'Err [mm]':<16} "
          f"{'Cond max':<12} {'qdot_raw max':<12}")
    print("-" * 110)
    for name, label, variants in traj_data:
        for vname, vlabel, vdata, _, _ in variants:
            err = compute_error(vdata)
            cond_max = vdata["cond_number"].max() if has_column(vdata, "cond_number") else 0
            raw_key = "qdot_raw_norm" if has_column(vdata, "qdot_raw_norm") else "qdot_norm"
            raw_max = vdata[raw_key].max() if has_column(vdata, raw_key) else 0
            print(f"{name:<22} {vlabel:<28} [{vdata['sigma_min'].min():.4f}, "
                  f"{vdata['sigma_min'].max():.4f}]   {err.mean():>6.1f}±{err.std():<6.1f}"
                  f"  {cond_max:>10.0f}  {raw_max:>10.2f}")
        print()
    print("=" * 110)


def main():
    subdirs = []
    for name in sorted(os.listdir(DATA_DIR)):
        d = os.path.join(DATA_DIR, name)
        if os.path.isdir(d):
            has_any = any(
                os.path.exists(os.path.join(d, f"{v[0]}.csv"))
                for v in VARIANTS
            )
            if has_any and name in INCLUDE_ONLY:
                subdirs.append(name)

    if not subdirs:
        print(f"ERROR: No CSV data found in {DATA_DIR}")
        sys.exit(1)

    traj_data = []
    for name in subdirs:
        d = os.path.join(DATA_DIR, name)
        label = TRAJECTORY_LABELS.get(name, name)
        variants = []
        for vfile, vlabel, vcolor, vls in VARIANTS:
            vdata = load_csv(os.path.join(d, f"{vfile}.csv"))
            if vdata is not None:
                variants.append((vfile, vlabel, vdata, vcolor, vls))
        if variants:
            traj_data.append((name, label, variants))

    if traj_data:
        print(f"Plotting {len(traj_data)} trajectories × 2 variants...")
        plot_all(traj_data, DATA_DIR)
        print_stats(traj_data)
    else:
        print("No data found.")


if __name__ == "__main__":
    main()
