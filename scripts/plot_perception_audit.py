"""Create compact diagnostic plots from perception DDI audit CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-csv", required=True)
    parser.add_argument("--component-audit-csv", required=True)
    parser.add_argument("--object-threshold-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(args.samples_csv)
    components = pd.read_csv(args.component_audit_csv)
    objects = pd.read_csv(args.object_threshold_csv)

    labels = {
        "count_score": "Object count",
        "visibility_mean_score": "Visibility mean",
        "visibility_max_score": "Visibility maximum",
        "lidar_sparsity_mean_score": "LiDAR sparsity mean",
        "lidar_sparsity_max_score": "LiDAR sparsity maximum",
        "truncation_mean_score": "Truncation mean",
        "truncation_max_score": "Truncation maximum",
    }
    plot_components = components.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(
        [labels.get(value, value) for value in plot_components["component"]],
        plot_components["max_bin_pct"],
        color=["#c23b4a" if value >= 80 else "#2f6f8f" for value in plot_components["max_bin_pct"]],
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.axvline(80, color="#8a1c2c", linestyle="--", linewidth=1, label="80% saturation warning")
    ax.set_xlim(0, 108)
    ax.set_xlabel("Share of keyframes in the component's most common extreme bin")
    ax.set_title("Perception DDI component saturation (nuScenes mini diagnostic)")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "component_saturation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(samples["perception_difficulty_score"], bins=18, color="#2f6f8f", edgecolor="white")
    ax.axvline(33.333, color="#687076", linestyle="--", linewidth=1.2, label="Fixed Low/Medium")
    ax.axvline(66.667, color="#c23b4a", linestyle="--", linewidth=1.2, label="Fixed Medium/High")
    ax.set_xlabel("Composite perception difficulty score")
    ax.set_ylabel("Keyframes")
    ax.set_title("Fixed 0-100 cutoffs leave the Low group empty (nuScenes mini)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "composite_score_distribution.png", dpi=180)
    plt.close(fig)

    lidar = objects[objects["family"] == "lidar_points_team"].copy()
    order = ["0-4", "5-19", "20-49", "50+"]
    lidar["bin"] = pd.Categorical(lidar["bin"], categories=order, ordered=True)
    lidar = lidar.sort_values("bin")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(lidar["bin"].astype(str), lidar["share_pct"], color=["#c23b4a", "#d58b3c", "#5b8c6a", "#2f6f8f"])
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylim(0, max(lidar["share_pct"]) * 1.18)
    ax.set_xlabel("LiDAR points inside a ground-truth 3D box")
    ax.set_ylabel("Share of objects")
    ax.set_title("Proposed LiDAR point bins (nuScenes mini diagnostic)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "lidar_point_bins.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
