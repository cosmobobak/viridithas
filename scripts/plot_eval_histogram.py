#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.8.0",
#     "numpy>=1.26.0",
# ]
# ///
"""
Plot evaluation histogram from eval-stats output.

Usage:
    uv run scripts/plot_eval_histogram.py <histogram_data.csv> [output.png]

Arguments:
    histogram_data.csv: CSV file with bin_start,count columns
    output.png: Optional output image path (default: shows interactive plot)

Example:
    # Generate histogram data
    ./target/release/viridithas eval-stats input.epd --histogram hist.csv

    # Plot it
    uv run scripts/plot_eval_histogram.py hist.csv

    # Save to file
    uv run scripts/plot_eval_histogram.py hist.csv eval_dist.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_histogram_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load histogram data from CSV file."""
    bins = []
    counts = []

    with open(csv_path) as f:
        # Skip header
        next(f)

        for line in f:
            line = line.strip()
            if not line:
                continue

            bin_start, count = line.split(",")
            bins.append(int(bin_start))
            counts.append(int(count))

    return np.array(bins), np.array(counts)


def plot_histogram(
    bins: np.ndarray, counts: np.ndarray, output_path: Path | None, bin_width: int
) -> None:
    """Create and display/save histogram plot."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create bar plot
    ax.bar(
        bins,
        counts,
        width=bin_width * 0.9,
        align="edge",
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )

    # Formatting
    ax.set_xlabel("Evaluation (centipawns)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title("Distribution of Static Evaluations", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add statistics text
    total_positions = counts.sum()
    mean_eval = np.average(bins + bin_width / 2, weights=counts)
    median_idx = np.searchsorted(np.cumsum(counts), total_positions / 2)
    median_eval = bins[median_idx] if median_idx < len(bins) else bins[-1]

    stats_text = f"Total positions: {total_positions:,}\n"
    stats_text += f"Mean eval: {mean_eval:.1f} cp\n"
    stats_text += f"Median eval: {median_eval:.1f} cp"

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Add zero line
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="0 cp")
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    grain = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Loading histogram data from {input_path}...")
    bins, counts = load_histogram_data(input_path)

    print(f"Loaded {len(bins)} bins with {counts.sum():,} total positions")
    print(f"Evaluation range: {bins.min():.0f} to {bins.max():.0f} centipawns")

    print("Generating plot...")
    plot_histogram(bins, counts, output_path, grain)

    if not output_path:
        print("Close the plot window to exit.")


if __name__ == "__main__":
    main()
