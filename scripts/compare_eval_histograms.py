#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.8.0",
#     "numpy>=1.26.0",
#     "seaborn>=0.13.0",
# ]
# ///
"""
Compare evaluation histograms from multiple datasets.

Usage:
    uv run scripts/compare_eval_histograms.py <file1.csv> <file2.csv> ... [--output output.png]

Arguments:
    file1.csv, file2.csv, ...: CSV files with bin_start,count columns
    --output: Optional output image path (default: shows interactive plot)

Example:
    # Compare three datasets
    uv run scripts/compare_eval_histograms.py hist-eleison.txt hist-anaphora.txt hist-mnestic.txt

    # Save to file
    uv run scripts/compare_eval_histograms.py hist-*.txt --output comparison.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# sns.set_style("whitegrid")

plt.rcParams["font.family"] = "TeX Gyre Heros"


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


def plot_comparison(
    datasets: list[tuple[str, np.ndarray, np.ndarray]],
    output_path: Path | None,
    bin_width: int,
) -> None:
    """Create and display/save comparison plot with curves."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use red-to-blue colormap
    cmap = plt.cm.coolwarm
    n_datasets = len(datasets)

    for i, (name, bins, counts) in enumerate(datasets):
        # Normalize counts to percentages for fair comparison
        total = counts.sum()
        percentages = (counts / total) * 100

        # Use bin centers for the curve
        bin_centers = bins + bin_width / 2

        # Interpolate color from red (0) to blue (1)
        color = cmap(i / (n_datasets - 1)) if n_datasets > 1 else cmap(0.5)
        ax.plot(
            bin_centers,
            percentages,
            label=f"{name}",
            color=color,
            linewidth=1.5,
            alpha=0.8,
        )

    max_eval = 3500
    ax.set_xlim(-50, max_eval)
    ax.set_ylim(-0.01, None)

    # Formatting
    ax.set_xlabel("Absolute evaluation", fontsize=12, fontweight="bold")
    ax.set_ylabel("Share of positions", fontsize=12, fontweight="bold")
    ax.set_yticks([])

    # Add zero line
    # ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="0 cp")

    ax.legend(loc="upper right", fontsize=10)

    ax.set_facecolor("white")
    fig.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(1.5)

    ax.tick_params(colors="black")
    ax.grid(False)

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

    # Parse arguments
    input_paths = []
    output_path = None
    bin_width = 10

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--output" and i + 1 < len(sys.argv):
            output_path = Path(sys.argv[i + 1])
            i += 2
        elif arg == "--bin-width" and i + 1 < len(sys.argv):
            bin_width = int(sys.argv[i + 1])
            i += 2
        else:
            input_paths.append(Path(arg))
            i += 1

    if not input_paths:
        print("Error: No input files specified", file=sys.stderr)
        sys.exit(1)

    # Load all datasets
    datasets = []
    for path in sorted(input_paths):
        if not path.exists():
            print(f"Error: Input file '{path}' does not exist", file=sys.stderr)
            sys.exit(1)

        print(f"Loading {path}...")
        bins, counts = load_histogram_data(path)
        # lambda-X.X-b800.txt -> λ = X.X
        if "lambda" in path.stem:
            name = f"λ = {path.stem[7:10]}"
        else:
            name = path.stem
        datasets.append((name, bins, counts))
        print(f"  {len(bins)} bins, {counts.sum():,} positions")

    print(f"\nGenerating comparison plot for {len(datasets)} datasets...")
    plot_comparison(datasets, output_path, bin_width)

    if not output_path:
        print("Close the plot window to exit.")


if __name__ == "__main__":
    main()
