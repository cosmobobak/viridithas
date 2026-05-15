# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "matplotlib",
# ]
# ///
"""
Plot statistics from the track! macro.

Reads JSON from stdin containing tracked value statistics and histograms,
then generates matplotlib visualisations.
"""

import json
import sys
import math
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore

palette = {
    "red": "#FF5000",
    "blue": "#50C0FF",
    "cyan": "#00A0E0",
    "grey": "#808080",
    "bg-a": "#101010",
    "bg-b": "#121212",
    "text": "#F0F0F0",
}

plt.rcParams["text.color"] = palette["text"]
plt.rcParams["axes.labelcolor"] = palette["text"]
plt.rcParams["axes.titlecolor"] = palette["text"]


def bucket_to_int_range(bucket: int) -> tuple[int, int] | None:
    """
    Convert bucket index to the integer range [lo, hi] that maps to it.
    Returns None if no integers map to this bucket.

    Bucket index is computed as: 65 + floor(log2(v^2)) for v > 0
    So for bucket b, we need integers v where floor(log2(v^2)) = b - 65.
    That means: 2^((b-65)/2) <= v < 2^((b-64)/2)
    """
    if bucket == 64:
        return (0, 0)
    elif bucket > 64:
        fine_log = bucket - 65
        lo = math.ceil(2 ** (fine_log / 2))
        hi = math.ceil(2 ** ((fine_log + 1) / 2)) - 1
        if lo > hi:
            return None
        return (lo, hi)
    else:
        fine_log = 63 - bucket
        lo = math.ceil(2 ** (fine_log / 2))
        hi = math.ceil(2 ** ((fine_log + 1) / 2)) - 1
        if lo > hi:
            return None
        return (-hi, -lo)


def format_int(v: int) -> str:
    """Format an integer for display."""
    if v == 0:
        return "0"
    abs_v = abs(v)
    sign = "-" if v < 0 else ""
    if abs_v >= 1_000_000:
        return f"{sign}{abs_v // 1_000_000}M"
    elif abs_v >= 1_000:
        return f"{sign}{abs_v // 1_000}K"
    else:
        return f"{sign}{abs_v}"


def bucket_to_label(bucket: int) -> str:
    """Convert bucket index to a human-readable label showing integer range."""
    int_range = bucket_to_int_range(bucket)
    if int_range is None:
        return ""  # empty bucket
    lo, hi = int_range
    if lo == hi:
        return format_int(lo)
    else:
        return format_int(lo)  # i could do something smart but whatever


def plot_tracked_value(ax, data: dict, index: int) -> None:
    """Plot a single tracked value's histogram on the given axes."""
    name: str = data["name"]
    count: int = data["count"]
    avg: int = data["avg"]
    avg_abs: int = data["avg_abs"]
    stddev: int = data["stddev"]
    min_val: int = data["min"]
    max_val: int = data["max"]
    histogram: list[int] = data["histogram"]

    if [0, 1] == [min_val, max_val]:
        return plot_tracked_boolean(ax, data, index)

    # find non-zero range
    nonzero_indices = [i for i, v in enumerate(histogram) if v > 0]
    if not nonzero_indices:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(name, fontsize=12)
        return

    lo, hi = min(nonzero_indices), max(nonzero_indices)
    # add some padding
    lo = max(0, lo - 1)
    hi = min(127, hi + 1)

    # filter to only buckets that can contain integer values
    all_buckets = list(range(lo, hi + 1))
    valid_buckets = [b for b in all_buckets if bucket_to_int_range(b) is not None]
    values = [histogram[b] for b in valid_buckets]

    if not values:
        ax.text(
            0.5,
            0.5,
            "No valid buckets",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(name, fontsize=12)
        return

    # Create bar chart
    ax.bar(
        range(len(valid_buckets)),
        values,
        color=palette["blue"],
        # edgecolor=palette["text"],
        # alpha=0.7,
    )

    # X-axis labels (show subset to avoid crowding)
    if len(valid_buckets) <= 16:
        tick_indices = list(range(len(valid_buckets)))
    else:
        step = max(1, len(valid_buckets) // 8)
        tick_indices = list(range(0, len(valid_buckets), step))

    ax.set_xticks(tick_indices)
    ax.set_xticklabels(
        [bucket_to_label(valid_buckets[i]) for i in tick_indices], fontsize=8
    )

    # Y-axis: use log scale if range is large
    if max(values) > 100 * min(v for v in values if v > 0):
        ax.set_yscale("log")

    # title with stats
    # name format is `file:line expr`, extract the expr part
    if " " in name:
        # split on first space to separate "file:line" from "expr"
        _, expr = name.split(" ", 1)
        # truncate long expressions
        short_name = expr if len(expr) <= 40 else expr[:37] + "…"
    else:
        short_name = name
    title = f"{short_name}\nn={count:,} μ={avg:.1f} |μ|={avg_abs:.1f}\nσ={stddev:.1f} min={min_val} max={max_val}"
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", labelsize=8)

    ax.set_facecolor(palette["bg-b"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(palette["text"])
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_color(palette["text"])
    ax.spines["bottom"].set_linewidth(1.5)

    ax.tick_params(colors=palette["text"])
    ax.grid(False)


def plot_tracked_boolean(ax, data: dict, index: int) -> None:
    """Plot a single tracked boolean's ratio on the given axes."""

    name: str = data["name"]
    count: int = data["count"]
    avg: int = data["avg"]

    # create pie chart
    labels = "true", "false"
    sizes = [avg * count, count - avg * count]
    ax.pie(
        sizes,
        labels=labels,
        colors=[palette["blue"], palette["red"]],
        radius=1,
        wedgeprops=dict(width=0.5, edgecolor="w"),
    )

    # title with stats
    # name format is `file:line expr`, extract the expr part
    if " " in name:
        # split on first space to separate "file:line" from "expr"
        _, expr = name.split(" ", 1)
        # truncate long expressions
        short_name = expr if len(expr) <= 40 else expr[:37] + "…"
    else:
        short_name = name
    title = f"{short_name}\nn={count:,} μ={avg:.1f}"
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", labelsize=8)

    ax.set_facecolor(palette["bg-b"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(palette["text"])
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_color(palette["text"])
    ax.spines["bottom"].set_linewidth(1.5)

    ax.tick_params(colors=palette["text"])
    ax.grid(False)


def main():
    # read JSON from stdin
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if not data:
        print("No tracked values to plot.", file=sys.stderr)
        sys.exit(0)

    # filter entries with no data
    data = [d for d in data if d.get("count", 0) > 0]

    if not data:
        print("No tracked values with data to plot.", file=sys.stderr)
        sys.exit(0)

    # determine grid size, prefer square-ish layouts
    n = len(data)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 3 * rows), facecolor=palette["bg-b"]
    )
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, d in enumerate(data):
        plot_tracked_value(axes[i], d, i)

    # hide unused subplots
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    # save to file
    output_path = Path("stats_plot.svg")
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}", file=sys.stderr)

    # show interactively
    plt.show()


if __name__ == "__main__":
    main()
