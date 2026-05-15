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
import matplotlib.ticker as mtick  # type: ignore

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
plt.rcParams["font.family"] = "Iosevka Signalis"


def format_int(v: int, sig_figs: int = 3) -> str:
    """Format an integer for display with the given number of significant figures."""
    if v == 0:
        return "0"
    sign = "-" if v < 0 else ""
    abs_v = abs(v)

    suffixes = ["", "K", "M", "G", "T", "P", "E"]
    magnitude = min(int(math.log10(abs_v)) // 3, len(suffixes) - 1)

    if magnitude == 0:
        return f"{sign}{abs_v}"

    scaled = abs_v / 1000**magnitude
    decimal_places = max(0, sig_figs - int(math.log10(scaled)) - 1)
    formatted = f"{scaled:.{decimal_places}f}"

    # Rounding may bump us into the next magnitude (e.g. 999_500 -> "1000K" -> "1.00M")
    if float(formatted) >= 1000 and magnitude < len(suffixes) - 1:
        magnitude += 1
        scaled /= 1000
        formatted = f"{scaled:.{sig_figs - 1}f}"

    return f"{sign}{formatted}{suffixes[magnitude]}"


def bucket_to_x_index(bucket: dict) -> int:
    if bucket["start"] == 0:
        assert bucket["end"] == 0
        return 0
    # average in log-space:
    s = math.log2(abs(bucket["start"]))
    e = math.log2(abs(bucket["end"]))
    m = (s + e) / 2
    x = int(round(math.copysign(math.exp2(m), bucket["start"])))
    return x


def alternate():
    while True:
        yield palette["blue"]
        yield palette["red"]


alt_map = alternate()


def plot_tracked_value(ax, data: dict, index: int) -> None:
    """Plot a single tracked value's histogram on the given axes."""
    name: str = data["name"]
    count: int = data["count"]
    avg: int = data["avg"]
    avg_abs: int = data["avg_abs"]
    stddev: int = data["stddev"]
    min_val: int = data["min"]
    max_val: int = data["max"]
    buckets: list[dict] = data["buckets"]

    if [0, 1] == [min_val, max_val]:
        return plot_tracked_boolean(ax, data, index)

    if not buckets:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(name, fontsize=12)
        return

    # normalise by bucket size
    values = [b["count"] / (abs(b["start"] - b["end"]) + 1) for b in buckets]
    # normalise by sum
    s = sum(values)
    values = [v / s for v in values]

    # Create bar chart
    color = next(alt_map)
    x = [bucket_to_x_index(b) for b in buckets]
    ax.plot(
        x,
        values,
        color=color,
        # edgecolor=palette["text"],
        # alpha=0.7,
    )
    ax.fill_between(x, values, color=color, alpha=0.55, linewidth=0, hatch="--")

    # X-axis labels (show subset to avoid crowding)
    if len(range(x[0], x[-1] + 1)) <= 16:
        tick_indices = list(range(x[0], x[-1] + 1))
    else:
        if x[0] < 0:
            lo = min(x[0], -x[-1])
            hi = max(-x[0], x[-1])
        else:
            lo = x[0]
            hi = x[-1]
        step = max(1, len(range(lo, hi + 1)) // 8)
        tick_indices = list(range(lo, hi + 1, step))
        pad = 0.03
        r = hi - lo + 1
        lo = lo - r * pad
        hi = hi + r * pad
        ax.set_xlim(lo, hi)

    ax.set_xticks(tick_indices)
    ax.set_xticklabels([format_int(i) for i in tick_indices], fontsize=8)

    # Y-axis: use log scale if range is large
    log = False
    # if max(values) > 100 * min(v for v in values if v > 0):
    #     log = True
    #     ax.set_yscale("log")
    # ax.set_xscale("log")

    # title with stats
    # name format is `file:line expr`, extract the expr part
    if " " in name:
        # split on first space to separate "file:line" from "expr"
        _, expr = name.split(" ", 1)
        # truncate long expressions
        short_name = expr if len(expr) <= 40 else expr[:37] + "…"
        short_name = short_name.replace("_", " ").upper()
    else:
        short_name = name
    title = f"{short_name}\nN={count:,} μ={avg:.1f} |μ|={avg_abs:.1f}\nσ={stddev:.1f} LO={min_val} HI={max_val}"
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", labelsize=12)

    ax.set_facecolor(palette["bg-b"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(palette["text"])
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_color(palette["text"])
    ax.spines["bottom"].set_linewidth(1.5)

    ax.tick_params(colors=palette["text"])
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.get_yaxis().set_ticks([])
    if log:
        ax.set_ylabel("LOG-DENSITY", fontsize=16)
    else:
        ax.set_ylabel("DENSITY", fontsize=16)

    ax.grid(False)


def plot_tracked_boolean(ax, data: dict, index: int) -> None:
    """Plot a single tracked boolean's ratio on the given axes."""

    name: str = data["name"]
    count: int = data["count"]
    avg: int = data["avg"]

    # create pie chart
    labels = ("TRUE", "FALSE")
    sizes = [avg * count, count - avg * count]
    wedges, _ = ax.pie(
        sizes,
        colors=[palette["blue"], palette["red"]],
        radius=1,
        startangle=120,
        wedgeprops=dict(width=0.5, edgecolor="w"),
    )

    # annotate each wedge with an arrow pointing outward, following the
    # matplotlib pie-and-donut-labels recipe:
    # https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_and_donut_labels.html
    bbox_props = dict(
        boxstyle="square,pad=0.3",
        fc=palette["text"],
        ec=palette["text"],
        lw=1.2,
    )
    kw = dict(
        arrowprops=dict(arrowstyle="-", color=palette["text"], lw=1.5),
        bbox=bbox_props,
        zorder=0,
        va="center_baseline",
        color=palette["bg-a"],
    )
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
        x = math.cos(math.radians(ang))
        y = math.sin(math.radians(ang))
        sign_x = 1 if x >= 0 else -1
        horizontalalignment = "left" if sign_x > 0 else "right"
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"]["connectionstyle"] = connectionstyle  # type: ignore
        ax.annotate(
            labels[i],
            xy=(x, y),
            xytext=(1.35 * sign_x, 1.4 * y),
            horizontalalignment=horizontalalignment,
            **kw,
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
    title = f"{short_name}\nN={count:,} μ={avg:.1f}"
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", labelsize=12)

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
        rows,
        cols,
        figsize=(16 / 3 * cols, 9 / 3 * rows),
        facecolor=palette["bg-b"],
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
