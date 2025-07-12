#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "qbstyles",
# ]
# ///

import matplotlib.pyplot as plt
import numpy as np
# from qbstyles import mpl_style

# mpl_style(dark=True)


def load_values(filename, prefix):
    """Load values from file with given prefix."""
    values = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith(f"{prefix}: "):
                try:
                    value = float(line.split(": ")[1].strip())
                    values.append(value)
                except (ValueError, IndexError):
                    continue
    return np.array(values)


def remap(values, scale):
    """Pass values through the sigmoid function, then scale them."""
    return 1 / (1 + np.exp(-values)) * scale


show_eval = False


def main():
    # Load data
    error_values = load_values("error_values.txt", "ERROR")
    eval_values = load_values("eval_values.txt", "EVAL")

    error_values = np.sqrt(remap(error_values, 0.25))
    eval_values = remap(eval_values, 1.0)

    print(f"Loaded {len(error_values)} error values")
    print(f"Loaded {len(eval_values)} evaluation values")

    # Create visualization - just density plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Density plots for both datasets
    ax.hist(
        error_values,
        bins=500,
        range=(0, 1 if show_eval else np.max(error_values)),
        density=True,
        alpha=0.6,
        label="Errors",
        # edgecolor="black",
    )
    if show_eval:
        ax.hist(
            eval_values,
            bins=500,
            range=(0, 1),
            density=True,
            alpha=0.6,
            label="Evaluations",
            # edgecolor="black",
        )
    ax.set_title("density plot of errors and evaluations")
    ax.set_xlabel("value (0 to 1 in sigmoid space)")
    ax.set_ylabel("density")
    ax.legend()

    plt.tight_layout()
    plt.savefig("density_plot.png", dpi=300, bbox_inches="tight")
    print("Visualization saved as 'density_plot.png'")

    # Statistics for both datasets
    for name, values in [("Error", error_values), ("Evaluation", eval_values)]:
        if len(values) > 0:
            print(f"\n{name} Statistics:")
            print(f"  Range: {values.min():.6f} to {values.max():.6f}")
            print(f"  Mean: {values.mean():.6f}")
            print(f"  Std: {values.std():.6f}")
        else:
            print(f"\n{name} Statistics: No data found")


if __name__ == "__main__":
    main()
