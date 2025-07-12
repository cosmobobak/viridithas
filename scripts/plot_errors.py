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

def load_error_values(filename):
    """Load error values from file, extracting numeric values."""
    values = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('ERROR: '):
                try:
                    value = float(line.split(': ')[1].strip())
                    values.append(value)
                except (ValueError, IndexError):
                    continue
    return np.array(values)

def main():
    # Load data
    error_values = load_error_values('error_values.txt')
    
    print(f"Loaded {len(error_values)} error values")
    print(f"Range: {error_values.min():.6f} to {error_values.max():.6f}")
    print(f"Mean: {error_values.mean():.6f}")
    print(f"Std: {error_values.std():.6f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Distribution of Benchmark Error Values', fontsize=16)
    # turn off grid for all axes
    # for ax in axes.flat:
    #     ax.grid(False)
    
    # Histogram
    axes[0, 0].hist(error_values, bins=100, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Histogram of Error Values')
    axes[0, 0].set_xlabel('Error Value')
    axes[0, 0].set_ylabel('Frequency')
    # axes[0, 0].grid(False)
    
    # Box plot
    axes[0, 1].boxplot(error_values)
    axes[0, 1].set_title('Box Plot of Error Values')
    axes[0, 1].set_ylabel('Error Value')
    # axes[0, 1].grid(False)
    
    # Density plot
    axes[1, 0].hist(error_values, bins=100, density=True, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Density Distribution')
    axes[1, 0].set_xlabel('Error Value')
    axes[1, 0].set_ylabel('Density')
    # axes[1, 0].grid(False)
    
    # Time series (sample)
    sample_size = min(10000, len(error_values))
    sample_indices = np.linspace(0, len(error_values)-1, sample_size, dtype=int)
    axes[1, 1].plot(error_values[sample_indices], alpha=0.6, linewidth=0.5)
    axes[1, 1].set_title(f'Error Values Over Time (Sample of {sample_size})')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Error Value')
    axes[1, 1].grid(False)
    
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'error_distribution.png'")
    
    # Additional statistics
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        print(f"{p}th: {np.percentile(error_values, p):.6f}")

if __name__ == "__main__":
    main()