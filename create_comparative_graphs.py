#!/usr/bin/env python3
"""
Comparative Graph Generator - Creates your 9 comparison graphs
"""

import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path


def create_all_comparative_graphs():
    """Create all 9 comparative graphs"""

    content_mappings = {
        'satellite': ['I', 'II', 'III'],
        'UAV': ['II', 'III', 'IV'],
        'grid': ['II', 'III', 'IV']
    }

    algorithms = ['LRU', 'Popularity', 'MAB_Original', 'MAB_Contextual', 'Federated_MAB']
    alpha_values = [0.25, 2.0]

    colors = {
        'LRU': '#FF6B6B',
        'Popularity': '#4ECDC4',
        'MAB_Original': '#45B7D1',
        'MAB_Contextual': '#96CEB4',
        'Federated_MAB': '#FFEAA7'
    }

    for alpha in alpha_values:
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'Retrieval Delay Comparison (α={alpha})', fontsize=18, fontweight='bold')

        plot_idx = 0
        for content_type in ['satellite', 'UAV', 'grid']:
            for category in content_mappings[content_type]:
                row = plot_idx // 3
                col = plot_idx % 3
                ax = axes[row, col]

                # Load data for each algorithm
                algorithm_data = {}
                for algorithm in algorithms:
                    pattern = f'delay_{algorithm}_{content_type}_{category}_{alpha}_*.txt'
                    files = glob.glob(pattern)

                    all_delays = []
                    for file in files:
                        try:
                            with open(file, 'r') as f:
                                delays = [float(line.strip()) for line in f if line.strip()]
                                all_delays.extend(delays)
                        except:
                            continue

                    if all_delays:
                        algorithm_data[algorithm] = {
                            'mean': np.mean(all_delays),
                            'std': np.std(all_delays)
                        }

                # Create bar chart
                if algorithm_data:
                    algs = list(algorithm_data.keys())
                    means = [algorithm_data[alg]['mean'] for alg in algs]
                    stds = [algorithm_data[alg]['std'] for alg in algs]
                    bar_colors = [colors[alg] for alg in algs]

                    bars = ax.bar(algs, means, yerr=stds, capsize=5,
                                  color=bar_colors, alpha=0.8, edgecolor='black')

                    ax.set_title(f'{content_type.upper()}-{category}', fontweight='bold')
                    ax.set_ylabel('Delay (seconds)')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3)

                    # Add value labels
                    for bar, mean_val in zip(bars, means):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height,
                                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)

                plot_idx += 1

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'comparative_analysis_alpha_{alpha}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created comparative_analysis_alpha_{alpha}.png")


if __name__ == "__main__":
    create_all_comparative_graphs()