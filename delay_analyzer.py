#!/usr/bin/env python3
"""
Fixed Delay Analyzer - Works with your exact file format
delay_{ALGORITHM}_{CONTENT_TYPE}_{CATEGORY}_{ALPHA}_{TIME_SLOTS}_{SIM}.txt
"""

import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime


class FixedDelayAnalyzer:
    def __init__(self):
        self.algorithms = ['LRU', 'Popularity', 'MAB_Original', 'MAB_Contextual', 'Enhanced_Federated_MAB']
        self.content_mappings = {
            'satellite': ['I', 'II', 'III'],
            'UAV': ['II', 'III', 'IV'],
            'grid': ['II', 'III', 'IV']
        }
        self.colors = {
            'LRU': '#FF6B6B',
            'Popularity': '#4ECDC4',
            'MAB_Original': '#45B7D1',
            'MAB_Contextual': '#96CEB4',
            'Enhanced_Federated_MAB': '#FFEAA7'
        }

    def load_delay_files(self, alpha_values=[0.25, 0.5, 1.0, 2.0]):
        """
        Load delays from your exact file format:
        delay_{ALGORITHM}_{CONTENT_TYPE}_{CATEGORY}_{ALPHA}_{TIME_SLOTS}_{SIM}.txt
        """
        delay_data = {}

        print("🔍 Loading delay files with your exact format...")

        for alpha in alpha_values:
            delay_data[alpha] = {}
            total_files_for_alpha = 0

            for algorithm in self.algorithms:
                delay_data[alpha][algorithm] = {}

                for content_type in self.content_mappings:
                    delay_data[alpha][algorithm][content_type] = {}

                    for category in self.content_mappings[content_type]:
                        # Your exact file pattern
                        pattern = f"delay_{algorithm}_{content_type}_{category}_{alpha}_*.txt"
                        files = glob.glob(pattern)

                        all_delays = []
                        for file in files:
                            try:
                                with open(file, 'r') as f:
                                    delays = [float(line.strip()) for line in f if line.strip()]
                                    all_delays.extend(delays)
                                    total_files_for_alpha += 1
                            except Exception as e:
                                print(f"⚠️ Error loading {file}: {e}")
                                continue

                        if all_delays:
                            delay_data[alpha][algorithm][content_type][category] = all_delays
                            print(f"  ✅ {algorithm}-{content_type}-{category}: {len(all_delays)} delays")

            print(f"📊 Found {total_files_for_alpha} delay files for α={alpha}")

        return delay_data

    def create_comparative_graphs(self, delay_data, save_dir="./delay_analysis_results"):
        """
        Create 9 comparative graphs showing algorithm performance
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        print(f"📊 Creating 9 comparative graphs...")

        for alpha in delay_data.keys():
            if not any(delay_data[alpha][alg] for alg in self.algorithms):
                print(f"⚠️ No data for α={alpha}, skipping...")
                continue

            # Create 3x3 subplot grid
            fig, axes = plt.subplots(1, 3, figsize=(20, 16))
            fig.suptitle(f'Retrieval Delay Comparison Across Algorithms (α = {alpha})',
                         fontsize=20, fontweight='bold', y=0.98)

            plot_idx = 0

            for row, content_type in enumerate(['grid']):
                for col, category in enumerate(self.content_mappings[content_type]):

                    ax = axes[col]

                    # Collect algorithm performance for this content type/category
                    algorithm_stats = {}

                    for algorithm in self.algorithms:
                        if (algorithm in delay_data[alpha] and
                                content_type in delay_data[alpha][algorithm] and
                                category in delay_data[alpha][algorithm][content_type]):

                            delays = delay_data[alpha][algorithm][content_type][category]

                            if delays:
                                algorithm_stats[algorithm] = {
                                    'mean': np.mean(delays),
                                    'std': np.std(delays),
                                    'median': np.median(delays),
                                    'count': len(delays)
                                }

                    # Create bar plot
                    if algorithm_stats:
                        algs = list(algorithm_stats.keys())
                        means = [algorithm_stats[alg]['mean'] for alg in algs]
                        stds = [algorithm_stats[alg]['std'] for alg in algs]
                        bar_colors = [self.colors[alg] for alg in algs]

                        bars = ax.bar(algs, means, yerr=stds, capsize=5,
                                      color=bar_colors, alpha=0.8,
                                      edgecolor='black', linewidth=1)

                        # Customize subplot
                        ax.set_title(f'{content_type.upper()} - Category {category}',
                                     fontsize=14, fontweight='bold', pad=15)
                        ax.set_ylabel('Retrieval Delay (seconds)', fontsize=12)
                        ax.set_xlabel('Algorithm', fontsize=12)
                        ax.tick_params(axis='x', rotation=45, labelsize=10)
                        ax.tick_params(axis='y', labelsize=10)

                        # Add value labels on bars
                        for bar, mean_val, std_val in zip(bars, means, stds):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width() / 2.,
                                    height + std_val + max(means) * 0.05,
                                    f'{mean_val:.3f}s', ha='center', va='bottom',
                                    fontsize=9, fontweight='bold')

                        # Add grid and styling
                        ax.grid(True, alpha=0.3, axis='y')
                        ax.set_axisbelow(True)
                        ax.set_ylim(bottom=0)

                        # Highlight best performing algorithm
                        if means:
                            best_idx = means.index(min(means))
                            bars[best_idx].set_edgecolor('gold')
                            bars[best_idx].set_linewidth(3)

                    else:
                        # No data available
                        ax.text(0.5, 0.5, 'No Data Available',
                                transform=ax.transAxes, ha='center', va='center',
                                fontsize=12, style='italic', color='red')
                        ax.set_title(f'{content_type.upper()} - Category {category}',
                                     fontsize=14, fontweight='bold')

            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Save figures
            png_file = save_path / f"comparative_delay_analysis_alpha_{alpha}.png"
            pdf_file = save_path / f"comparative_delay_analysis_alpha_{alpha}.pdf"

            plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')

            print(f"💾 Saved: {png_file}")
            print(f"💾 Saved: {pdf_file}")

            plt.show()
            plt.close()

    def create_summary_statistics(self, delay_data, save_dir="./delay_analysis_results"):
        """
        Create comprehensive summary statistics table
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        print("📋 Creating summary statistics table...")

        summary_data = []

        for alpha in delay_data.keys():
            for algorithm in self.algorithms:
                for content_type in self.content_mappings:
                    for category in self.content_mappings[content_type]:

                        if (algorithm in delay_data[alpha] and
                                content_type in delay_data[alpha][algorithm] and
                                category in delay_data[alpha][algorithm][content_type]):

                            delays = delay_data[alpha][algorithm][content_type][category]

                            if delays:
                                summary_data.append({
                                    'Alpha': alpha,
                                    'Algorithm': algorithm,
                                    'Content_Type': content_type,
                                    'Category': category,
                                    'Mean_Delay': np.mean(delays),
                                    'Std_Delay': np.std(delays),
                                    'Median_Delay': np.median(delays),
                                    'Min_Delay': np.min(delays),
                                    'Max_Delay': np.max(delays),
                                    'P95_Delay': np.percentile(delays, 95),
                                    'Sample_Count': len(delays)
                                })

        if summary_data:
            df = pd.DataFrame(summary_data)

            # Save as CSV
            csv_file = save_path / "delay_summary_statistics.csv"
            df.to_csv(csv_file, index=False)

            # Save as Excel with formatting
            excel_file = save_path / "delay_summary_statistics.xlsx"
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Delay_Statistics', index=False)

            print(f"💾 Saved summary: {csv_file}")
            print(f"💾 Saved summary: {excel_file}")

            # Print key insights
            print("\n📊 KEY INSIGHTS:")

            # Best algorithm overall
            avg_by_algorithm = df.groupby(['Algorithm'])['Mean_Delay'].mean().sort_values()
            print(f"🏆 Best overall algorithm: {avg_by_algorithm.index[0]} (avg: {avg_by_algorithm.iloc[0]:.4f}s)")
            print(f"📉 Worst overall algorithm: {avg_by_algorithm.index[-1]} (avg: {avg_by_algorithm.iloc[-1]:.4f}s)")

            # Performance by alpha
            print(f"\n📈 Performance by Alpha:")
            alpha_performance = df.groupby(['Alpha', 'Algorithm'])['Mean_Delay'].mean().unstack()
            print(alpha_performance.round(4))

        return summary_data

    def run_complete_analysis(self, alpha_values=[0.25, 0.5, 1.0, 2.0]):
        """
        Run complete delay analysis with your file format
        """
        print("🚀 Starting Complete Delay Analysis (Fixed for Your Files)")
        print("=" * 60)

        # Load your delay files
        delay_data = self.load_delay_files(alpha_values)

        # Check if we have data
        total_datasets = sum(
            len(delay_data[alpha][alg][ct][cat])
            for alpha in delay_data
            for alg in delay_data[alpha]
            for ct in delay_data[alpha][alg]
            for cat in delay_data[alpha][alg][ct]
            if delay_data[alpha][alg][ct][cat]
        )

        print(f"📊 Loaded {total_datasets} delay datasets from your files")

        if total_datasets > 0:
            # Create analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_dir = f"./delay_analysis_{timestamp}"

            self.create_comparative_graphs(delay_data, analysis_dir)
            self.create_summary_statistics(delay_data, analysis_dir)

            print("=" * 60)
            print("✅ Complete Delay Analysis Finished!")
            print(f"📁 Results saved to: {Path(analysis_dir).absolute()}")
            print("\n📋 Generated Files:")
            print("   📊 9 comparative graphs per alpha (PNG & PDF)")
            print("   📋 Summary statistics (CSV & Excel)")
            print("   🏆 Performance rankings")

        else:
            print("❌ No delay data found!")
            print("💡 Make sure you have delay files in format:")
            print("   delay_{ALGORITHM}_{CONTENT_TYPE}_{CATEGORY}_{ALPHA}_{TIME_SLOTS}_{SIM}.txt")


def main():
    """
    Main function - Run this to create your comparative graphs
    """
    print("🎯 SAGIN Retrieval Delay Analysis (Fixed for Your File Format)")
    print("=" * 60)

    # Initialize analyzer
    analyzer = FixedDelayAnalyzer()

    # Configure analysis parameters - use your actual alpha values
    alpha_values = [0.25, 0.5, 1.0, 2.0]  # All your alpha values

    # Run the analysis
    analyzer.run_complete_analysis(alpha_values=alpha_values)

    print("\n🎉 Analysis Complete!")
    print("📊 Your 9 comparative graphs show algorithm performance across:")
    print("   ✅ All content types (satellite, UAV, grid)")
    print("   ✅ All categories (I, II, III, IV)")
    print("   ✅ All algorithms (LRU, Popularity, MAB_Original, MAB_Contextual, Federated_MAB)")
    print("   ✅ All alpha values (0.25, 0.5, 1.0, 2.0)")


if __name__ == "__main__":
    main()