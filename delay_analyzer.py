#!/usr/bin/env python3
"""
Fixed Delay Analyzer - Supports timestamp subdirectories under results/
Usage:
    python delay_analyzer.py --timestamp 20260416_094517
    python delay_analyzer.py                           # scans current directory
"""

import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import argparse
import os
from pathlib import Path
from datetime import datetime


class FixedDelayAnalyzer:
    def __init__(self, base_dir="."):
        """
        base_dir: directory where delay_*.txt files are located
        """
        self.base_dir = base_dir
        self.algorithms = [
            'LRU',
            'Popularity',
            'MAB_Original',
            'MAB_Contextual',
            'MAB_Contextual_EnergyAware',           # 新增
            'Federated_MAB',
            'Federated_MAB_EnergyAware',            # 新增
            'Enhanced_Federated_MAB',
            'Enhanced_Federated_MAB_EnergyAware'    # 新增
        ]
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
            'MAB_Contextual_EnergyAware': '#1ABC9C',      # 新颜色
            'Federated_MAB': '#F39C12',
            'Federated_MAB_EnergyAware': '#E67E22',       # 新颜色
            'Enhanced_Federated_MAB': '#FFEAA7',
            'Enhanced_Federated_MAB_EnergyAware': '#D35400' # 新颜色
        }

    def _alpha_has_samples(self, delay_data, alpha):
        for algorithm in self.algorithms:
            for content_type in self.content_mappings:
                for category in self.content_mappings[content_type]:
                    values = delay_data.get(alpha, {}).get(algorithm, {}).get(content_type, {}).get(category, [])
                    if values:
                        return True
        return False

    def load_delay_files(self, alpha_values=[0.25, 0.5, 1.0, 2.0]):
        """
        Load delays from base_dir using exact file format.
        """
        delay_data = {}
        print(f"🔍 Loading delay files from: {self.base_dir}")

        for alpha in alpha_values:
            delay_data[alpha] = {}
            total_files_for_alpha = 0

            for algorithm in self.algorithms:
                delay_data[alpha][algorithm] = {}

                for content_type in self.content_mappings:
                    delay_data[alpha][algorithm][content_type] = {}

                    for category in self.content_mappings[content_type]:
                        pattern = os.path.join(
                            self.base_dir,
                            f"delay_{algorithm}_{content_type}_{category}_{alpha}_*.txt"
                        )
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

    def create_comparative_graphs(self, delay_data, save_dir):
        """
        Create 9 comparative graphs per alpha, saved in save_dir.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"📊 Creating comparative graphs...")
        skipped = []
        generated = 0

        for alpha in delay_data.keys():
            if not self._alpha_has_samples(delay_data, alpha):
                print(f"⚠️ No valid samples for α={alpha}, skipping...")
                skipped.append({'alpha': alpha, 'reason': 'no_samples_for_alpha'})
                continue

            valid_content_types = []
            for content_type in ['satellite', 'UAV', 'grid']:
                has_data = False
                for algorithm in self.algorithms:
                    for category in self.content_mappings[content_type]:
                        values = delay_data.get(alpha, {}).get(algorithm, {}).get(content_type, {}).get(category, [])
                        if values:
                            has_data = True
                            break
                    if has_data:
                        break
                if has_data:
                    valid_content_types.append(content_type)

            if not valid_content_types:
                skipped.append({'alpha': alpha, 'reason': 'no_content_type_data'})
                continue

            # Dynamic subplot count: only content types with data.
            fig, axes = plt.subplots(1, len(valid_content_types), figsize=(7 * len(valid_content_types), 6))
            if len(valid_content_types) == 1:
                axes = [axes]
            fig.suptitle(f'Retrieval Delay Comparison Across Algorithms (α = {alpha})',
                         fontsize=20, fontweight='bold', y=0.98)

            alpha_plotted = False

            for idx, content_type in enumerate(valid_content_types):
                ax = axes[idx]
                base_categories = self.content_mappings[content_type]

                categories = []
                for cat in base_categories:
                    if any(delay_data.get(alpha, {}).get(alg, {}).get(content_type, {}).get(cat, []) for alg in self.algorithms):
                        categories.append(cat)

                if not categories:
                    skipped.append({'alpha': alpha, 'reason': f'no_categories_with_data:{content_type}'})
                    continue

                # We will create grouped bars for each category within this content_type
                # Instead of 9 subplots (grid-only previously), now 3 subplots each showing multiple categories
                # Let's prepare data: for each algorithm, list of mean delays per category
                algs_present = []
                category_means = {cat: [] for cat in categories}
                category_stds = {cat: [] for cat in categories}

                for algorithm in self.algorithms:
                    if (algorithm in delay_data[alpha] and content_type in delay_data[alpha][algorithm]):
                        alg_means = []
                        alg_stds = []
                        for cat in categories:
                            if cat in delay_data[alpha][algorithm][content_type]:
                                delays = delay_data[alpha][algorithm][content_type][cat]
                                alg_means.append(np.mean(delays))
                                alg_stds.append(np.std(delays))
                            else:
                                alg_means.append(np.nan)
                                alg_stds.append(0.0)
                        if any(m > 0 for m in alg_means):
                            algs_present.append(algorithm)
                            for i, cat in enumerate(categories):
                                category_means[cat].append(alg_means[i])
                                category_stds[cat].append(alg_stds[i])

                if not algs_present:
                    skipped.append({'alpha': alpha, 'reason': f'no_algorithms_with_data:{content_type}'})
                    continue

                alpha_plotted = True

                x = np.arange(len(categories))
                width = 0.8 / len(algs_present)

                for i, alg in enumerate(algs_present):
                    means = [category_means[cat][i] for cat in categories]
                    stds = [category_stds[cat][i] for cat in categories]
                    bars = ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
                                  label=alg, color=self.colors.get(alg, '#888'),
                                  edgecolor='black', linewidth=1)

                ax.set_title(f'{content_type.upper()} Content', fontsize=14, fontweight='bold')
                ax.set_ylabel('Retrieval Delay (seconds)', fontsize=12)
                ax.set_xlabel('Category', fontsize=12)
                ax.set_xticks(x + width * (len(algs_present) - 1) / 2)
                ax.set_xticklabels(categories)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_axisbelow(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            if not alpha_plotted:
                plt.close()
                skipped.append({'alpha': alpha, 'reason': 'alpha_has_no_plottable_series'})
                continue

            png_file = save_path / f"comparative_delay_analysis_alpha_{alpha}.png"
            pdf_file = save_path / f"comparative_delay_analysis_alpha_{alpha}.pdf"

            plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')

            print(f"💾 Saved: {png_file}")
            print(f"💾 Saved: {pdf_file}")

            plt.close()
            generated += 1

        return {
            'generated_count': generated,
            'skipped': skipped,
        }

    def create_summary_statistics(self, delay_data, save_dir):
        """
        Create CSV and Excel summary of delay statistics.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

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

            csv_file = save_path / "delay_summary_statistics.csv"
            df.to_csv(csv_file, index=False)

            excel_file = save_path / "delay_summary_statistics.xlsx"
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Delay_Statistics', index=False)

            print(f"💾 Saved summary: {csv_file}")
            print(f"💾 Saved summary: {excel_file}")

            # Print key insights
            print("\n📊 KEY INSIGHTS:")
            avg_by_algorithm = df.groupby(['Algorithm'])['Mean_Delay'].mean().sort_values()
            if not avg_by_algorithm.empty:
                print(f"🏆 Best overall algorithm: {avg_by_algorithm.index[0]} (avg: {avg_by_algorithm.iloc[0]:.4f}s)")
                print(f"📉 Worst overall algorithm: {avg_by_algorithm.index[-1]} (avg: {avg_by_algorithm.iloc[-1]:.4f}s)")

            print(f"\n📈 Performance by Alpha:")
            alpha_performance = df.groupby(['Alpha', 'Algorithm'])['Mean_Delay'].mean().unstack()
            print(alpha_performance.round(4))

        return summary_data

    def run_complete_analysis(self, alpha_values=[0.25, 0.5, 1.0, 2.0], output_subdir="analysis/delay"):
        """
        Run complete analysis, storing results in base_dir/output_subdir.
        """
        print("🚀 Starting Complete Delay Analysis")
        print("=" * 60)

        delay_data = self.load_delay_files(alpha_values)

        total_datasets = sum(
            len(delay_data[alpha][alg][ct][cat])
            for alpha in delay_data
            for alg in delay_data[alpha]
            for ct in delay_data[alpha][alg]
            for cat in delay_data[alpha][alg][ct]
            if delay_data[alpha][alg][ct][cat]
        )

        print(f"📊 Loaded {total_datasets} delay datasets")

        if total_datasets > 0:
            analysis_dir = Path(self.base_dir) / output_subdir
            plot_result = self.create_comparative_graphs(delay_data, analysis_dir)
            self.create_summary_statistics(delay_data, analysis_dir)

            skipped = plot_result.get('skipped', [])
            if skipped:
                skipped_file = analysis_dir / "skipped_plots.csv"
                pd.DataFrame(skipped).to_csv(skipped_file, index=False)
                print(f"📝 Saved skipped plot report: {skipped_file}")

            print("=" * 60)
            print("✅ Complete Delay Analysis Finished!")
            print(f"📁 Results saved to: {analysis_dir.absolute()}")
            print("\n📋 Generated Files:")
            print(f"   📊 comparative delay graphs (PNG & PDF): {plot_result.get('generated_count', 0)} alpha files")
            print("   📋 summary statistics (CSV & Excel)")
            if skipped:
                print(f"   📝 skipped plot records: {len(skipped)}")
        else:
            print("❌ No delay data found!")
            print("💡 Make sure delay files are present in the specified directory.")


def main():
    parser = argparse.ArgumentParser(description='Analyze delay files from simulation results.')
    parser.add_argument('--timestamp', type=str, help='Timestamp subdirectory under results/ (e.g., 20260416_094517)')
    args = parser.parse_args()

    print("🎯 SAGIN Retrieval Delay Analysis (Timestamp Support)")
    print("=" * 60)

    if args.timestamp:
        base_dir = os.path.join("results", args.timestamp)
        if not os.path.isdir(base_dir):
            print(f"❌ Directory not found: {base_dir}")
            return
    else:
        base_dir = "."

    analyzer = FixedDelayAnalyzer(base_dir=base_dir)
    analyzer.run_complete_analysis(alpha_values=[0.25, 0.5, 1.0, 2.0])

    print("\n🎉 Analysis Complete!")


if __name__ == "__main__":
    main()