#!/usr/bin/env python3
"""
Fixed Delay Analyzer (Readable Version)
---------------------------------------
Works with files named:
  delay_{ALGORITHM}_{CONTENT_TYPE}_{CATEGORY}_{ALPHA}_{TIME_SLOTS}_{SIM}.txt

Key improvements:
- Large, consistent fonts everywhere (bar labels, axis labels, ticks, titles)
- Clean, wide subplots with constrained_layout to avoid overlaps
- One figure per content type (Satellite, UAV, Grid) for each alpha
- Big error caps, bar edges; value labels placed smartly (inside or above)
- Summary statistics (CSV + Excel) and console insights

Usage:
    python delay_analyzer.py
"""

import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FixedDelayAnalyzer:
    def __init__(self):
        # Algorithms expected in your file names
        self.algorithms = [
            'LRU',
            'Popularity',
            'MAB_Original',
            'MAB_Contextual',
            'Enhanced_Federated_MAB',
        ]

        # Map of content types and their categories
        self.content_mappings = {
            'satellite': ['I', 'II', 'III'],
            'UAV': ['II', 'III', 'IV'],
            'grid': ['II', 'III', 'IV'],
        }

        # Consistent, high-contrast colors
        self.colors = {
            'LRU': '#FF6B6B',                    # coral red
            'Popularity': '#4ECDC4',             # teal
            'MAB_Original': '#45B7D1',           # steel blue
            'MAB_Contextual': '#96CEB4',         # mint
            'Enhanced_Federated_MAB': '#FFD166', # warm yellow
        }

        # Apply large, readable style
        self.set_mpl_style(base=18)  # bump to 20 if you want even larger text

    # ----------------------------- STYLE ---------------------------------- #
    def set_mpl_style(self, base=16):
        """
        Set large, readable fonts and crisp lines.
        """
        plt.rcParams.update({
            # DPI
            'figure.dpi': 160,
            'savefig.dpi': 400,

            # Fonts
            'font.size': base,               # default text
            'axes.titlesize': base + 4,      # subplot title
            'axes.labelsize': base + 2,      # axes labels
            'xtick.labelsize': base,
            'ytick.labelsize': base,
            'legend.fontsize': base,

            'figure.titlesize': base + 6,    # suptitle
            'axes.titleweight': 'bold',
            'axes.labelweight': 'semibold',

            # Lines/ticks
            'axes.linewidth': 1.3,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,

            # Layout
            'figure.autolayout': False,  # we will use constrained_layout=True on figures
        })

    # ----------------------------- IO ------------------------------------- #
    def load_delay_files(self, alpha_values=[0.25, 0.5, 1.0, 2.0]):
        """
        Load delays from files:
          delay_{ALGORITHM}_{CONTENT_TYPE}_{CATEGORY}_{ALPHA}_{TIME_SLOTS}_{SIM}.txt
        Returns:
          delay_data[alpha][algorithm][content_type][category] -> list of delays
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
                        pattern = f"delay_{algorithm}_{content_type}_{category}_{alpha}_*.txt"
                        files = glob.glob(pattern)

                        all_delays = []
                        for file in files:
                            try:
                                with open(file, 'r') as f:
                                    delays = [float(line.strip()) for line in f if line.strip()]
                                    if delays:
                                        all_delays.extend(delays)
                                        total_files_for_alpha += 1
                            except Exception as e:
                                print(f"⚠️ Error loading {file}: {e}")
                                continue

                        if all_delays:
                            delay_data[alpha][algorithm][content_type][category] = all_delays
                            print(f"  ✅ α={alpha} | {algorithm} - {content_type} - {category}: {len(all_delays)} samples")

            print(f"📊 Found {total_files_for_alpha} file(s) for α={alpha}")

        return delay_data

    # --------------------------- PLOTTING --------------------------------- #
    def _plot_content_type_panel(self, ax, content_type, category, stats):
        """
        Draw one subplot: a bar chart across algorithms for a given content_type & category.
        stats is dict[algorithm] -> {'mean':..., 'std':..., 'median':..., 'count':...}
        """
        if not stats:
            ax.text(0.5, 0.5, 'No Data Available',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=plt.rcParams['font.size'], style='italic', color='red')
            ax.set_title(f'{content_type.upper()} - Category {category}', pad=10)
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.3, axis='y')
            return

        algs = list(stats.keys())
        means = [stats[alg]['mean'] for alg in algs]
        stds  = [stats[alg]['std']  for alg in algs]
        bar_colors = [self.colors.get(alg, '#888888') for alg in algs]

        bars = ax.bar(
            algs, means, yerr=stds, capsize=6,
            color=bar_colors, alpha=0.9, edgecolor='black', linewidth=1.6
        )

        ax.set_title(f'{content_type.upper()} - Category {category}', pad=12)
        ax.set_ylabel('Retrieval Delay (s)')
        ax.set_xlabel('Algorithm')
        ax.tick_params(axis='x', rotation=15)

        # Make x tick labels heavier for readability
        for lbl in ax.get_xticklabels():
            lbl.set_fontweight('semibold')

        # grid and baseline
        ax.grid(True, alpha=0.35, axis='y')
        ax.set_axisbelow(True)

        # Headroom to avoid clipping labels
        if means:
            ymax = max(means)
            ax.set_ylim(bottom=0, top=ymax * 1.40)

        # Value labels: place inside if tall enough, else above
        val_fs = plt.rcParams['font.size']
        ymax = max(means) if means else 1.0
        for bar, mean_val, std_val in zip(bars, means, stds):
            h = bar.get_height()
            label = f'{mean_val:.3f}s'
            if h > (0.30 * ymax):  # inside the bar
                ax.text(
                    bar.get_x() + bar.get_width()/2., h * 0.88, label,
                    ha='center', va='top', fontsize=val_fs, color='black',
                    fontweight='semibold'
                )
            else:  # above the bar
                ax.text(
                    bar.get_x() + bar.get_width()/2., h + (0.05 * ymax), label,
                    ha='center', va='bottom', fontsize=val_fs,
                    fontweight='semibold'
                )

        # Highlight best (lowest mean delay)
        if means:
            best_idx = int(np.argmin(means))
            bars[best_idx].set_edgecolor('#DAA520')  # goldenrod
            bars[best_idx].set_linewidth(3)

    def create_comparative_graphs(self, delay_data, save_dir="./delay_analysis_results"):
        """
        For each alpha:
          - Generate one big figure per content type (satellite, UAV, grid)
          - Each figure has 3 wide subplots (one per category) with large fonts
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"📊 Creating comparative graphs per content type...")

        for alpha in delay_data.keys():
            # Check if any content exists for this alpha
            has_any = False
            for alg in self.algorithms:
                if delay_data[alpha].get(alg, {}):
                    for ct in delay_data[alpha][alg]:
                        if delay_data[alpha][alg][ct]:
                            has_any = True
                            break
                if has_any:
                    break

            if not has_any:
                print(f"⚠️ No data for α={alpha}, skipping plots...")
                continue

            # For each content type, create a 1x3 figure (categories)
            for content_type in self.content_mappings:
                categories = self.content_mappings[content_type]
                ncols = len(categories)

                # Wider figure so labels breathe; tall enough for big fonts
                fig, axes = plt.subplots(
                    1, ncols, figsize=(6.8 * ncols, 6.2),
                    constrained_layout=True, sharey=True
                )
                if ncols == 1:
                    axes = [axes]

                fig.suptitle(
                    f'Retrieval Delay Comparison (α = {alpha}) — {content_type.upper()}',
                    fontweight='bold', y=0.995
                )

                for col, category in enumerate(categories):
                    ax = axes[col]

                    # Build stats for this panel
                    panel_stats = {}
                    for algorithm in self.algorithms:
                        delays = (
                            delay_data
                            .get(alpha, {})
                            .get(algorithm, {})
                            .get(content_type, {})
                            .get(category, [])
                        )
                        if delays:
                            panel_stats[algorithm] = {
                                'mean': float(np.mean(delays)),
                                'std': float(np.std(delays)),
                                'median': float(np.median(delays)),
                                'count': int(len(delays))
                            }

                    self._plot_content_type_panel(ax, content_type, category, panel_stats)

                # Save this figure
                png_file = save_path / f"delay_compare_alpha_{alpha}_{content_type}.png"
                pdf_file = save_path / f"delay_compare_alpha_{alpha}_{content_type}.pdf"
                fig.savefig(png_file, dpi=400, bbox_inches='tight', facecolor='white')
                fig.savefig(pdf_file, bbox_inches='tight', facecolor='white')
                print(f"💾 Saved: {png_file}")
                print(f"💾 Saved: {pdf_file}")
                plt.close(fig)

    # ---------------------- SUMMARY / INSIGHTS ---------------------------- #
    def create_summary_statistics(self, delay_data, save_dir="./delay_analysis_results"):
        """
        Create comprehensive summary statistics table as CSV + Excel.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print("📋 Creating summary statistics table...")

        summary_rows = []
        for alpha in delay_data.keys():
            for algorithm in self.algorithms:
                for content_type in self.content_mappings:
                    for category in self.content_mappings[content_type]:
                        delays = (
                            delay_data
                            .get(alpha, {})
                            .get(algorithm, {})
                            .get(content_type, {})
                            .get(category, [])
                        )
                        if delays:
                            summary_rows.append({
                                'Alpha': alpha,
                                'Algorithm': algorithm,
                                'Content_Type': content_type,
                                'Category': category,
                                'Mean_Delay': float(np.mean(delays)),
                                'Std_Delay': float(np.std(delays)),
                                'Median_Delay': float(np.median(delays)),
                                'Min_Delay': float(np.min(delays)),
                                'Max_Delay': float(np.max(delays)),
                                'P95_Delay': float(np.percentile(delays, 95)),
                                'Sample_Count': int(len(delays)),
                            })

        if not summary_rows:
            print("❌ No summary to write (no delays found).")
            return []

        df = pd.DataFrame(summary_rows)

        # Save as CSV
        csv_file = save_path / "delay_summary_statistics.csv"
        df.to_csv(csv_file, index=False)

        # Save as Excel
        excel_file = save_path / "delay_summary_statistics.xlsx"
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Delay_Statistics', index=False)
        except Exception as e:
            print(f"⚠️ Could not write Excel with openpyxl: {e}")
            print("   Saving only CSV. Install openpyxl to enable Excel export.")

        print(f"💾 Saved summary: {csv_file}")
        if excel_file.exists():
            print(f"💾 Saved summary: {excel_file}")

        # Console insights
        print("\n📊 KEY INSIGHTS:")
        avg_by_algorithm = df.groupby(['Algorithm'])['Mean_Delay'].mean().sort_values()
        print(f"🏆 Best overall algorithm (lowest mean delay): {avg_by_algorithm.index[0]} "
              f"(avg: {avg_by_algorithm.iloc[0]:.4f}s)")
        print(f"📉 Worst overall algorithm: {avg_by_algorithm.index[-1]} "
              f"(avg: {avg_by_algorithm.iloc[-1]:.4f}s)")

        print("\n📈 Performance by Alpha (mean delay):")
        alpha_perf = df.groupby(['Alpha', 'Algorithm'])['Mean_Delay'].mean().unstack()
        print(alpha_perf.round(4))

        return summary_rows

    # ------------------------------ RUN ---------------------------------- #
    def run_complete_analysis(self, alpha_values=[0.25, 0.5, 1.0, 2.0]):
        """
        Full pipeline:
          1) Load delay files
          2) Plot comparative figures (per content type)
          3) Generate summary tables
        """
        print("🚀 Starting Complete Delay Analysis (Readable Version)")
        print("=" * 70)

        delay_data = self.load_delay_files(alpha_values)

        # Count datasets
        total_sets = 0
        for alpha in delay_data:
            for alg in delay_data[alpha]:
                for ct in delay_data[alpha][alg]:
                    for cat in delay_data[alpha][alg][ct]:
                        total_sets += len(delay_data[alpha][alg][ct][cat])

        print(f"📊 Loaded {total_sets} delay samples across all files")

        if total_sets == 0:
            print("❌ No delay data found!")
            print("💡 Ensure files exist as:")
            print("   delay_{ALGORITHM}_{CONTENT_TYPE}_{CATEGORY}_{ALPHA}_{TIME_SLOTS}_{SIM}.txt")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"./delay_analysis_{timestamp}"

        self.create_comparative_graphs(delay_data, out_dir)
        self.create_summary_statistics(delay_data, out_dir)

        print("=" * 70)
        print("✅ Complete Delay Analysis Finished!")
        print(f"📁 Results saved to: {Path(out_dir).absolute()}")
        print("\n📋 Generated:")
        print("   📊 Comparative figures per α and content type (PNG & PDF)")
        print("   📋 Summary statistics (CSV & Excel)")
        print("   🏆 Console insights (best/worst, by-α table)")

# ------------------------------ CLI -------------------------------------- #
def main():
    """
    Main entry point.
    """
    print("🎯 SAGIN Retrieval Delay Analysis (Readable Version)")
    print("=" * 70)

    analyzer = FixedDelayAnalyzer()
    alpha_values = [0.25, 0.5, 1.0, 2.0]  # adjust if needed
    analyzer.run_complete_analysis(alpha_values=alpha_values)

    print("\n🎉 Analysis Complete!")

if __name__ == "__main__":
    main()
