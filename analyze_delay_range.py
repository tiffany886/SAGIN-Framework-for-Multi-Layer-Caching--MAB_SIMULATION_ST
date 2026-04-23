#!/usr/bin/env python3
"""
按结果目录时间范围汇总 delay_summary_statistics.csv，并输出统计分析与可视化。

示例:
python analyze_delay_range.py \
  --start 20260416_232910 \
  --end 20260418_004408
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


REQUIRED_COLUMNS = [
    "Alpha",
    "Algorithm",
    "Content_Type",
    "Category",
    "Mean_Delay",
    "Std_Delay",
    "Median_Delay",
    "Min_Delay",
    "Max_Delay",
    "P95_Delay",
    "Sample_Count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按时间范围汇总 delay_summary_statistics 并分析")
    parser.add_argument("--start", required=True, help="起始目录时间戳，如 20260416_232910")
    parser.add_argument("--end", required=True, help="结束目录时间戳，如 20260418_004408")
    parser.add_argument("--results-dir", default="results", help="结果目录，默认 results")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录，默认 delay_analysis_range_<start>_to_<end>",
    )
    return parser.parse_args()


def pick_run_dirs(results_dir: Path, start: str, end: str) -> List[Path]:
    candidates = [p for p in results_dir.iterdir() if p.is_dir()]
    picked = [p for p in sorted(candidates) if start <= p.name <= end]
    return picked


def read_and_merge(run_dirs: List[Path]) -> Tuple[pd.DataFrame, List[str]]:
    frames = []
    missing = []

    for run_dir in run_dirs:
        csv_path = run_dir / "delay_analysis" / "delay_summary_statistics.csv"
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue
        df = pd.read_csv(csv_path)
        lack_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if lack_cols:
            raise ValueError(f"{csv_path} 缺少字段: {lack_cols}")
        df["Run_ID"] = run_dir.name
        frames.append(df)

    if not frames:
        raise ValueError("未读到任何有效 CSV 数据")

    merged = pd.concat(frames, ignore_index=True)
    return merged, missing


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    wsum = weights.sum()
    if wsum <= 0:
        return float(series.mean())
    return float(np.average(series, weights=weights))


def holm_adjust(pvalues: List[float]) -> List[float]:
    """Holm-Bonferroni 校正，保持输入顺序返回调整后的 p 值。"""
    m = len(pvalues)
    if m == 0:
        return []

    ranked = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for i, (idx, pvalue) in enumerate(ranked):
        value = min((m - i) * pvalue, 1.0)
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted


def build_stats_tables(df: pd.DataFrame, out_dir: Path) -> dict:
    out = {}
    data = df.copy()
    data["Sample_Count"] = pd.to_numeric(data["Sample_Count"], errors="coerce").fillna(0)

    algo_rows = []
    for algo, g in data.groupby("Algorithm"):
        algo_rows.append(
            {
                "Algorithm": algo,
                "Runs": g["Run_ID"].nunique(),
                "Rows": len(g),
                "Total_Samples": int(g["Sample_Count"].sum()),
                "Weighted_Mean_Delay": weighted_mean(g["Mean_Delay"], g["Sample_Count"]),
                "Weighted_P95_Delay": weighted_mean(g["P95_Delay"], g["Sample_Count"]),
                "Mean_of_Mean_Delay": float(g["Mean_Delay"].mean()),
                "Std_of_Mean_Delay": float(g["Mean_Delay"].std(ddof=0)),
            }
        )
    algo_df = pd.DataFrame(algo_rows).sort_values("Weighted_Mean_Delay")
    algo_path = out_dir / "stats_by_algorithm.csv"
    algo_df.to_csv(algo_path, index=False)
    out["algo"] = algo_df

    alpha_rows = []
    for (alpha, algo), g in data.groupby(["Alpha", "Algorithm"]):
        alpha_rows.append(
            {
                "Alpha": alpha,
                "Algorithm": algo,
                "Rows": len(g),
                "Total_Samples": int(g["Sample_Count"].sum()),
                "Weighted_Mean_Delay": weighted_mean(g["Mean_Delay"], g["Sample_Count"]),
                "Weighted_P95_Delay": weighted_mean(g["P95_Delay"], g["Sample_Count"]),
            }
        )
    alpha_df = pd.DataFrame(alpha_rows).sort_values(["Alpha", "Weighted_Mean_Delay"])
    alpha_path = out_dir / "stats_by_alpha_algorithm.csv"
    alpha_df.to_csv(alpha_path, index=False)
    out["alpha"] = alpha_df

    cc_rows = []
    for (ctype, category, algo), g in data.groupby(["Content_Type", "Category", "Algorithm"]):
        cc_rows.append(
            {
                "Content_Type": ctype,
                "Category": category,
                "Algorithm": algo,
                "Rows": len(g),
                "Total_Samples": int(g["Sample_Count"].sum()),
                "Weighted_Mean_Delay": weighted_mean(g["Mean_Delay"], g["Sample_Count"]),
                "Weighted_P95_Delay": weighted_mean(g["P95_Delay"], g["Sample_Count"]),
            }
        )
    cc_df = pd.DataFrame(cc_rows).sort_values(["Content_Type", "Category", "Weighted_Mean_Delay"])
    cc_path = out_dir / "stats_by_content_category_algorithm.csv"
    cc_df.to_csv(cc_path, index=False)
    out["cc"] = cc_df

    return out


def build_significance_tables(df: pd.DataFrame, tables: dict, out_dir: Path) -> dict:
    """基于场景级配对数据进行显著性检验。"""
    out: dict = {}
    algo_order = tables["algo"]["Algorithm"].tolist()
    scenario_cols = ["Alpha", "Content_Type", "Category"]
    matrix = (
        df.pivot_table(
            index=scenario_cols,
            columns="Algorithm",
            values="Mean_Delay",
            aggfunc="mean",
        )
        .reindex(columns=algo_order)
        .sort_index()
    )

    if matrix.isna().any().any():
        raise ValueError("显著性检验所需的场景矩阵存在缺失值")

    friedman_stat, friedman_p = friedmanchisquare(*[matrix[a].values for a in algo_order])
    friedman_df = pd.DataFrame(
        [
            {
                "Test": "Friedman",
                "Algorithms": len(algo_order),
                "Scenarios": len(matrix),
                "Statistic": float(friedman_stat),
                "P_Value": float(friedman_p),
            }
        ]
    )
    friedman_df.to_csv(out_dir / "significance_friedman_test.csv", index=False)
    out["friedman"] = friedman_df

    ranks = matrix.apply(lambda row: pd.Series(rankdata(row.values, method="average"), index=algo_order), axis=1)
    rank_summary = pd.DataFrame(
        {
            "Algorithm": algo_order,
            "Average_Rank": ranks.mean(axis=0).values,
            "Median_Delay": matrix.median(axis=0).values,
            "Mean_Delay": matrix.mean(axis=0).values,
        }
    ).sort_values("Average_Rank")
    rank_summary.to_csv(out_dir / "stats_by_algorithm_rank.csv", index=False)
    out["rank_summary"] = rank_summary

    pairwise_rows = []
    pvalues = []
    pair_index = []
    for i, left in enumerate(algo_order):
        for right in algo_order[i + 1 :]:
            left_values = matrix[left].values
            right_values = matrix[right].values
            try:
                stat, pvalue = wilcoxon(left_values, right_values, zero_method="wilcox", alternative="two-sided", mode="auto")
            except ValueError:
                stat, pvalue = 0.0, 1.0

            diff = left_values - right_values
            pairwise_rows.append(
                {
                    "Algorithm_1": left,
                    "Algorithm_2": right,
                    "Scenarios": len(diff),
                    "Mean_Diff(A1-A2)": float(diff.mean()),
                    "Median_Diff(A1-A2)": float(np.median(diff)),
                    "Wins_A1": int((diff < 0).sum()),
                    "Wins_A2": int((diff > 0).sum()),
                    "Wilcoxon_Statistic": float(stat),
                    "P_Value": float(pvalue),
                }
            )
            pvalues.append(float(pvalue))
            pair_index.append((left, right))

    adjusted = holm_adjust(pvalues)
    for row, adj in zip(pairwise_rows, adjusted):
        row["Holm_P_Value"] = float(adj)
        row["Significant_0.05"] = bool(adj < 0.05)

    pairwise_df = pd.DataFrame(pairwise_rows).sort_values(["Holm_P_Value", "P_Value"])
    pairwise_df.to_csv(out_dir / "significance_pairwise_wilcoxon.csv", index=False)
    out["pairwise"] = pairwise_df

    significant = pairwise_df[pairwise_df["Significant_0.05"]].copy()
    significant.to_csv(out_dir / "significance_pairwise_wilcoxon_significant.csv", index=False)
    out["significant"] = significant

    # 生成矩阵，便于作图
    p_matrix = pd.DataFrame(np.ones((len(algo_order), len(algo_order))), index=algo_order, columns=algo_order)
    for row in pairwise_df.itertuples(index=False):
        p_matrix.loc[row.Algorithm_1, row.Algorithm_2] = row.Holm_P_Value
        p_matrix.loc[row.Algorithm_2, row.Algorithm_1] = row.Holm_P_Value
    out["p_matrix"] = p_matrix

    return out


def plot_figures(df: pd.DataFrame, tables: dict, sig_tables: dict, out_dir: Path) -> None:
    plt.style.use("ggplot")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    algo_order = tables["algo"]["Algorithm"].tolist()
    plot_data = [df.loc[df["Algorithm"] == algo, "Mean_Delay"].values for algo in algo_order]
    ax1.boxplot(plot_data, tick_labels=algo_order, patch_artist=True)
    ax1.set_title("Mean Delay Distribution by Algorithm")
    ax1.set_xlabel("Algorithm")
    ax1.set_ylabel("Mean Delay")
    fig1.tight_layout()
    fig1.savefig(out_dir / "fig1_boxplot_mean_delay_by_algorithm.png", dpi=200)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    alpha_df = tables["alpha"]
    for algo in sorted(alpha_df["Algorithm"].unique()):
        g = alpha_df[alpha_df["Algorithm"] == algo].sort_values("Alpha")
        ax2.plot(g["Alpha"], g["Weighted_Mean_Delay"], marker="o", label=algo)
    ax2.set_title("Weighted Mean Delay vs Alpha")
    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("Weighted Mean Delay")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / "fig2_weighted_mean_delay_vs_alpha.png", dpi=200)
    plt.close(fig2)

    cc_df = tables["cc"]
    cc_df = cc_df.copy()
    cc_df["CC"] = cc_df["Content_Type"].astype(str) + "_" + cc_df["Category"].astype(str)
    pivot = cc_df.pivot_table(index="CC", columns="Algorithm", values="Weighted_Mean_Delay", aggfunc="mean")

    fig3, ax3 = plt.subplots(figsize=(10, 8))
    img = ax3.imshow(pivot.values, aspect="auto")
    ax3.set_xticks(range(len(pivot.columns)))
    ax3.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax3.set_yticks(range(len(pivot.index)))
    ax3.set_yticklabels(pivot.index)
    ax3.set_title("Heatmap of Weighted Mean Delay (Content_Category x Algorithm)")
    cbar = fig3.colorbar(img, ax=ax3)
    cbar.set_label("Weighted Mean Delay")
    fig3.tight_layout()
    fig3.savefig(out_dir / "fig3_heatmap_content_category_algorithm.png", dpi=200)
    plt.close(fig3)

    # 显著性图 1：平均秩条形图
    rank_df = sig_tables["rank_summary"].sort_values("Average_Rank")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.barh(rank_df["Algorithm"], rank_df["Average_Rank"], color="#4C78A8")
    ax4.invert_yaxis()
    ax4.set_title("Average Rank by Algorithm (lower is better)")
    ax4.set_xlabel("Average Rank")
    ax4.set_ylabel("Algorithm")
    for y, value in enumerate(rank_df["Average_Rank"]):
        ax4.text(value + 0.03, y, f"{value:.2f}", va="center", fontsize=9)
    fig4.tight_layout()
    fig4.savefig(out_dir / "fig4_average_rank_by_algorithm.png", dpi=200)
    plt.close(fig4)

    # 显著性图 2：Holm 校正后的 pairwise p-value 热力图
    p_matrix = sig_tables["p_matrix"]
    algo_order = p_matrix.index.tolist()
    display = -np.log10(p_matrix.replace(0, np.nextafter(0, 1)).values)
    np.fill_diagonal(display, np.nan)
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    im = ax5.imshow(display, cmap="viridis", aspect="auto")
    ax5.set_xticks(range(len(algo_order)))
    ax5.set_xticklabels(algo_order, rotation=30, ha="right")
    ax5.set_yticks(range(len(algo_order)))
    ax5.set_yticklabels(algo_order)
    ax5.set_title("Pairwise Significance Heatmap (-log10 Holm-adjusted p)")
    cbar = fig5.colorbar(im, ax=ax5)
    cbar.set_label("-log10(adjusted p)")
    fig5.tight_layout()
    fig5.savefig(out_dir / "fig5_pairwise_significance_heatmap.png", dpi=200)
    plt.close(fig5)


def write_report(df: pd.DataFrame, tables: dict, sig_tables: dict, out_dir: Path, start: str, end: str, missing: List[str]) -> None:
    algo_df = tables["algo"]
    best = algo_df.iloc[0]
    worst = algo_df.iloc[-1]

    alpha_best = (
        tables["alpha"]
        .sort_values(["Alpha", "Weighted_Mean_Delay"])
        .groupby("Alpha", as_index=False)
        .first()
    )

    lines = []
    lines.append(f"# Delay 汇总分析报告 ({start} ~ {end})")
    lines.append("")
    lines.append("## 数据覆盖")
    lines.append(f"- 汇总行数: {len(df)}")
    lines.append(f"- 覆盖运行批次: {df['Run_ID'].nunique()}")
    lines.append(f"- 算法数量: {df['Algorithm'].nunique()}")
    lines.append(f"- Alpha 数量: {df['Alpha'].nunique()}")
    lines.append(f"- 内容类型数量: {df['Content_Type'].nunique()}")
    lines.append(f"- 分类数量: {df['Category'].nunique()}")
    lines.append("")

    lines.append("## 关键结论")
    lines.append(
        "- 最优算法(加权平均 Mean_Delay 最低): "
        f"{best['Algorithm']} ({best['Weighted_Mean_Delay']:.6f})"
    )
    lines.append(
        "- 最慢算法(加权平均 Mean_Delay 最高): "
        f"{worst['Algorithm']} ({worst['Weighted_Mean_Delay']:.6f})"
    )
    gap_pct = ((worst["Weighted_Mean_Delay"] - best["Weighted_Mean_Delay"]) / worst["Weighted_Mean_Delay"] * 100)
    lines.append(f"- 最优相对最慢的延迟降幅: {gap_pct:.2f}%")
    lines.append("")

    friedman = sig_tables["friedman"].iloc[0]
    significant = sig_tables["significant"]
    lines.append("## 显著性检验")
    lines.append(
        f"- Friedman 检验: statistic={friedman['Statistic']:.4f}, p={friedman['P_Value']:.6g}, "
        f"scenarios={int(friedman['Scenarios'])}"
    )
    lines.append(f"- Holm 校正后显著的两两比较数: {len(significant)}")
    if not significant.empty:
        lines.append("- 最显著的若干算法对:")
        for _, row in significant.head(8).iterrows():
            lines.append(
                f"  - {row['Algorithm_1']} vs {row['Algorithm_2']}: "
                f"holm_p={row['Holm_P_Value']:.6g}, mean_diff={row['Mean_Diff(A1-A2)']:.6f}"
            )
    lines.append("")

    lines.append("## 各 Alpha 最优算法")
    for _, row in alpha_best.iterrows():
        lines.append(
            f"- Alpha={row['Alpha']}: {row['Algorithm']} "
            f"(Weighted_Mean_Delay={row['Weighted_Mean_Delay']:.6f})"
        )
    lines.append("")

    if missing:
        lines.append("## 缺失文件")
        for p in missing:
            lines.append(f"- {p}")
        lines.append("")

    lines.append("## 产物清单")
    lines.append("- merged_delay_summary.csv")
    lines.append("- stats_by_algorithm.csv")
    lines.append("- stats_by_alpha_algorithm.csv")
    lines.append("- stats_by_content_category_algorithm.csv")
    lines.append("- fig1_boxplot_mean_delay_by_algorithm.png")
    lines.append("- fig2_weighted_mean_delay_vs_alpha.png")
    lines.append("- fig3_heatmap_content_category_algorithm.png")
    lines.append("- fig4_average_rank_by_algorithm.png")
    lines.append("- fig5_pairwise_significance_heatmap.png")
    lines.append("- significance_friedman_test.csv")
    lines.append("- significance_pairwise_wilcoxon.csv")
    lines.append("- significance_pairwise_wilcoxon_significant.csv")
    lines.append("- stats_by_algorithm_rank.csv")

    (out_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir or f"delay_analysis_range_{args.start}_to_{args.end}")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = pick_run_dirs(results_dir, args.start, args.end)
    if not run_dirs:
        raise ValueError("指定时间范围内没有找到任何结果目录")

    merged, missing = read_and_merge(run_dirs)
    merged.to_csv(out_dir / "merged_delay_summary.csv", index=False)

    tables = build_stats_tables(merged, out_dir)
    sig_tables = build_significance_tables(merged, tables, out_dir)
    plot_figures(merged, tables, sig_tables, out_dir)
    write_report(merged, tables, sig_tables, out_dir, args.start, args.end, missing)

    print("分析完成，输出目录:", out_dir)


if __name__ == "__main__":
    main()
