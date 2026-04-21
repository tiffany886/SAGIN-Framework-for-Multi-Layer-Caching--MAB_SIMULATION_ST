#!/usr/bin/env python3
"""
将 delay_summary_statistics.csv 转为对比柱状图。
用法：
    python plot_delay_csv.py                               # 使用默认的 all_delay_summary.csv
    python plot_delay_csv.py --input your_file.csv         # 指定输入文件
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# ---------------------------- 配置 ----------------------------
# 可自定义颜色（与之前脚本保持一致）
COLORS = {
    'LRU': '#E74C3C',
    'Popularity': '#3498DB',
    'MAB_Original': '#9B59B6',
    'MAB_Contextual': '#2ECC71',
    'MAB_Contextual_EnergyAware': '#1ABC9C',
    'Federated_MAB': '#F39C12',
    'Federated_MAB_EnergyAware': '#E67E22',
    'Enhanced_Federated_MAB': '#FFEAA7',
    'Enhanced_Federated_MAB_EnergyAware': '#D35400',
}
# --------------------------------------------------------------

def plot_category(df, content_type, category, output_prefix):
    """绘制指定内容类型和类别的延迟对比图"""
    subset = df[(df['Content_Type'] == content_type) & (df['Category'] == category)]
    if subset.empty:
        print(f"⚠️ 无数据: {content_type} - {category}")
        return

    # 透视表：行为 Alpha，列为 Algorithm，值为 Mean_Delay
    pivot = subset.pivot_table(index='Alpha', columns='Algorithm', values='Mean_Delay', aggfunc='mean')
    if pivot.empty:
        return

    # 绘图
    plt.figure(figsize=(10, 6))
    pivot.plot(kind='bar', edgecolor='black', color=[COLORS.get(alg, '#888') for alg in pivot.columns])
    plt.title(f'{content_type.upper()} Category {category} - Mean Retrieval Delay', fontweight='bold', fontsize=14)
    plt.xlabel('Zipf α', fontsize=12)
    plt.ylabel('Mean Delay (seconds)', fontsize=12)
    plt.legend(title='Algorithm', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()

    # 保存
    for ext in ['png', 'pdf']:
        outfile = f"{output_prefix}_{content_type}_{category}.{ext}"
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {outfile}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='CSV 延迟数据转图表')
    parser.add_argument('--input', type=str, default='all_delay_summary.csv',
                        help='输入的 CSV 文件路径')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 文件不存在: {args.input}")
        print("   请先运行 merge_delay_summaries.py 生成汇总文件，或指定正确的路径。")
        return

    df = pd.read_csv(args.input)
    print(f"📊 读取数据: {len(df)} 行")

    # 绘制关键对比图
    plot_category(df, 'satellite', 'I', 'fig_delay')
    plot_category(df, 'UAV', 'IV', 'fig_delay')

    print("\n✅ 图表生成完毕。")

if __name__ == "__main__":
    main()