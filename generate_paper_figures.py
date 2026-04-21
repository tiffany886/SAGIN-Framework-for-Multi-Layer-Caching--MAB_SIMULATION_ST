#!/usr/bin/env python3
"""
从 results/ 下的指定时间戳子文件夹（或所有子文件夹）读取 JSON 和 delay 文件，
生成论文级命中率对比图和延迟汇总图，图表保存在对应子目录的 figures/ 下。

用法：
    python generate_paper_figures.py
    python generate_paper_figures.py --timestamp 20260416_094517
"""

import json
import glob
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------- 配置 ----------------------------
ALPHA_VALUES = [0.25, 0.5, 1.0, 2.0]
ALGORITHMS = ['LRU', 'Popularity', 'MAB_Original', 'MAB_Contextual', 'Federated_MAB']
CONTENT_TYPES = ['satellite', 'UAV', 'grid']
CATEGORIES = {
    'satellite': ['I', 'II', 'III'],
    'UAV': ['II', 'III', 'IV'],
    'grid': ['II', 'III', 'IV']
}
COLORS = {
    'LRU': '#E74C3C',
    'Popularity': '#3498DB',
    'MAB_Original': '#9B59B6',
    'MAB_Contextual': '#2ECC71',
    'Federated_MAB': '#F39C12',
}
# --------------------------------------------------------------


def load_hit_data(timestamp=None):
    """加载 JSON 文件，返回 DataFrame"""
    if timestamp:
        json_pattern = f'results/{timestamp}/hit_ratio_analysis_*.json'
    else:
        json_pattern = 'results/*/hit_ratio_analysis_*.json'

    json_files = glob.glob(json_pattern)
    if not json_files:
        print(f"❌ 未找到 JSON 文件，模式: {json_pattern}")
        return None

    records = []
    for fpath in json_files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    record = data[0]
                else:
                    record = data
                records.append(record)
        except Exception as e:
            print(f"⚠️ 跳过损坏文件 {fpath}: {e}")

    df = pd.DataFrame(records)
    return df


def plot_hit_ratios(df, save_dir='.'):
    """绘制命中率柱状图（分组显示 Vehicle/UAV/Overall）"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    tiers = ['vehicle', 'uav', 'overall']
    titles = ['Vehicle Hit Ratio', 'UAV Hit Ratio', 'Overall Hit Ratio']

    for idx, tier in enumerate(tiers):
        ax = axes[idx]
        col = f'{tier}_hit_ratio' if tier != 'overall' else 'overall_hit_ratio'
        if col not in df.columns:
            continue

        pivot = df.pivot_table(index='alpha', columns='algorithm', values=col, aggfunc='mean')
        # 按预设顺序排列算法列
        available_algs = [alg for alg in ALGORITHMS if alg in pivot.columns]
        pivot = pivot.reindex(columns=available_algs)

        pivot.plot(kind='bar', ax=ax,
                   color=[COLORS.get(alg, '#888') for alg in pivot.columns],
                   edgecolor='black')

        ax.set_title(titles[idx], fontsize=12, fontweight='bold')
        ax.set_xlabel('Zipf α', fontsize=10)
        ax.set_ylabel('Hit Ratio (%)', fontsize=10)
        ax.legend(title='Algorithm', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fig_hit_ratio.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'fig_hit_ratio.pdf'), bbox_inches='tight')
    print(f"💾 命中率图已保存: {save_dir}/fig_hit_ratio.png/pdf")
    plt.close(fig)


def load_delay_data(timestamp=None):
    """从 results 下读取 delay_*.txt，返回按 (alpha, algorithm, type, cat) 组织的延迟列表"""
    if timestamp:
        delay_pattern = f'results/{timestamp}/delay_*.txt'
    else:
        delay_pattern = 'results/*/delay_*.txt'

    delay_files = glob.glob(delay_pattern)
    if not delay_files:
        print(f"⚠️ 未找到 delay_*.txt 文件，跳过延迟图绘制。模式: {delay_pattern}")
        return None

    data = defaultdict(list)  # key: (alpha, alg, ctype, cat) -> list of delays
    for fpath in delay_files:
        fname = os.path.basename(fpath)
        # 解析文件名: delay_ALGORITHM_CONTENT_TYPE_CATEGORY_ALPHA_TIMESLOTS_SIM.txt
        parts = fname.replace('.txt', '').split('_')
        if len(parts) < 7:
            continue
        alg = parts[1]
        ctype = parts[2]
        cat = parts[3]
        try:
            alpha = float(parts[4])
        except ValueError:
            continue

        try:
            delays = np.loadtxt(fpath)
            if delays.size > 0:
                data[(alpha, alg, ctype, cat)].extend(delays.flatten())
        except Exception as e:
            print(f"⚠️ 读取 {fpath} 失败: {e}")

    return data


def plot_delay_summary(delay_data, save_dir='.'):
    """绘制延迟汇总柱状图（每个内容类型一张子图，按类别分组，默认展示 α=1.0）"""
    if not delay_data:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, ctype in enumerate(CONTENT_TYPES):
        ax = axes[idx]
        cats = CATEGORIES[ctype]
        algs = ALGORITHMS

        means = {}
        for alg in algs:
            means[alg] = []
            for cat in cats:
                key = (1.0, alg, ctype, cat)  # 可修改为展示其他 α
                delays = delay_data.get(key, [])
                means[alg].append(np.mean(delays) if delays else 0)

        x = np.arange(len(cats))
        width = 0.15
        for i, alg in enumerate(algs):
            ax.bar(x + i*width, means[alg], width, label=alg,
                   color=COLORS.get(alg, '#888'))

        ax.set_title(f'{ctype.upper()} Content Delay (α=1.0)', fontweight='bold')
        ax.set_xlabel('Category')
        ax.set_ylabel('Delay (seconds)')
        ax.set_xticks(x + width * (len(algs)-1)/2)
        ax.set_xticklabels(cats)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fig_delay_summary.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'fig_delay_summary.pdf'), bbox_inches='tight')
    print(f"💾 延迟汇总图已保存: {save_dir}/fig_delay_summary.png/pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='生成论文级命中率和延迟图表')
    parser.add_argument('--timestamp', type=str,
                        help='指定 results 下的子目录名，例如 20260416_094517')
    args = parser.parse_args()

    if args.timestamp:
        output_dir = f'results/{args.timestamp}/figures'
    else:
        output_dir = 'paper_figures'

    os.makedirs(output_dir, exist_ok=True)

    df = load_hit_data(args.timestamp)
    if df is not None:
        plot_hit_ratios(df, output_dir)

    delay_data = load_delay_data(args.timestamp)
    if delay_data:
        plot_delay_summary(delay_data, output_dir)

    print(f"\n✅ 所有图表已保存至: {output_dir}/")


if __name__ == "__main__":
    main()