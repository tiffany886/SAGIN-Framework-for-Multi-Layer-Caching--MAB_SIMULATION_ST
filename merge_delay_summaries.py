#!/usr/bin/env python3
"""
合并所有 results/*/delay_analysis/delay_summary_statistics.csv，
生成全局延迟对比总表和核心指标图表。
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt
import os

def merge_all_summaries():
    csv_files = glob.glob('results/*/delay_analysis/delay_summary_statistics.csv')
    if not csv_files:
        print("❌ 未找到任何 delay_summary_statistics.csv 文件。")
        return None

    df_list = []
    for f in csv_files:
        df = pd.read_csv(f)
        df_list.append(df)

    merged = pd.concat(df_list, ignore_index=True)
    merged.to_csv('all_delay_summary.csv', index=False)
    print(f"✅ 已合并 {len(csv_files)} 个文件 → all_delay_summary.csv")
    return merged

def plot_key_delays(df):
    """绘制核心对比图：卫星 I 类（大文件）延迟按算法和 α 分组"""
    sat_i = df[(df['Content_Type'] == 'satellite') & (df['Category'] == 'I')]
    if sat_i.empty:
        print("⚠️ 无卫星 I 类数据")
        return

    pivot = sat_i.pivot_table(index='Alpha', columns='Algorithm', values='Mean_Delay', aggfunc='mean')
    
    plt.figure(figsize=(10, 6))
    pivot.plot(kind='bar', edgecolor='black')
    plt.title('Satellite Category I (HDF 10MB) Mean Retrieval Delay', fontweight='bold')
    plt.xlabel('Zipf α')
    plt.ylabel('Mean Delay (seconds)')
    plt.legend(title='Algorithm')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_satellite_I_delay_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig_satellite_I_delay_comparison.pdf', bbox_inches='tight')
    print("💾 卫星 I 类延迟对比图已保存。")
    plt.show()

if __name__ == "__main__":
    df = merge_all_summaries()
    if df is not None:
        plot_key_delays(df)