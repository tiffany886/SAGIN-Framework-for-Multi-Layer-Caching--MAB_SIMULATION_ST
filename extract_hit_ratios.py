#!/usr/bin/env python3
"""
从 results/ 下的指定时间戳子文件夹（或所有子文件夹）提取 hit_ratio_analysis_*.json，
按 α 值分组计算平均命中率，并输出 CSV 文件和终端表格。

用法：
    python extract_hit_ratios.py                         # 扫描所有 results/*/ 子目录
    python extract_hit_ratios.py --timestamp 20260416_094517  # 仅分析指定时间戳文件夹
"""

import json
import glob
import os
import argparse
from collections import defaultdict
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='提取命中率并按 α 汇总')
    parser.add_argument('--timestamp', type=str,
                        help='指定 results 下的子目录名，例如 20260416_094517')
    args = parser.parse_args()

    if args.timestamp:
        json_pattern = f'results/{args.timestamp}/hit_ratio_analysis_*.json'
        output_dir = f'results/{args.timestamp}'
    else:
        json_pattern = 'results/*/hit_ratio_analysis_*.json'
        output_dir = '.'

    json_files = glob.glob(json_pattern)
    if not json_files:
        print(f"❌ 未找到匹配的 JSON 文件，模式: {json_pattern}")
        return

    print(f"🔍 找到 {len(json_files)} 个 JSON 文件")

    alpha_data = defaultdict(list)

    for fpath in json_files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                # JSON 文件通常是一个包含单个元素的列表
                if isinstance(data, list) and len(data) > 0:
                    record = data[0]
                else:
                    record = data

                alpha = record.get('alpha')
                if alpha is None:
                    continue

                alpha_data[alpha].append({
                    'algorithm': record.get('algorithm', 'unknown'),
                    'vehicle_hit': record.get('vehicle_hit_ratio', 0.0),
                    'uav_hit': record.get('uav_hit_ratio', 0.0),
                    'bs_hit': record.get('bs_hit_ratio', 0.0),
                    'overall_hit': record.get('overall_hit_ratio', 0.0),
                    'vehicle_cache': record.get('vehicle_cache_ratio', 0.0),
                    'uav_cache': record.get('uav_cache_ratio', 0.0),
                    'bs_cache': record.get('bs_cache_ratio', 0.0),
                    'overall_cache': record.get('overall_cache_ratio', 0.0),
                })
        except Exception as e:
            print(f"⚠️ 读取文件失败 {fpath}: {e}")

    if not alpha_data:
        print("❌ 没有提取到有效数据。")
        return

    # 计算每个 α 的平均值
    summary_rows = []
    for alpha in sorted(alpha_data.keys()):
        items = alpha_data[alpha]
        n = len(items)
        avg = {
            'alpha': alpha,
            'count': n,
            'vehicle_hit_avg': sum(x['vehicle_hit'] for x in items) / n,
            'uav_hit_avg': sum(x['uav_hit'] for x in items) / n,
            'bs_hit_avg': sum(x['bs_hit'] for x in items) / n,
            'overall_hit_avg': sum(x['overall_hit'] for x in items) / n,
            'vehicle_cache_avg': sum(x['vehicle_cache'] for x in items) / n,
            'uav_cache_avg': sum(x['uav_cache'] for x in items) / n,
            'bs_cache_avg': sum(x['bs_cache'] for x in items) / n,
            'overall_cache_avg': sum(x['overall_cache'] for x in items) / n,
        }
        summary_rows.append(avg)

    df = pd.DataFrame(summary_rows)

    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, 'alpha_hit_ratio_summary.csv')
    df.to_csv(csv_file, index=False, float_format='%.4f')
    print(f"💾 已保存汇总 CSV: {csv_file}")

    print("\n📊 各 α 值下的平均命中率（单位：%）：")
    print("-" * 70)
    print(f"{'α':<6} {'样本数':<6} {'车辆命中':<10} {'UAV命中':<10} {'BS命中':<10} {'总体命中':<10}")
    for row in summary_rows:
        print(f"{row['alpha']:<6} {row['count']:<6} "
              f"{row['vehicle_hit_avg']:<10.2f} {row['uav_hit_avg']:<10.2f} "
              f"{row['bs_hit_avg']:<10.2f} {row['overall_hit_avg']:<10.2f}")
    print("-" * 70)


if __name__ == "__main__":
    main()