#!/usr/bin/env python3
"""
🎯 STATISTICAL WRAPPER - Uses YOUR EXISTING working main.py
============================================================
This imports your existing run_single_simulation() and wraps it
with multiple runs and statistical analysis.

Usage:
    python statistical_wrapper.py --num_runs 10 --batch
    python statistical_wrapper.py --num_runs 3 --algorithm MAB_Contextual --alpha 1.0
"""

import numpy as np
import random
import time
import argparse
import json
import pandas as pd
from datetime import datetime
from scipy import stats
from pathlib import Path
import sys

# Import YOUR existing working functions from main.py
from main import run_single_simulation, create_network, analyze_performance_EXACT


class StatisticalWrapper:
    """
    Wraps your existing working simulation with statistical analysis.
    """
    
    def __init__(self, num_runs=10, confidence_level=0.95):
        self.num_runs = num_runs
        self.confidence_level = confidence_level
        self.results_dir = Path("statistical_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.algorithms = ['LRU', 'Popularity', 'MAB_Original', 'MAB_Contextual', 
                          'Federated_MAB', 'Enhanced_Federated_MAB']
        self.alpha_values = [0.25, 0.5, 1.0, 2.0]
        self.time_slots = 400
        
        print(f"📊 Statistical Wrapper Initialized")
        print(f"   • Runs per config: {num_runs}")
        print(f"   • Using YOUR existing run_single_simulation()")
    
    def set_seed(self, seed):
        """Set random seeds"""
        random.seed(seed)
        np.random.seed(seed)
    
    def run_with_seed(self, algorithm, alpha, seed):
        """Run YOUR existing simulation with a specific seed"""
        self.set_seed(seed)
        
        try:
            # Call YOUR existing working function!
            result = run_single_simulation(algorithm, alpha, self.time_slots)
            
            if result:
                result['seed'] = seed
                return result
            return None
            
        except Exception as e:
            print(f"   ⚠️ Error (seed={seed}): {e}")
            return None
    
    def calculate_statistics(self, results_list, metric):
        """Calculate mean, std, CI for a metric"""
        values = [r[metric] for r in results_list if r and metric in r]
        
        if len(values) < 2:
            val = values[0] if values else 0
            return {
                'mean': val, 'std': 0, 'ci_margin': 0,
                'ci_lower': val, 'ci_upper': val,
                'n_runs': len(values), 'raw_values': values
            }
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        stderr = stats.sem(values)
        t_crit = stats.t.ppf(0.975, len(values) - 1)
        ci_margin = t_crit * stderr
        
        return {
            'mean': mean,
            'std': std,
            'ci_margin': ci_margin,
            'ci_lower': mean - ci_margin,
            'ci_upper': mean + ci_margin,
            'n_runs': len(values),
            'raw_values': values
        }
    
    def run_experiment(self, algorithm, alpha):
        """Run multiple simulations for one config"""
        print(f"\n🔬 {algorithm} | α={alpha} | {self.num_runs} runs")
        print("-" * 50)
        
        # Generate seeds
        base_seed = abs(hash(f"{algorithm}_{alpha}")) % (2**31)
        seeds = [base_seed + i * 7919 for i in range(self.num_runs)]  # Prime spacing
        
        run_results = []
        for i, seed in enumerate(seeds):
            print(f"   Run {i+1}/{self.num_runs} (seed={seed})...", end=" ", flush=True)
            
            result = self.run_with_seed(algorithm, alpha, seed)
            
            if result and result.get('overall_hit_ratio', 0) > 0:
                run_results.append(result)
                print(f"✓ {result['overall_hit_ratio']:.2f}%")
            elif result:
                run_results.append(result)
                print(f"⚠️ {result['overall_hit_ratio']:.2f}%")
            else:
                print("✗ Failed")
        
        # Compile statistics
        metrics = ['vehicle_hit_ratio', 'uav_hit_ratio', 'bs_hit_ratio', 'overall_hit_ratio',
                   'vehicle_cache_ratio', 'uav_cache_ratio', 'bs_cache_ratio', 'overall_cache_ratio']
        
        stats_result = {
            'algorithm': algorithm,
            'alpha': alpha,
            'num_runs': len(run_results)
        }
        
        for metric in metrics:
            stats_result[metric] = self.calculate_statistics(run_results, metric)
        
        # Summary
        if run_results:
            o = stats_result['overall_hit_ratio']
            print(f"   📊 Result: {o['mean']:.2f}% ± {o['ci_margin']:.2f}%")
        
        return stats_result
    
    def run_batch(self):
        """Run all configurations"""
        print(f"\n{'='*60}")
        print("📊 BATCH STATISTICAL ANALYSIS")
        print(f"{'='*60}")
        print(f"Total: {len(self.algorithms)} algorithms × {len(self.alpha_values)} alphas × {self.num_runs} runs")
        print(f"= {len(self.algorithms) * len(self.alpha_values) * self.num_runs} simulations")
        print(f"{'='*60}")
        
        all_results = []
        start = time.time()
        
        for alpha in self.alpha_values:
            for algo in self.algorithms:
                result = self.run_experiment(algo, alpha)
                all_results.append(result)
                
                # Progress
                done = len(all_results)
                total = len(self.algorithms) * len(self.alpha_values)
                elapsed = time.time() - start
                eta = (elapsed / done) * (total - done) if done > 0 else 0
                print(f"   ⏱️ {done}/{total} configs | {elapsed/60:.1f}m elapsed | {eta/60:.1f}m remaining")
        
        self.save_results(all_results)
        return all_results
    
    def save_results(self, results):
        """Save to JSON and CSV"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON
        json_path = self.results_dir / f"stats_{ts}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 JSON: {json_path}")
        
        # CSV
        rows = []
        for r in results:
            row = {'Algorithm': r['algorithm'], 'Alpha': r['alpha'], 'N': r['num_runs']}
            for m in ['overall_hit_ratio', 'vehicle_hit_ratio', 'uav_hit_ratio', 'bs_hit_ratio']:
                s = r[m]
                row[f'{m}_mean'] = f"{s['mean']:.2f}"
                row[f'{m}_std'] = f"{s['std']:.2f}"
                row[f'{m}_ci'] = f"±{s['ci_margin']:.2f}"
            rows.append(row)
        
        csv_path = self.results_dir / f"stats_{ts}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"📊 CSV: {csv_path}")
        
        # Print table
        print(f"\n{'='*90}")
        print("📊 RESULTS SUMMARY")
        print(f"{'='*90}")
        print(f"{'Algorithm':<22} {'α':<6} {'Overall %':<20} {'Vehicle %':<18} {'N':<5}")
        print(f"{'-'*90}")
        for r in results:
            o = r['overall_hit_ratio']
            v = r['vehicle_hit_ratio']
            print(f"{r['algorithm']:<22} {r['alpha']:<6} "
                  f"{o['mean']:>6.2f} ± {o['ci_margin']:<6.2f}   "
                  f"{v['mean']:>6.2f} ± {v['ci_margin']:<6.2f}   "
                  f"{r['num_runs']:<5}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--alpha', type=float)
    args = parser.parse_args()
    
    wrapper = StatisticalWrapper(num_runs=args.num_runs)
    
    if args.batch:
        wrapper.run_batch()
    elif args.algorithm and args.alpha:
        result = wrapper.run_experiment(args.algorithm, args.alpha)
        wrapper.save_results([result])
    else:
        print("Usage:")
        print("  python statistical_wrapper.py --num_runs 10 --batch")
        print("  python statistical_wrapper.py --num_runs 3 --algorithm MAB_Contextual --alpha 1.0")


if __name__ == "__main__":
    main()
