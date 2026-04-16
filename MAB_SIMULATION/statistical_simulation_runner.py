#!/usr/bin/env python3
"""
🎯 STATISTICAL SIMULATION RUNNER (FIXED)
=========================================
Wraps your EXISTING working simulation with multiple runs for statistical analysis.
Uses your original run_single_simulation() function directly.

Usage:
    python statistical_simulation_runner_fixed.py --num_runs 10 --batch
    python statistical_simulation_runner_fixed.py --num_runs 5 --algorithm MAB_Contextual --alpha 1.0
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
import warnings

warnings.filterwarnings('ignore')

# Import your EXISTING modules
from communication import Communication
from federated_mab import FederatedAggregator
from enhanced_federated_mab import EnhancedFederatedAggregator
from vehicle_ccn import Vehicle
from uav_ccn import UAV
from satellite_ccn import Satellite
from bs_ccn import BaseStation
from gs_ccn import GroundStation


class StatisticalSimulationRunner:
    """
    Runs multiple simulations with different random seeds using YOUR ORIGINAL code.
    """

    def __init__(self, num_runs=10, confidence_level=0.95):
        self.num_runs = num_runs
        self.confidence_level = confidence_level
        self.results_dir = Path("statistical_results")
        self.results_dir.mkdir(exist_ok=True)

        # Configuration matching YOUR main.py
        self.algorithms = ['LRU', 'Popularity', 'MAB_Original', 'MAB_Contextual',
                           'Federated_MAB', 'Enhanced_Federated_MAB']
        self.alpha_values = [0.25, 0.5, 1.0, 2.0]
        self.time_slots = 400

        print(f"📊 Statistical Simulation Runner Initialized")
        print(f"   • Number of runs per config: {num_runs}")
        print(f"   • Confidence level: {confidence_level * 100}%")

    def set_random_seed(self, seed):
        """Set all random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)

    def create_network(self, algorithm_type):
        """Create network - EXACT COPY from your main.py"""
        print(f"🏗️ Creating network for {algorithm_type}")

        if algorithm_type == "Enhanced_Federated_MAB":
            aggregator = EnhancedFederatedAggregator()
        else:
            aggregator = FederatedAggregator()

        grid_size = 10
        uav_grid_size = 20

        # Create satellites
        satellites = {}
        for i in range(1, 4):
            satellite = Satellite(f"Satellite{i}")
            satellites[f"Satellite{i}"] = satellite

        # Create ground station
        ground_station = GroundStation("GS1")

        # Create UAVs
        uavs = []
        for i in range(1, 26):
            uav = UAV(f"UAV{i}", grid_size, uav_grid_size, aggregator, algorithm=algorithm_type)
            uavs.append(uav)

        # Create vehicles
        vehicles = []
        for i in range(1, 51):
            vehicle = Vehicle(f"Vehicle{i}", grid_size, 5, 1, aggregator, algorithm=algorithm_type)
            vehicles.append(vehicle)

        # Create base stations
        base_stations = []
        for i in range(1, 4):
            bs = BaseStation(f"BS{i}", 15, grid_size, aggregator, algorithm=algorithm_type)
            base_stations.append(bs)

        # Setup UAV neighbors
        for i, uav in enumerate(uavs):
            uav.neighbors = []
            if i > 0:
                uav.neighbors.append(uavs[i - 1])
            if i < len(uavs) - 1:
                uav.neighbors.append(uavs[i + 1])

        return satellites, uavs, vehicles, base_stations, ground_station, aggregator

    def run_single_simulation(self, algorithm, alpha, seed):
        """
        Run simulation - USING YOUR EXACT SIMULATION LOOP from main.py
        """
        # Set random seed
        self.set_random_seed(seed)

        time_slots = self.time_slots
        simulation_round = 1

        try:
            # Create network - YOUR CODE
            satellites, uavs, vehicles, base_stations, ground_station, aggregator = \
                self.create_network(algorithm)

            # Create communication - YOUR CODE
            communication = Communication(
                satellites, base_stations, vehicles, uavs,
                ground_station, alpha, simulation_round, time_slots
            )
            communication.current_algorithm = algorithm

            # YOUR EXACT PARAMETERS from main.py
            grid_size = 10
            no_of_content_each_category = 10
            no_of_request_genertaed_in_each_timeslot = 3
            uav_content_generation_period = 5
            consecutive_slots_g = 5
            epsilon = 0.1

            # Communication schedule - YOUR CODE
            communication_schedule = {slot: [1, 2, 3] for slot in range(1, time_slots + 1)}

            # ============================================================
            # YOUR EXACT SIMULATION LOOP FROM main.py - UNCHANGED
            # ============================================================
            for slot in range(1, time_slots + 1):
                current_time = (slot - 1) * 60

                # Satellite operations
                if (slot - 1) % consecutive_slots_g == 0:
                    for satellite_id in satellites:
                        satellites[satellite_id].run(
                            satellites, communication, communication_schedule,
                            current_time, slot, no_of_content_each_category, ground_station
                        )
                    ground_station.run(current_time, satellites)

                # UAV operations
                for uav in uavs:
                    uav.run(
                        current_time, communication, communication_schedule, slot, satellites,
                        no_of_content_each_category, uav_content_generation_period, epsilon, uavs
                    )

                # Vehicle operations
                # NOTE: Adjust arguments to match YOUR Vehicle.run() signature
                for vehicle in vehicles:
                    vehicle.run(current_time, slot, time_slots, vehicles, uavs, base_stations, satellites,
                                communication, no_of_request_genertaed_in_each_timeslot,
                                no_of_content_each_category)

                # Base station operations
                for bs in base_stations:
                    bs.run(current_time, slot)

                # Federated learning aggregation
                if algorithm in ["Federated_MAB", "Enhanced_Federated_MAB"] and slot > 10 and ((slot - 1) % 10 == 0):
                    aggregator.aggregate_updates()

            # ============================================================
            # ANALYZE PERFORMANCE - YOUR EXACT CODE FROM main.py
            # ============================================================
            metrics = self.analyze_performance_exact(vehicles, uavs, base_stations)
            metrics['seed'] = seed
            metrics['algorithm'] = algorithm
            metrics['alpha'] = alpha

            return metrics

        except Exception as e:
            print(f"   ⚠️ Error in simulation (seed={seed}): {e}")
            import traceback
            traceback.print_exc()
            return None

    def analyze_performance_exact(self, vehicles, uavs, base_stations):
        """
        EXACT COPY of analyze_performance_EXACT from your main.py
        """
        # Vehicle performance - EXACT variables from your code
        vehicle_cache_hits = sum(v.cache_hit for v in vehicles)
        vehicle_cache_request = sum(v.request_receive_cache for v in vehicles)
        vehicle_source_hits = sum(v.source_hit for v in vehicles)
        vehicle_total_requests = sum(getattr(v, 'total_request', 0) for v in vehicles)

        # UAV performance - EXACT variables (u.content_hit, NOT u.cache_hit!)
        uav_cache_hits = sum(u.content_hit for u in uavs)
        uav_cache_request = sum(u.request_receive_from_other_source for u in uavs)
        uav_source_hits = sum(u.content_hit_for_direct_uav for u in uavs)
        uav_sagin_hits = sum(u.content_hit_for_sagin_links for u in uavs)
        uav_total_requests = sum(u.global_request_receive_count for u in uavs)

        # BS performance - EXACT variables
        bs_cache_hits = sum(b.cache_hit for b in base_stations)
        bs_cache_request = sum(b.request_receive_cache for b in base_stations)
        bs_source_hits = sum(b.source_hit for b in base_stations)
        bs_total_requests = sum(b.total_request for b in base_stations)

        # Overall
        total_cache_hits = vehicle_cache_hits + uav_cache_hits + bs_cache_hits
        total_source_hits = vehicle_source_hits + uav_source_hits + bs_source_hits + uav_sagin_hits
        total_content_hits = total_cache_hits + total_source_hits
        total_requests = vehicle_total_requests + uav_total_requests + bs_total_requests

        # Calculate ratios - EXACT formulas from your code
        vehicle_hit_ratio = ((vehicle_cache_hits + vehicle_source_hits) / max(1, vehicle_total_requests)) * 100
        uav_hit_ratio = ((uav_cache_hits + uav_source_hits + uav_sagin_hits) / max(1, uav_total_requests)) * 100
        bs_hit_ratio = ((bs_cache_hits + bs_source_hits) / max(1, bs_total_requests)) * 100
        overall_hit_ratio = (total_content_hits / max(1, total_requests)) * 100

        vehicle_cache_ratio = (vehicle_cache_hits / max(1, vehicle_cache_request)) * 100
        uav_cache_ratio = (uav_cache_hits / max(1, uav_cache_request)) * 100
        bs_cache_ratio = (bs_cache_hits / max(1, bs_cache_request)) * 100
        overall_cache_ratio = (total_cache_hits / max(1, total_content_hits)) * 100

        return {
            'vehicle_hit_ratio': vehicle_hit_ratio,
            'uav_hit_ratio': uav_hit_ratio,
            'bs_hit_ratio': bs_hit_ratio,
            'overall_hit_ratio': overall_hit_ratio,
            'vehicle_cache_ratio': vehicle_cache_ratio,
            'uav_cache_ratio': uav_cache_ratio,
            'bs_cache_ratio': bs_cache_ratio,
            'overall_cache_ratio': overall_cache_ratio,
            'total_requests': total_requests,
            'total_cache_hits': total_cache_hits,
            # Raw data for verification
            'raw_vehicle_hits': vehicle_cache_hits,
            'raw_uav_hits': uav_cache_hits,
            'raw_bs_hits': bs_cache_hits
        }

    def calculate_statistics(self, data_list, metric_name):
        """Calculate statistics for a metric across multiple runs"""
        data = np.array([d[metric_name] for d in data_list if d is not None])

        if len(data) < 2:
            return {
                'mean': data[0] if len(data) == 1 else np.nan,
                'std': 0.0,
                'ci_lower': data[0] if len(data) == 1 else np.nan,
                'ci_upper': data[0] if len(data) == 1 else np.nan,
                'ci_margin': 0.0,
                'n_runs': len(data),
                'raw_values': data.tolist()
            }

        mean = np.mean(data)
        std = np.std(data, ddof=1)
        stderr = stats.sem(data)
        t_critical = stats.t.ppf((1 + self.confidence_level) / 2, len(data) - 1)
        ci_margin = t_critical * stderr

        return {
            'mean': mean,
            'std': std,
            'stderr': stderr,
            'ci_lower': mean - ci_margin,
            'ci_upper': mean + ci_margin,
            'ci_margin': ci_margin,
            'min': np.min(data),
            'max': np.max(data),
            'n_runs': len(data),
            'raw_values': data.tolist()
        }

    def run_statistical_experiment(self, algorithm, alpha):
        """Run multiple simulations for one configuration"""
        print(f"\n🔬 Running {self.num_runs} simulations: {algorithm} (α={alpha})")

        base_seed = hash(f"{algorithm}_{alpha}") % (2 ** 31)
        seeds = [base_seed + i * 1000 for i in range(self.num_runs)]

        run_results = []
        for i, seed in enumerate(seeds):
            print(f"   Run {i + 1}/{self.num_runs} (seed={seed})...", end=" ", flush=True)
            result = self.run_single_simulation(algorithm, alpha, seed)
            if result:
                run_results.append(result)
                print(f"✓ Overall: {result['overall_hit_ratio']:.2f}% | "
                      f"Veh: {result['vehicle_hit_ratio']:.2f}% | "
                      f"UAV: {result['uav_hit_ratio']:.2f}%")
            else:
                print("✗ Failed")

        # Calculate statistics
        metrics = ['vehicle_hit_ratio', 'uav_hit_ratio', 'bs_hit_ratio', 'overall_hit_ratio',
                   'vehicle_cache_ratio', 'uav_cache_ratio', 'bs_cache_ratio', 'overall_cache_ratio']

        statistics = {
            'algorithm': algorithm,
            'alpha': alpha,
            'num_runs': len(run_results),
            'seeds_used': seeds[:len(run_results)]
        }

        for metric in metrics:
            statistics[metric] = self.calculate_statistics(run_results, metric)

        # Print summary for this config
        if run_results:
            overall = statistics['overall_hit_ratio']
            print(f"   📊 Summary: {overall['mean']:.2f}% ± {overall['ci_margin']:.2f}% (95% CI)")

        return statistics

    def run_full_batch_analysis(self):
        """Run analysis for all configurations"""
        print(f"\n{'=' * 70}")
        print("📊 FULL STATISTICAL BATCH ANALYSIS")
        print(f"{'=' * 70}")
        print(f"Algorithms: {', '.join(self.algorithms)}")
        print(f"Alpha values: {self.alpha_values}")
        print(f"Runs per config: {self.num_runs}")
        print(f"Time slots: {self.time_slots}")
        total_simulations = len(self.algorithms) * len(self.alpha_values) * self.num_runs
        print(f"Total simulations: {total_simulations}")
        print(f"{'=' * 70}")

        all_results = []
        start_time = time.time()

        for alpha in self.alpha_values:
            for algorithm in self.algorithms:
                stats_result = self.run_statistical_experiment(algorithm, alpha)
                all_results.append(stats_result)

                elapsed = time.time() - start_time
                completed = len(all_results)
                total = len(self.algorithms) * len(self.alpha_values)
                remaining = (elapsed / completed) * (total - completed) if completed > 0 else 0
                print(f"   ⏱️ Progress: {completed}/{total} | "
                      f"Elapsed: {elapsed / 60:.1f}min | Remaining: {remaining / 60:.1f}min")

        self.save_results(all_results)
        return all_results

    def save_results(self, results):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON with full details
        json_file = self.results_dir / f"statistical_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 JSON saved: {json_file}")

        # CSV summary
        csv_data = []
        for r in results:
            row = {
                'Algorithm': r['algorithm'],
                'Alpha': r['alpha'],
                'N_Runs': r['num_runs'],
            }
            for metric in ['vehicle_hit_ratio', 'uav_hit_ratio', 'bs_hit_ratio', 'overall_hit_ratio']:
                s = r[metric]
                row[f'{metric}_mean'] = s['mean']
                row[f'{metric}_std'] = s['std']
                row[f'{metric}_ci_lower'] = s['ci_lower']
                row[f'{metric}_ci_upper'] = s['ci_upper']
            csv_data.append(row)

        csv_file = self.results_dir / f"statistical_summary_{timestamp}.csv"
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        print(f"📊 CSV saved: {csv_file}")

        # Print summary table
        self.print_summary(results)

    def print_summary(self, results):
        """Print formatted summary"""
        print(f"\n{'=' * 100}")
        print("📊 STATISTICAL SUMMARY (Mean ± 95% CI)")
        print(f"{'=' * 100}")
        print(f"{'Algorithm':<25} {'Alpha':<8} {'Overall Hit %':<25} {'N Runs':<8}")
        print(f"{'-' * 100}")

        for r in results:
            overall = r['overall_hit_ratio']
            ci_str = f"{overall['mean']:.2f} ± {overall['ci_margin']:.2f}"
            ci_range = f"[{overall['ci_lower']:.2f}, {overall['ci_upper']:.2f}]"
            print(f"{r['algorithm']:<25} {r['alpha']:<8} {ci_str:<15} {ci_range:<15} {r['num_runs']:<8}")


def main():
    parser = argparse.ArgumentParser(description="Statistical Simulation Runner")
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of runs per configuration (default: 10)')
    parser.add_argument('--batch', action='store_true',
                        help='Run full batch analysis')
    parser.add_argument('--algorithm', type=str,
                        help='Single algorithm to test')
    parser.add_argument('--alpha', type=float,
                        help='Single alpha value to test')

    args = parser.parse_args()

    runner = StatisticalSimulationRunner(num_runs=args.num_runs)

    if args.batch:
        runner.run_full_batch_analysis()
    elif args.algorithm and args.alpha:
        stats = runner.run_statistical_experiment(args.algorithm, args.alpha)
        runner.save_results([stats])
    else:
        print("Usage:")
        print("  Batch:  python statistical_simulation_runner_fixed.py --num_runs 10 --batch")
        print(
            "  Single: python statistical_simulation_runner_fixed.py --num_runs 5 --algorithm MAB_Contextual --alpha 1.0")


if __name__ == "__main__":
    main()