#!/usr/bin/env python3
"""
🎯 CORRECT BATCH ANALYSIS - EXACT VARIABLE NAMES
=================================================
Uses EXACT variable names from your analyze_performance() method for:
- vehicle_hit_ratio, uav_hit_ratio, bs_hit_ratio, overall_hit_ratio
- vehicle_cache_ratio, uav_cache_ratio, bs_cache_ratio, overall_cache_ratio
"""

import numpy as np
import time
import argparse
import json
import pandas as pd
import math
from datetime import datetime
import logging
from tqdm import tqdm
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Import your modules
from communication import Communication
from federated_mab import FederatedAggregator
from enhanced_federated_mab import EnhancedFederatedAggregator
from vehicle_ccn import Vehicle
from uav_ccn import UAV
from satellite_ccn import Satellite
from bs_ccn import BaseStation
from gs_ccn import GroundStation


def create_network(algorithm_type):
    """Create network components with specified algorithm"""
    logger.info(f"🗏️ Creating network for {algorithm_type}")

    if algorithm_type == "Enhanced_Federated_MAB":
        aggregator = EnhancedFederatedAggregator()
    elif algorithm_type == "Federated_MAB":
        aggregator = FederatedAggregator()
    else:
        aggregator = None  # LRU / Popularity / MAB_Original / MAB_Contextual
    #grid_size = 10
    #uav_grid_size = 20

    # Create satellites
    satellites = {}
    for i in range(1, 3):
        satellite = Satellite(f"Satellite{i}")
        satellites[f"Satellite{i}"] = satellite

    # Create ground station
    ground_station = GroundStation("GS1")

    # Create UAVs

    # --- choose sensible grid values (match your older setup) ---
    grid_size = 100  # 10 km region represented by 100x100 logical grid cells
    uav_grid_size = 50  # each UAV covers a 20x20 block -> (100//20)=5 per side

    # --- compute UAV grid dynamically like before ---
    uavs = []
    uav_side = grid_size // uav_grid_size  # e.g., 5
    uav_count = uav_side * uav_side  # e.g., 25

    # Create UAVs with IDs UAV1..UAV{uav_count}
    for i in range(uav_count):
        uav = UAV(f"UAV{i + 1}", grid_size, uav_grid_size, aggregator, algorithm=algorithm_type)
        uavs.append(uav)

    # --- set 2-D neighbors (top/bottom/left/right) like your earlier code ---
    for i in range(uav_count):
        row = i // uav_side
        col = i % uav_side

        neigh = []
        # top
        if row > 0:
            neigh.append(uavs[i - uav_side])
        # bottom
        if row < uav_side - 1:
            neigh.append(uavs[i + uav_side])
        # left
        if col > 0:
            neigh.append(uavs[i - 1])
        # right
        if col < uav_side - 1:
            neigh.append(uavs[i + 1])

        uavs[i].neighbors = neigh


    # Create vehicles
    vehicles = []
    for i in range(1, 10):
        vehicle = Vehicle(f"Vehicle{i}", grid_size, 20, 2, aggregator, algorithm=algorithm_type)
        vehicles.append(vehicle)

    # vehicles already created in a list called `vehicles`
    num_rows = int(math.sqrt(len(vehicles)))
    num_cols = max(1, len(vehicles) // num_rows)
    row_step = grid_size // num_rows
    col_step = grid_size // num_cols

    k = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if k >= len(vehicles):
                break
            x = j * col_step + col_step // 2
            y = i * row_step + row_step // 2
            vehicles[k].current_location = (min(x, grid_size - 1), min(y, grid_size - 1))
            k += 1

    # Create base stations
    base_stations = []
    for i in range(1, 1):
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


def analyze_performance_EXACT(vehicles, uavs, base_stations, algorithm, alpha, time_slots):
    """📊 USES EXACT VARIABLE NAMES FROM YOUR analyze_performance METHOD"""

    logger.info(f"📊 Analyzing performance for {algorithm} (α={alpha})")

    # ====================================================================
    # EXACT VARIABLE NAMES FROM YOUR main.py analyze_performance METHOD
    # ====================================================================

    # Vehicle performance - EXACT variables
    vehicle_cache_hits = sum(v.cache_hit for v in vehicles)
    vehicle_cache_request = sum(v.request_receive_cache for v in vehicles)
    vehicle_source_hits = sum(v.source_hit for v in vehicles)
    vehicle_total_requests = sum(getattr(v, 'total_request', 0) for v in vehicles)

    # UAV performance - EXACT variables
    uav_cache_hits = sum(u.content_hit for u in uavs)  # ✅ u.content_hit (NOT u.cache_hit!)
    uav_cache_request = sum(u.request_receive_from_other_source for u in uavs)
    uav_source_hits = sum(u.content_hit_for_direct_uav for u in uavs)
    uav_sagin_hits = sum(u.content_hit_for_sagin_links for u in uavs)
    uav_total_requests = sum(u.global_request_receive_count for u in uavs)

    # BS performance - EXACT variables
    bs_cache_hits = sum(b.cache_hit for b in base_stations)
    bs_cache_request = sum(b.request_receive_cache for b in base_stations)
    bs_source_hits = sum(b.source_hit for b in base_stations)
    bs_total_requests = sum(b.total_request for b in base_stations)

    # ====================================================================
    # EXACT CALCULATIONS FROM YOUR CODE
    # ====================================================================

    # Overall performance
    total_cache_hits = vehicle_cache_hits + uav_cache_hits + bs_cache_hits
    total_source_hits = vehicle_source_hits + uav_source_hits + bs_source_hits + uav_sagin_hits
    total_content_hits = total_cache_hits + total_source_hits

    # Calculate hit ratios - EXACT same formula as your code
    vehicle_hit_ratio = ((vehicle_cache_hits + vehicle_source_hits) / max(1, vehicle_total_requests)) * 100
    uav_hit_ratio = ((uav_cache_hits + uav_source_hits + uav_sagin_hits) / max(1, uav_total_requests)) * 100
    bs_hit_ratio = ((bs_cache_hits + bs_source_hits) / max(1, bs_total_requests)) * 100
    overall_hit_ratio = (total_content_hits / max(1,
                                                  vehicle_total_requests + uav_total_requests + bs_total_requests)) * 100

    # Cache hit ratios - EXACT same formula as your code
    vehicle_cache_ratio = (vehicle_cache_hits / max(1, vehicle_cache_request)) * 100
    uav_cache_ratio = (uav_cache_hits / max(1, uav_cache_request)) * 100
    bs_cache_ratio = (bs_cache_hits / max(1, bs_cache_request)) * 100
    overall_cache_ratio = (total_cache_hits / max(1, total_content_hits)) * 100

    # Debug verification
    logger.debug(f"   🔍 Raw Data Verification:")
    logger.debug(
        f"   Vehicle: cache_hits={vehicle_cache_hits}, cache_requests={vehicle_cache_request}, source_hits={vehicle_source_hits}, total_requests={vehicle_total_requests}")
    logger.debug(
        f"   UAV: cache_hits={uav_cache_hits}, cache_requests={uav_cache_request}, source_hits={uav_source_hits}, sagin_hits={uav_sagin_hits}, total_requests={uav_total_requests}")
    logger.debug(
        f"   BS: cache_hits={bs_cache_hits}, cache_requests={bs_cache_request}, source_hits={bs_source_hits}, total_requests={bs_total_requests}")

    return {
        'algorithm': algorithm,
        'alpha': alpha,
        'time_slots': time_slots,

        # ✅ YOUR REQUESTED HIT RATIOS
        'vehicle_hit_ratio': vehicle_hit_ratio,
        'uav_hit_ratio': uav_hit_ratio,
        'bs_hit_ratio': bs_hit_ratio,
        'overall_hit_ratio': overall_hit_ratio,

        # ✅ YOUR REQUESTED CACHE RATIOS
        'vehicle_cache_ratio': vehicle_cache_ratio,
        'uav_cache_ratio': uav_cache_ratio,
        'bs_cache_ratio': bs_cache_ratio,
        'overall_cache_ratio': overall_cache_ratio,

        # Raw data for verification
        'raw_data': {
            'vehicle': {
                'cache_hits': vehicle_cache_hits,
                'cache_requests': vehicle_cache_request,
                'source_hits': vehicle_source_hits,
                'total_requests': vehicle_total_requests
            },
            'uav': {
                'cache_hits': uav_cache_hits,
                'cache_requests': uav_cache_request,
                'source_hits': uav_source_hits,
                'sagin_hits': uav_sagin_hits,
                'total_requests': uav_total_requests
            },
            'bs': {
                'cache_hits': bs_cache_hits,
                'cache_requests': bs_cache_request,
                'source_hits': bs_source_hits,
                'total_requests': bs_total_requests
            },
            'overall': {
                'total_cache_hits': total_cache_hits,
                'total_source_hits': total_source_hits,
                'total_content_hits': total_content_hits,
                'total_requests': vehicle_total_requests + uav_total_requests + bs_total_requests
            }
        }
    }


def run_single_simulation(algorithm, alpha, time_slots, simulation_round=1):
    """🎯 YOUR EXISTING WORKING SIMULATION - UNCHANGED"""

    start_time = time.time()

    # Create archive directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path("results") / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Created archive directory: {archive_dir}")

    logger.info(f"\n🚀 RUNNING SIMULATION")
    logger.info(f"   Algorithm: {algorithm}")
    logger.info(f"   Alpha (Zipf): {alpha}")
    logger.info(f"   Time Slots: {time_slots}")
    logger.info("=" * 50)

    # Create network - YOUR EXISTING CODE
    satellites, uavs, vehicles, base_stations, ground_station, aggregator = create_network(algorithm)

    # Create communication - YOUR EXISTING CODE
    communication = Communication(satellites, base_stations, vehicles, uavs, ground_station, alpha, simulation_round,
                                  time_slots, archive_dir)


    # ✅ ADD THIS (with correct name):
    communication.current_algorithm = algorithm

    # 🔥 CRITICAL: ADD ALPHA TO ALL NODES (INSERT THIS BLOCK HERE!)
    logger.info(f"🔧 Setting alpha={alpha} to all nodes...")

    # Pass alpha to all vehicles
    for vehicle in vehicles:
        vehicle.current_alpha = alpha
        logger.info(f"   🚗 {vehicle.vehicle_id}: alpha={alpha}")

    # Pass alpha to all UAVs
    for uav in uavs:
        uav.current_alpha = alpha
        logger.info(f"   🚁 {uav.uav_id}: alpha={alpha}")

    # Pass alpha to all base stations
    for bs in base_stations:
        bs.current_alpha = alpha
        logger.info(f"   🏢 {bs.bs_id}: alpha={alpha}")


    # 🎯 YOUR EXISTING SIMULATION PARAMETERS - UNCHANGED
    grid_size = 100
    no_of_content_each_category = 10
    no_of_request_genertaed_in_each_timeslot = 50
    uav_content_generation_period = 5
    consecutive_slots_g = 5
    epsilon = 0.1

    # Communication schedule - YOUR EXISTING CODE
    communication_schedule = {slot: [1, 2, 3] for slot in range(1, time_slots + 1)}

    # 🎯 YOUR EXISTING WORKING SIMULATION LOOP - COMPLETELY UNCHANGED
    for slot in tqdm(range(1, time_slots + 1), desc="Simulating", unit="slot"):
        current_time = (slot - 1) * 60

        # ✅ Satellite operations - YOUR EXISTING CODE
        if (slot - 1) % consecutive_slots_g == 0:
            for satellite_id in satellites:
                satellites[satellite_id].run(satellites, communication, communication_schedule,
                                             current_time, slot, no_of_content_each_category, ground_station)
            ground_station.run(current_time, satellites)

        # ✅ UAV operations - YOUR EXISTING CODE
        for uav in uavs:
            uav.run(current_time, communication, communication_schedule, slot, satellites,
                    no_of_content_each_category, uav_content_generation_period, epsilon, uavs)

        # ✅ Vehicle operations - YOUR EXISTING CODE
        for vehicle in vehicles:
            vehicle.run(current_time, slot, time_slots, vehicles, uavs, base_stations, satellites,
                        communication, no_of_request_genertaed_in_each_timeslot,
                        no_of_content_each_category)

        # ✅ Base station operations - YOUR EXISTING CODE
        for bs in base_stations:
            bs.run(current_time, slot)

        # ✅ FIX: Add Enhanced_Federated_MAB to condition
        #print(f"slottt{slot}, condition={(slot - 1) % 10 == 0}")
        if algorithm in ["Federated_MAB", "Enhanced_Federated_MAB"] and slot > 10 and ((slot - 1) % 10 == 0):
            logger.debug(f"🔍 TRIGGERING aggregate_updates() for {algorithm} at slot {slot}")  # ADD THIS
            aggregator.aggregate_updates()
        else:
            if algorithm == "Enhanced_Federated_MAB":
                logger.debug(f"🔍 NOT TRIGGERING: algorithm={algorithm}, slot={slot}, condition={(slot - 1) % 10 == 0}")

                # ✅ Communication processing - YOUR EXISTING CODE
        communication.run(vehicles, uavs, base_stations, satellites, grid_size, current_time,
                          slot, communication_schedule, time_slots, ground_station)



    # 🎯 ANALYZE WITH EXACT VARIABLES
    performance = analyze_performance_EXACT(vehicles, uavs, base_stations, algorithm, alpha, time_slots)

    end_time = time.time()
    logger.info(f"Simulation completed in {end_time - start_time:.2f} seconds")

    return performance, archive_dir


def save_results_to_files(results, archive_dir=None):
    """Save results to JSON and CSV files with EXACT format you requested"""

    if not results:
        logger.error("❌ No results to save!")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use archive_dir if provided, else current directory
    base_dir = Path(archive_dir) if archive_dir else Path(".")

    # ====================================================================
    # JSON FILE - DETAILED RESULTS
    # ====================================================================
    json_filename = f"hit_ratio_analysis_{timestamp}.json"
    json_filepath = base_dir / json_filename
    with open(json_filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n💾 DETAILED JSON: {json_filepath}")

    # ====================================================================
    # CSV FILE - SUMMARY TABLE WITH YOUR EXACT COLUMNS
    # ====================================================================
    csv_data = []
    for result in results:
        row = {
            'Algorithm': result['algorithm'],
            'Alpha': result['alpha'],
            'Time_Slots': result['time_slots'],

            # ✅ YOUR EXACT REQUESTED HIT RATIOS
            'vehicle_hit_ratio': result['vehicle_hit_ratio'],
            'uav_hit_ratio': result['uav_hit_ratio'],
            'bs_hit_ratio': result['bs_hit_ratio'],
            'overall_hit_ratio': result['overall_hit_ratio'],

            # ✅ YOUR EXACT REQUESTED CACHE RATIOS
            'vehicle_cache_ratio': result['vehicle_cache_ratio'],
            'uav_cache_ratio': result['uav_cache_ratio'],
            'bs_cache_ratio': result['bs_cache_ratio'],
            'overall_cache_ratio': result['overall_cache_ratio'],

            # Raw verification data
            'Vehicle_Cache_Hits': result['raw_data']['vehicle']['cache_hits'],
            'UAV_Cache_Hits': result['raw_data']['uav']['cache_hits'],
            'BS_Cache_Hits': result['raw_data']['bs']['cache_hits'],
            'Total_Cache_Hits': result['raw_data']['overall']['total_cache_hits'],
            'Total_Requests': result['raw_data']['overall']['total_requests']
        }
        csv_data.append(row)

    csv_filename = f"hit_ratio_analysis_{timestamp}.csv"
    csv_filepath = base_dir / csv_filename
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_filepath, index=False)

    print(f"📊 CSV FILE: {csv_filepath}")

    # ====================================================================
    # CONSOLE OUTPUT - SUMMARY TABLE
    # ====================================================================
    print(f"\n📋 HIT RATIO & CACHE RATIO SUMMARY")
    print("=" * 140)
    print(
        f"{'Algorithm':<15} {'Alpha':<6} {'VehHit%':<8} {'UAVHit%':<8} {'BSHit%':<8} {'Overall%':<9} {'VehCache%':<10} {'UAVCache%':<10} {'BSCache%':<10} {'OverCache%':<11}")
    print("-" * 140)

    for result in results:
        print(f"{result['algorithm']:<15} {result['alpha']:<6.2f} "
              f"{result['vehicle_hit_ratio']:<8.4f} {result['uav_hit_ratio']:<8.4f} {result['bs_hit_ratio']:<8.4f} {result['overall_hit_ratio']:<9.4f} "
              f"{result['vehicle_cache_ratio']:<10.4f} {result['uav_cache_ratio']:<10.4f} {result['bs_cache_ratio']:<10.4f} {result['overall_cache_ratio']:<11.4f}")


def run_batch_analysis():
    """🎯 BATCH ANALYSIS FOR ALL ALGORITHMS AND ALPHA VALUES"""

    print(f"\n🎯 BATCH HIT RATIO ANALYSIS")
    print("=" * 60)

    # Configuration - YOUR REQUIREMENTS
    #algorithms = ['LRU', 'Popularity', 'MAB_Original', 'MAB_Contextual', 'Enhanced_Federated_MAB']  # ✅ 5 algorithms
    #algorithms =['LRU']
    algorithms = ['MAB_Contextual']
    alpha_values = [0.25, 0.5, 1.0, 2.0]  # ✅ All alpha values
    time_slots_options = [300]  # ✅ All simulation slots

    total_configs = len(algorithms) * len(alpha_values) * len(time_slots_options)

    print(f"📊 Configuration:")
    print(f"   • Algorithms: {len(algorithms)} ({', '.join(algorithms)})")
    print(f"   • Alpha values: {len(alpha_values)} ({', '.join(map(str, alpha_values))})")
    print(f"   • Time slots: {len(time_slots_options)} ({', '.join(map(str, time_slots_options))})")
    print(f"   • Total configurations: {total_configs}")

    # Confirm execution
    # response = input(f"\n⚠️ This will run {total_configs} configurations. Continue? (y/n): ")
    # if response.lower() != 'y':
    #     print("❌ Batch analysis cancelled.")
    #     return None
    response = 'y'

    batch_results = []
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_archive_dir = Path("results") / f"batch_{batch_timestamp}"
    batch_archive_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Created batch summary directory: {batch_archive_dir}")

    config_count = 0
    start_time = time.time()

    for alpha in alpha_values:  # ✅ All alpha values
        for time_slots in time_slots_options:  # ✅ All simulation slots
            for algorithm in algorithms:  # ✅ 5 algorithms
                config_count += 1

                print(f"\n⚙️ Configuration {config_count}/{total_configs}")
                print(f"   🔄 α={alpha} | slots={time_slots} | {algorithm}")

                try:
                    # 🎯 RUN YOUR EXISTING WORKING SIMULATION
                    performance, sim_archive_dir = run_single_simulation(algorithm, alpha, time_slots)

                    if performance:
                        performance['config_id'] = config_count
                        performance['timestamp'] = datetime.now().isoformat()
                        batch_results.append(performance)

                        # Save each run's hit-ratio files to its own run archive directory.
                        save_results_to_files([performance], sim_archive_dir)

                        # Show key results
                        print(f"      ✅ Vehicle Hit: {performance['vehicle_hit_ratio']:.4f}%")
                        print(f"      ✅ UAV Hit: {performance['uav_hit_ratio']:.4f}%")
                        print(f"      ✅ BS Hit: {performance['bs_hit_ratio']:.4f}%")
                        print(f"      ✅ Overall Hit: {performance['overall_hit_ratio']:.4f}%")
                        print(f"      ✅ Vehicle Cache: {performance['vehicle_cache_ratio']:.4f}%")
                        print(f"      ✅ UAV Cache: {performance['uav_cache_ratio']:.4f}%")
                        print(f"      ✅ BS Cache: {performance['bs_cache_ratio']:.4f}%")
                        print(f"      ✅ Overall Cache: {performance['overall_cache_ratio']:.4f}%")
                    else:
                        print(f"      ❌ Simulation failed")

                    # Progress update
                    elapsed = time.time() - start_time
                    remaining = (elapsed / config_count) * (total_configs - config_count)
                    print(f"      ⏱️ Elapsed: {elapsed / 60:.1f}min | Est. remaining: {remaining / 60:.1f}min")

                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    continue

    # Save batch summary (all runs) to a separate batch summary directory.
    if batch_results:
        save_results_to_files(batch_results, batch_archive_dir)

        # Algorithm ranking
        print(f"\n🏆 ALGORITHM RANKING BY OVERALL HIT RATIO:")
        print("-" * 60)

        df = pd.DataFrame([{
            'Algorithm': result['algorithm'],
            'Alpha': result['alpha'],
            'Overall_Hit_Ratio': result['overall_hit_ratio']
        } for result in batch_results])

        algo_stats = df.groupby('Algorithm')['Overall_Hit_Ratio'].agg(['mean', 'std']).round(4)
        algo_stats = algo_stats.sort_values('mean', ascending=False)

        rank = 1
        for algorithm, stats in algo_stats.iterrows():
            print(f"{rank}. {algorithm:<15} | Avg: {stats['mean']:6.4f}% ± {stats['std']:5.4f}%")
            rank += 1

    return batch_results


def main():
    parser = argparse.ArgumentParser(description="HIT RATIO ANALYSIS")

    # Single simulation mode
    parser.add_argument('--algorithm', type=str,
                        choices=['LRU', 'Popularity', 'MAB_Original', 'MAB_Contextual', 'Federated_MAB', 'Enhanced_Federated_MAB'],
                        help='Algorithm for single simulation')
    parser.add_argument('--alpha', type=float, help='Zipf alpha parameter')
    parser.add_argument('--time_slots', type=int, help='Number of time slots')

    # Batch mode
    parser.add_argument('--batch', action='store_true',
                        help='Run batch analysis across all algorithms and parameters')

    args = parser.parse_args()

    print("🎯 HIT RATIO & CACHE RATIO ANALYSIS")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        if args.batch:
            # Batch mode - ALL ALGORITHMS, ALL ALPHA VALUES, ALL SIMULATION SLOTS
            run_batch_analysis()

        elif args.algorithm and args.alpha is not None and args.time_slots:
            # Single simulation mode
            performance, archive_dir = run_single_simulation(args.algorithm, args.alpha, args.time_slots)
            if performance:
                # Save single result
                save_results_to_files([performance], archive_dir)

        else:
            print("❌ Invalid arguments. Use either:")
            print("   Single: --algorithm MAB_Contextual --alpha 2.0 --time_slots 100")
            print("   Batch:  --batch")

    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

    print(f"\n🏁 Analysis finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()