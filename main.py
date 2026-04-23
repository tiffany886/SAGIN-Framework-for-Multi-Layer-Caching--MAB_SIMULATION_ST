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
import traceback
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


def get_energy_lambda_for_algorithm(algorithm_type):
    energy_lambda_map = {
        'MAB_Contextual': 0.0,
        'Federated_MAB': 0.0,
        'Enhanced_Federated_MAB': 0.0,
        'MAB_Contextual_EnergyAware': 0.2,
        'Federated_MAB_EnergyAware': 0.2,
        'Enhanced_Federated_MAB_EnergyAware': 0.2,
    }
    return energy_lambda_map.get(algorithm_type, 0.0)


def create_network(algorithm_type, energy_lambda=None):
    """Create network components with specified algorithm"""
    algorithm_type = str(algorithm_type).strip()
    logger.info(f"🗏️ Creating network for {algorithm_type}")

    if energy_lambda is None:
        energy_lambda = get_energy_lambda_for_algorithm(algorithm_type)

    if algorithm_type in ["Enhanced_Federated_MAB", "Enhanced_Federated_MAB_EnergyAware"]:
        aggregator = EnhancedFederatedAggregator()
    elif algorithm_type in ["Federated_MAB", "Federated_MAB_EnergyAware"]:
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
        uav = UAV(f"UAV{i + 1}", grid_size, uav_grid_size, aggregator, algorithm=algorithm_type,
                  energy_lambda=energy_lambda)
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
        vehicle = Vehicle(f"Vehicle{i}", grid_size, 20, 2, aggregator, algorithm=algorithm_type,
                          energy_lambda=energy_lambda)
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
        bs = BaseStation(f"BS{i}", 15, grid_size, aggregator, algorithm=algorithm_type,
                         energy_lambda=energy_lambda)
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

    # Energy performance
    total_energy_consumed = 0.0
    total_energy_samples = 0
    for node_group in (vehicles, uavs, base_stations):
        for node in node_group:
            for entity_type, coord_dict in getattr(node, 'record', {}).items():
                for coord, category_dict in coord_dict.items():
                    for category, content_no_dict in category_dict.items():
                        for content_no, content_info in content_no_dict.items():
                            total_energy_consumed += float(content_info.get('energy_sum', 0.0))
                            total_energy_samples += int(content_info.get('energy_count', 0))

    # ====================================================================
    # EXACT CALCULATIONS FROM YOUR CODE
    # ====================================================================

    # Overall performance
    total_cache_hits = vehicle_cache_hits + uav_cache_hits + bs_cache_hits
    total_source_hits = vehicle_source_hits + uav_source_hits + bs_source_hits + uav_sagin_hits
    # Keep initialized even if logic above is edited in future branches.
    total_content_hits = 0
    total_content_hits = total_cache_hits + total_source_hits
    avg_energy_per_request = total_energy_consumed / max(1, total_energy_samples)
    energy_efficiency = total_content_hits / max(1e-6, total_energy_consumed)

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

        # Energy metrics
        'avg_energy_per_request': avg_energy_per_request,
        'energy_efficiency': energy_efficiency,
        'total_energy_consumed': total_energy_consumed,
        'energy_samples': total_energy_samples,

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
                'total_requests': vehicle_total_requests + uav_total_requests + bs_total_requests,
                'total_energy_consumed': total_energy_consumed,
                'energy_samples': total_energy_samples,
            }
        }
    }


def run_single_simulation(algorithm, alpha, time_slots, simulation_round=1, output_root=None, run_subdir=None):
    """🎯 YOUR EXISTING WORKING SIMULATION - UNCHANGED"""

    algorithm = str(algorithm).strip()
    start_time = time.time()

    # Create archive directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_root = Path(output_root) if output_root else Path("results")
    archive_dir = base_root / run_subdir if run_subdir else base_root / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Created archive directory: {archive_dir}")

    logger.info(f"\n🚀 RUNNING SIMULATION")
    logger.info(f"   Algorithm: {algorithm}")
    logger.info(f"   Alpha (Zipf): {alpha}")
    logger.info(f"   Time Slots: {time_slots}")
    logger.info("=" * 50)

    # Create network - YOUR EXISTING CODE
    energy_lambda = get_energy_lambda_for_algorithm(algorithm)
    satellites, uavs, vehicles, base_stations, ground_station, aggregator = create_network(algorithm, energy_lambda)

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
        bs.energy_lambda = energy_lambda

    for vehicle in vehicles:
        vehicle.energy_lambda = energy_lambda

    for uav in uavs:
        uav.energy_lambda = energy_lambda


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
        if algorithm in [
            "Federated_MAB",
            "Federated_MAB_EnergyAware",
            "Enhanced_Federated_MAB",
            "Enhanced_Federated_MAB_EnergyAware",
        ] and slot > 10 and ((slot - 1) % 10 == 0):
            logger.debug(f"🔍 TRIGGERING aggregate_updates() for {algorithm} at slot {slot}")  # ADD THIS
            aggregator.aggregate_updates()
        else:
            if algorithm in ["Enhanced_Federated_MAB", "Enhanced_Federated_MAB_EnergyAware"]:
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

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

            # ✅ ENERGY METRICS
            'avg_energy_per_request': result.get('avg_energy_per_request', 0.0),
            'energy_efficiency': result.get('energy_efficiency', 0.0),
            'total_energy_consumed': result.get('total_energy_consumed', 0.0),
            'energy_samples': result.get('energy_samples', 0),

            # Raw verification data
            'Vehicle_Cache_Hits': result['raw_data']['vehicle']['cache_hits'],
            'UAV_Cache_Hits': result['raw_data']['uav']['cache_hits'],
            'BS_Cache_Hits': result['raw_data']['bs']['cache_hits'],
            'Total_Cache_Hits': result['raw_data']['overall']['total_cache_hits'],
            'Total_Requests': result['raw_data']['overall']['total_requests'],
            'Total_Energy_Consumed': result['raw_data']['overall'].get('total_energy_consumed', 0.0),
            'Energy_Samples': result['raw_data']['overall'].get('energy_samples', 0),
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
    print(f"\n📋 HIT RATIO, CACHE RATIO & ENERGY SUMMARY")
    print("=" * 175)
    print(
        f"{'Algorithm':<15} {'Alpha':<6} {'VehHit%':<8} {'UAVHit%':<8} {'BSHit%':<8} {'Overall%':<9} {'VehCache%':<10} {'UAVCache%':<10} {'BSCache%':<10} {'OverCache%':<11} {'AvgEnergy':<10} {'Eff':<8}")
    print("-" * 175)

    for result in results:
        print(f"{result['algorithm']:<15} {result['alpha']:<6.2f} "
              f"{result['vehicle_hit_ratio']:<8.4f} {result['uav_hit_ratio']:<8.4f} {result['bs_hit_ratio']:<8.4f} {result['overall_hit_ratio']:<9.4f} "
              f"{result['vehicle_cache_ratio']:<10.4f} {result['uav_cache_ratio']:<10.4f} {result['bs_cache_ratio']:<10.4f} {result['overall_cache_ratio']:<11.4f} "
              f"{result.get('avg_energy_per_request', 0.0):<10.4f} {result.get('energy_efficiency', 0.0):<8.4f}")

    return {
        'json': str(json_filepath),
        'csv': str(csv_filepath),
    }


def save_batch_summary_files(results, summary_dir):
    """Save one deterministic batch summary in summary_dir."""
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    json_filepath = summary_dir / "hit_ratio_analysis_all.json"
    csv_filepath = summary_dir / "hit_ratio_analysis_all.csv"

    with open(json_filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    csv_data = []
    for result in results:
        csv_data.append({
            'Algorithm': result['algorithm'],
            'Alpha': result['alpha'],
            'Time_Slots': result['time_slots'],
            'vehicle_hit_ratio': result['vehicle_hit_ratio'],
            'uav_hit_ratio': result['uav_hit_ratio'],
            'bs_hit_ratio': result['bs_hit_ratio'],
            'overall_hit_ratio': result['overall_hit_ratio'],
            'vehicle_cache_ratio': result['vehicle_cache_ratio'],
            'uav_cache_ratio': result['uav_cache_ratio'],
            'bs_cache_ratio': result['bs_cache_ratio'],
            'overall_cache_ratio': result['overall_cache_ratio'],
            'avg_energy_per_request': result.get('avg_energy_per_request', 0.0),
            'energy_efficiency': result.get('energy_efficiency', 0.0),
            'total_energy_consumed': result.get('total_energy_consumed', 0.0),
            'energy_samples': result.get('energy_samples', 0),
            'Vehicle_Cache_Hits': result['raw_data']['vehicle']['cache_hits'],
            'UAV_Cache_Hits': result['raw_data']['uav']['cache_hits'],
            'BS_Cache_Hits': result['raw_data']['bs']['cache_hits'],
            'Total_Cache_Hits': result['raw_data']['overall']['total_cache_hits'],
            'Total_Requests': result['raw_data']['overall']['total_requests'],
            'Total_Energy_Consumed': result['raw_data']['overall'].get('total_energy_consumed', 0.0),
            'Energy_Samples': result['raw_data']['overall'].get('energy_samples', 0),
        })

    pd.DataFrame(csv_data).to_csv(csv_filepath, index=False)

    return {
        'json': str(json_filepath),
        'csv': str(csv_filepath),
    }


def _config_key(algorithm, alpha, time_slots):
    return f"{str(algorithm).strip()}|{float(alpha):.6f}|{int(time_slots)}"


def _manifest_path(batch_archive_dir):
    return Path(batch_archive_dir) / "manifest.json"


def _load_manifest(batch_archive_dir):
    path = _manifest_path(batch_archive_dir)
    if not path.exists():
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load manifest: %s", path)
        return None


def _save_manifest(batch_archive_dir, manifest):
    path = _manifest_path(batch_archive_dir)
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)


def _init_manifest(batch_archive_dir, algorithms, alpha_values, time_slots_options):
    return {
        'batch_id': Path(batch_archive_dir).name,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'status': 'running',
        'config': {
            'algorithms': algorithms,
            'alpha_values': alpha_values,
            'time_slots_options': time_slots_options,
            'total_planned': len(algorithms) * len(alpha_values) * len(time_slots_options),
        },
        'paths': {
            'runs_dir': str(Path(batch_archive_dir) / 'runs'),
            'summary_dir': str(Path(batch_archive_dir) / 'summary'),
            'analysis_dir': str(Path(batch_archive_dir) / 'analysis'),
        },
        'summary_files': {},
        'configs': {},
    }


def _extract_performance_records_from_json(json_path):
    records = []
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and {'algorithm', 'alpha', 'time_slots'}.issubset(item.keys()):
                    records.append(item)
        elif isinstance(data, dict) and {'algorithm', 'alpha', 'time_slots'}.issubset(data.keys()):
            records.append(data)
    except Exception:
        logger.exception("Failed to parse existing batch result JSON: %s", json_path)
    return records


def _load_existing_batch_results(batch_archive_dir):
    loaded_by_key = {}

    # 1) Load per-run outputs from run_* directories (new layout).
    runs_dir = Path(batch_archive_dir) / "runs"
    for run_dir in sorted(runs_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        for json_file in sorted(run_dir.glob("hit_ratio_analysis_*.json")):
            for rec in _extract_performance_records_from_json(json_file):
                key = _config_key(rec['algorithm'], rec['alpha'], rec['time_slots'])
                loaded_by_key[key] = rec

    # 2) Also load existing batch summary JSON(s) for compatibility.
    for summary_json in sorted(Path(batch_archive_dir).glob("hit_ratio_analysis_*.json")):
        for rec in _extract_performance_records_from_json(summary_json):
            key = _config_key(rec['algorithm'], rec['alpha'], rec['time_slots'])
            if key not in loaded_by_key:
                loaded_by_key[key] = rec

    return loaded_by_key


def run_batch_analysis(resume_batch_dir=None):
    """🎯 BATCH ANALYSIS FOR ALL ALGORITHMS AND ALPHA VALUES"""

    print(f"\n🎯 BATCH HIT RATIO ANALYSIS")
    print("=" * 60)

    # Configuration - YOUR REQUIREMENTS
    #algorithms = ['LRU', 'Popularity', 'MAB_Original', 'MAB_Contextual', 'Enhanced_Federated_MAB']  # ✅ 5 algorithms
    #algorithms =['LRU']
    algorithms = [
        'LRU',
        'Popularity',
        'MAB_Original',
        'MAB_Contextual',
        'MAB_Contextual_EnergyAware',
        'Federated_MAB',
        'Federated_MAB_EnergyAware',
        'Enhanced_Federated_MAB',
        'Enhanced_Federated_MAB_EnergyAware',
    ]
    alpha_values = [0.25, 0.5, 1.0, 2.0]  # ✅ All alpha values
    time_slots_options = [300]  # ✅ All simulation slots

    all_configs = [
        (algorithm, alpha, time_slots)
        for alpha in alpha_values
        for time_slots in time_slots_options
        for algorithm in algorithms
    ]
    total_configs = len(all_configs)

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

    existing_results_by_key = {}
    manifest = None
    if resume_batch_dir:
        batch_archive_dir = Path(resume_batch_dir)
        if not batch_archive_dir.exists():
            raise FileNotFoundError(f"Resume batch directory not found: {batch_archive_dir}")

        (batch_archive_dir / "runs").mkdir(parents=True, exist_ok=True)
        (batch_archive_dir / "summary").mkdir(parents=True, exist_ok=True)
        (batch_archive_dir / "analysis").mkdir(parents=True, exist_ok=True)

        manifest = _load_manifest(batch_archive_dir)
        if manifest is None:
            manifest = _init_manifest(batch_archive_dir, algorithms, alpha_values, time_slots_options)

        existing_results_by_key = _load_existing_batch_results(batch_archive_dir)
        logger.info(f"📁 Resuming batch summary directory: {batch_archive_dir}")
        logger.info(f"📦 Found {len(existing_results_by_key)} completed configurations")
    else:
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_archive_dir = Path("results") / f"batch_{batch_timestamp}"
        batch_archive_dir.mkdir(parents=True, exist_ok=True)
        (batch_archive_dir / "runs").mkdir(parents=True, exist_ok=True)
        (batch_archive_dir / "summary").mkdir(parents=True, exist_ok=True)
        (batch_archive_dir / "analysis").mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created batch summary directory: {batch_archive_dir}")
        manifest = _init_manifest(batch_archive_dir, algorithms, alpha_values, time_slots_options)

    manifest['updated_at'] = datetime.now().isoformat()
    _save_manifest(batch_archive_dir, manifest)

    runs_dir = batch_archive_dir / "runs"
    summary_dir = batch_archive_dir / "summary"

    completed_keys = set(existing_results_by_key.keys())
    pending_configs = [cfg for cfg in all_configs if _config_key(*cfg) not in completed_keys]

    print(f"   • Already completed: {len(completed_keys)}")
    print(f"   • Pending: {len(pending_configs)}")

    batch_results = list(existing_results_by_key.values())

    config_count = 0
    start_time = time.time()

    for algorithm, alpha, time_slots in pending_configs:
        config_count += 1
        cfg_key = _config_key(algorithm, alpha, time_slots)
        run_folder = f"run_{len(completed_keys) + config_count:03d}_{algorithm}_a{alpha}_t{time_slots}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        run_dir = runs_dir / run_folder

        manifest['configs'][cfg_key] = {
            'status': 'running',
            'algorithm': algorithm,
            'alpha': alpha,
            'time_slots': time_slots,
            'run_dir': str(run_dir),
            'started_at': datetime.now().isoformat(),
        }
        manifest['updated_at'] = datetime.now().isoformat()
        _save_manifest(batch_archive_dir, manifest)

        print(f"\n⚙️ Configuration {len(completed_keys) + config_count}/{total_configs}")
        print(f"   🔄 α={alpha} | slots={time_slots} | {algorithm}")

        try:
            # 🎯 RUN YOUR EXISTING WORKING SIMULATION
            performance, sim_archive_dir = run_single_simulation(
                algorithm,
                alpha,
                time_slots,
                output_root=runs_dir,
                run_subdir=run_folder,
            )

            if performance:
                performance['config_id'] = len(completed_keys) + config_count
                performance['timestamp'] = datetime.now().isoformat()
                batch_results.append(performance)

                # Save each run's hit-ratio files to its own run archive directory.
                run_files = save_results_to_files([performance], sim_archive_dir)

                manifest['configs'][cfg_key].update({
                    'status': 'completed',
                    'finished_at': datetime.now().isoformat(),
                    'hit_json': run_files['json'],
                    'hit_csv': run_files['csv'],
                })

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
            remaining_runs = len(pending_configs) - config_count
            remaining = (elapsed / max(1, config_count)) * remaining_runs
            print(f"      ⏱️ Elapsed: {elapsed / 60:.1f}min | Est. remaining: {remaining / 60:.1f}min")

        except Exception as e:
            print(f"      ❌ Error: {e}")
            logger.exception(
                "Batch config failed: algorithm=%s alpha=%s slots=%s",
                algorithm,
                alpha,
                time_slots,
            )
            manifest['configs'][cfg_key].update({
                'status': 'failed',
                'finished_at': datetime.now().isoformat(),
                'error': str(e),
            })
            manifest['updated_at'] = datetime.now().isoformat()
            _save_manifest(batch_archive_dir, manifest)
            traceback.print_exc()
            continue

        manifest['updated_at'] = datetime.now().isoformat()
        _save_manifest(batch_archive_dir, manifest)

    # Save batch summary (all runs) to a separate batch summary directory.
    if batch_results:
        summary_files = save_batch_summary_files(batch_results, summary_dir)
        manifest['summary_files'] = summary_files

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

    success_count = sum(1 for c in manifest['configs'].values() if c.get('status') == 'completed')
    fail_count = sum(1 for c in manifest['configs'].values() if c.get('status') == 'failed')
    manifest['status'] = 'completed' if fail_count == 0 else 'completed_with_errors'
    manifest['updated_at'] = datetime.now().isoformat()
    manifest['final_stats'] = {
        'planned': total_configs,
        'completed': success_count,
        'failed': fail_count,
        'pending': max(0, total_configs - success_count - fail_count),
    }
    _save_manifest(batch_archive_dir, manifest)

    print("\n📦 Batch output layout:")
    print(f"   • Runs: {runs_dir}")
    print(f"   • Summary: {summary_dir}")
    print(f"   • Manifest: {_manifest_path(batch_archive_dir)}")

    return batch_results


def main():
    parser = argparse.ArgumentParser(description="HIT RATIO ANALYSIS")

    # Single simulation mode
    parser.add_argument('--algorithm', type=str,
                        choices=['LRU', 'Popularity', 'MAB_Original', 'MAB_Contextual', 'MAB_Contextual_EnergyAware',
                                 'Federated_MAB', 'Federated_MAB_EnergyAware', 'Enhanced_Federated_MAB',
                                 'Enhanced_Federated_MAB_EnergyAware'],
                        help='Algorithm for single simulation')
    parser.add_argument('--alpha', type=float, help='Zipf alpha parameter')
    parser.add_argument('--time_slots', type=int, help='Number of time slots')

    # Batch mode
    parser.add_argument('--batch', action='store_true',
                        help='Run batch analysis across all algorithms and parameters')
    parser.add_argument('--resume_batch_dir', type=str,
                        help='Resume an existing batch directory, e.g. results/batch_20260419_120000')

    args = parser.parse_args()

    print("🎯 HIT RATIO & CACHE RATIO ANALYSIS")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        if args.batch:
            # Batch mode - ALL ALGORITHMS, ALL ALPHA VALUES, ALL SIMULATION SLOTS
            run_batch_analysis(args.resume_batch_dir)

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