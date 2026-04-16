#!/usr/bin/env python3
"""
🎯 正确的批量分析 - 精确变量名称
=================================================
使用您的 analyze_performance() 方法中的精确变量名称：
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


# Import your modules
from communication import Communication
from federated_mab import FederatedAggregator
from enhanced_federated_mab import EnhancedFederatedAggregator
from vehicle_ccn import Vehicle
from uav_ccn import UAV
from satellite_ccn import Satellite
from bs_ccn import BaseStation
from gs_ccn import GroundStation

# 创建所有节点（卫星、无人机、车辆、基站、地面站、聚合器），建立邻居关系，并分配算法。
def create_network(algorithm_type):
    """使用指定算法创建网络组件"""
    print(f"🗏️ 为 {algorithm_type} 创建网络")

    # 创建聚合器
    if algorithm_type == "Enhanced_Federated_MAB":
        aggregator = EnhancedFederatedAggregator()
    else:
        aggregator = FederatedAggregator()  # Original unchanged
    #grid_size = 10
    #uav_grid_size = 20

    # 创建卫星 创建3科卫星 1-3
    satellites = {}
    for i in range(1, 4):
        satellite = Satellite(f"Satellite{i}")
        satellites[f"Satellite{i}"] = satellite

    # 创建地面站
    ground_station = GroundStation("GS1")

    # 创建无人机。网格大小 100×100，每个无人机覆盖 20×20 的网格，所以总共 25 架无人机
    # 为每架无人机分配一个 ID（UAV1～UAV25），并传入聚合器、算法等
    # --- choose sensible grid values (match your older setup) ---
    grid_size = 100  # 10 km region represented by 100x100 logical grid cells
    uav_grid_size = 20  # each UAV covers a 20x20 block -> (100//20)=5 per side

    # --- compute UAV grid dynamically like before ---
    uavs = []
    uav_side = grid_size // uav_grid_size  # e.g., 5
    uav_count = uav_side * uav_side  # e.g., 25

    # Create UAVs with IDs UAV1..UAV{uav_count}
    for i in range(uav_count):
        uav = UAV(f"UAV{i + 1}", grid_size, uav_grid_size, aggregator, algorithm=algorithm_type)
        uavs.append(uav)

    # 设置无人机的二维邻居
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


    # 创建车辆 50辆车
    vehicles = []
    for i in range(1, 51):
        vehicle = Vehicle(f"Vehicle{i}", grid_size, 20, 2, aggregator, algorithm=algorithm_type)
        vehicles.append(vehicle)

    # 均匀分布车辆的初始位置
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

    # 创建基站 3个基站 1-3
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


def analyze_performance_EXACT(vehicles, uavs, base_stations, algorithm, alpha, time_slots):
    """📊 使用您的 analyze_performance 方法中的精确变量名称"""

    print(f"📊 为 {algorithm} 分析性能 (α={alpha})")

    # ====================================================================
    # 来自您的 main.py analyze_performance 方法的精确变量名称
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
    # 来自您的代码的精确计算
    # ====================================================================

    # 整体性能
    total_cache_hits = vehicle_cache_hits + uav_cache_hits + bs_cache_hits
    total_source_hits = vehicle_source_hits + uav_source_hits + bs_source_hits + uav_sagin_hits
    total_content_hits = total_cache_hits + total_source_hits

    # 计算命中率 - 与您的代码完全相同的公式
    vehicle_hit_ratio = ((vehicle_cache_hits + vehicle_source_hits) / max(1, vehicle_total_requests)) * 100
    uav_hit_ratio = ((uav_cache_hits + uav_source_hits + uav_sagin_hits) / max(1, uav_total_requests)) * 100
    bs_hit_ratio = ((bs_cache_hits + bs_source_hits) / max(1, bs_total_requests)) * 100
    overall_hit_ratio = (total_content_hits / max(1,
                                                  vehicle_total_requests + uav_total_requests + bs_total_requests)) * 100

    # 缓存命中率 - 与您的代码完全相同的公式
    vehicle_cache_ratio = (vehicle_cache_hits / max(1, vehicle_cache_request)) * 100
    uav_cache_ratio = (uav_cache_hits / max(1, uav_cache_request)) * 100
    bs_cache_ratio = (bs_cache_hits / max(1, bs_cache_request)) * 100
    overall_cache_ratio = (total_cache_hits / max(1, total_content_hits)) * 100

    # 调试验证
    print(f"   🔍 原始数据验证:")
    print(
        f"   Vehicle: cache_hits={vehicle_cache_hits}, cache_requests={vehicle_cache_request}, source_hits={vehicle_source_hits}, total_requests={vehicle_total_requests}")
    print(
        f"   UAV: cache_hits={uav_cache_hits}, cache_requests={uav_cache_request}, source_hits={uav_source_hits}, sagin_hits={uav_sagin_hits}, total_requests={uav_total_requests}")
    print(
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

# 运行一次完整的仿真
def run_single_simulation(algorithm, alpha, time_slots, simulation_round=1):
    """🎯 您现有的工作模拟 - 未更改"""

    print(f"\n🚀 运行模拟")
    print(f"   算法: {algorithm}")
    print(f"   Alpha (Zipf): {alpha}")
    print(f"   时间槽: {time_slots}")
    print("=" * 50)

    # 创建网络 - 您现有的代码
    satellites, uavs, vehicles, base_stations, ground_station, aggregator = create_network(algorithm)

    # 创建通信 - 您现有的代码
    communication = Communication(satellites, base_stations, vehicles, uavs, ground_station, alpha, simulation_round,
                                  time_slots)


    # ✅ 添加此项（使用正确名称）:
    communication.current_algorithm = algorithm

    # 🔥 关键: 将 ALPHA 添加到所有节点（在此处插入此块！）
    print(f"🔧 将 alpha={alpha} 设置到所有节点...")

    # 将 alpha 传递给所有车辆
    for vehicle in vehicles:
        vehicle.current_alpha = alpha
        print(f"   🚗 {vehicle.vehicle_id}: alpha={alpha}")

    # 将 alpha 传递给所有 UAV
    for uav in uavs:
        uav.current_alpha = alpha
        print(f"   🚁 {uav.uav_id}: alpha={alpha}")

    # 将 alpha 传递给所有基站
    for bs in base_stations:
        bs.current_alpha = alpha
        print(f"   🏢 {bs.bs_id}: alpha={alpha}")


    # 🎯 YOUR EXISTING SIMULATION PARAMETERS - UNCHANGED
    grid_size = 100
    no_of_content_each_category = 3
    no_of_request_genertaed_in_each_timeslot = 3
    uav_content_generation_period = 5
    consecutive_slots_g = 5
    epsilon = 0.1

    # 生成通信调度表
    communication_schedule = {slot: [1, 2, 3] for slot in range(1, time_slots + 1)}

    # 🎯 YOUR EXISTING WORKING SIMULATION LOOP - COMPLETELY UNCHANGED
    for slot in range(1, time_slots + 1):
        current_time = (slot - 1) * 60

        # ✅ Satellite operations - YOUR EXISTING CODE 卫星每5个时间槽生成内容
        if (slot - 1) % consecutive_slots_g == 0:
            for satellite_id in satellites:
                satellites[satellite_id].run(satellites, communication, communication_schedule,
                                             current_time, slot, no_of_content_each_category, ground_station)
            ground_station.run(current_time, satellites)

        # ✅ UAV operations - YOUR EXISTING CODE 无人机每个时间槽运行，处理请求、更新Q值等
        for uav in uavs:
            uav.run(current_time, communication, communication_schedule, slot, satellites,
                    no_of_content_each_category, uav_content_generation_period, epsilon, uavs)
 
        # ✅ Vehicle operations - YOUR EXISTING CODE  车辆每个时间槽运行，处理请求、更新Q值等
        for vehicle in vehicles:
            vehicle.run(current_time, slot, time_slots, vehicles, uavs, base_stations, satellites,
                        communication, no_of_request_genertaed_in_each_timeslot,
                        no_of_content_each_category)

        # ✅ Base station operations - YOUR EXISTING CODE 基站每个时间槽运行，处理请求、更新Q值等
        for bs in base_stations:
            bs.run(current_time, slot)

        # ✅ FIX: Add Enhanced_Federated_MAB to condition
        #print(f"slottt{slot}, condition={(slot - 1) % 10 == 0}")
        # 联邦MAB聚合器每10个时间槽聚合一次Q值
        if algorithm in ["Federated_MAB", "Enhanced_Federated_MAB"] and slot > 10 and ((slot - 1) % 10 == 0):
            print(f"🔍 TRIGGERING aggregate_updates() for {algorithm} at slot {slot}")  # ADD THIS
            aggregator.aggregate_updates()
        else:
            if algorithm == "Enhanced_Federated_MAB":
                print(f"🔍 NOT TRIGGERING: algorithm={algorithm}, slot={slot}, condition={(slot - 1) % 10 == 0}")

                # ✅ Communication processing - YOUR EXISTING CODE
        # 处理通信调度表中的通信事件
        communication.run(vehicles, uavs, base_stations, satellites, grid_size, current_time,
                          slot, communication_schedule, time_slots, ground_station)



        # Progress indicator
        if slot % max(1, time_slots // 10) == 0:
            progress = (slot / time_slots) * 100
            print(f"   📊 Progress: {progress:.0f}% (slot {slot}/{time_slots})")

    # 🎯 ANALYZE WITH EXACT VARIABLES
    performance = analyze_performance_EXACT(vehicles, uavs, base_stations, algorithm, alpha, time_slots)

    return performance


def save_results_to_files(results):
    """Save results to JSON and CSV files with EXACT format you requested"""

    if not results:
        print("❌ No results to save!")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ====================================================================
    # JSON FILE - DETAILED RESULTS
    # ====================================================================
    json_filename = f"hit_ratio_analysis_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 DETAILED JSON: {json_filename}")

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
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_filename, index=False)

    print(f"📊 CSV FILE: {csv_filename}")

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

#  批量运行所有算法和 α 值的组合
def run_batch_analysis():
    """🎯 BATCH ANALYSIS FOR ALL ALGORITHMS AND ALPHA VALUES"""

    print(f"\n🎯 BATCH HIT RATIO ANALYSIS")
    print("=" * 60)

    # Configuration - YOUR REQUIREMENTS
    algorithms = ['LRU', 'Popularity', 'MAB_Original', 'MAB_Contextual', 'Enhanced_Federated_MAB']  # ✅ 5 algorithms
    #algorithms = ['LRU', 'Popularity', 'MAB_Original', 'Federated_MAB', 'MAB_Contextual']
    alpha_values = [0.25, 0.5, 1.0, 2.0]  # ✅ All alpha values
    time_slots_options = [300]  # ✅ All simulation slots

    total_configs = len(algorithms) * len(alpha_values) * len(time_slots_options)

    print(f"📊 Configuration:")
    print(f"   • Algorithms: {len(algorithms)} ({', '.join(algorithms)})")
    print(f"   • Alpha values: {len(alpha_values)} ({', '.join(map(str, alpha_values))})")
    print(f"   • Time slots: {len(time_slots_options)} ({', '.join(map(str, time_slots_options))})")
    print(f"   • Total configurations: {total_configs}")

    # Confirm execution
    response = input(f"\n⚠️ This will run {total_configs} configurations. Continue? (y/n): ")
    if response.lower() != 'y':
        print("❌ Batch analysis cancelled.")
        return None

    batch_results = []
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
                    performance = run_single_simulation(algorithm, alpha, time_slots)

                    if performance:
                        performance['config_id'] = config_count
                        performance['timestamp'] = datetime.now().isoformat()
                        batch_results.append(performance)

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

    # Save results to JSON and CSV
    if batch_results:
        save_results_to_files(batch_results)

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
            performance = run_single_simulation(args.algorithm, args.alpha, args.time_slots)
            if performance:
                # Save single result
                save_results_to_files([performance])

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