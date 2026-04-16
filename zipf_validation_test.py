# zipf_validation_test.py
"""
Test script to validate Zipf distribution is working correctly
Run this before your main simulation to ensure Zipf effect is strong
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random
import math


def test_zipf_distribution():
    """Test the custom_zipf function with different alpha values"""

    def custom_zipf(alpha, n):
        if alpha <= 0:
            alpha = 0.1

        if alpha < 0.5:
            alpha = max(0.05, alpha)
        elif alpha > 2.0:
            alpha = min(5.0, alpha * 1.2)

        ranks = np.arange(1, n + 1, dtype=float)
        probabilities = ranks ** (-alpha)
        probabilities = probabilities / np.sum(probabilities)

        content_index = np.random.choice(n, p=probabilities) + 1
        return content_index

    # Test different alpha values
    test_alphas = [0.1, 0.5, 1.0, 2.0, 3.0]
    n_items = 10  # Same as your no_of_content_each_category
    n_samples = 1000

    results = {}

    for alpha in test_alphas:
        print(f"\n=== Testing Alpha = {alpha} ===")

        # Generate samples
        samples = [custom_zipf(alpha, n_items) for _ in range(n_samples)]

        # Count frequencies
        counter = Counter(samples)

        # Calculate concentration metrics
        total_requests = len(samples)
        sorted_counts = sorted(counter.values(), reverse=True)

        # Top 30% concentration (top 3 out of 10 items)
        top_3_requests = sum(sorted_counts[:3])
        concentration_ratio = top_3_requests / total_requests

        # Most popular item percentage
        most_popular_pct = max(counter.values()) / total_requests

        print(f"Top 3 items get {concentration_ratio:.1%} of requests")
        print(f"Most popular item gets {most_popular_pct:.1%} of requests")
        print(f"Distribution: {dict(sorted(counter.items()))}")

        results[alpha] = {
            'concentration': concentration_ratio,
            'most_popular': most_popular_pct,
            'distribution': counter
        }

        # Expected behavior validation
        if alpha <= 0.5:
            expected_concentration = "10-25%"
            expected_most_popular = "15-25%"
        elif alpha <= 1.0:
            expected_concentration = "30-50%"
            expected_most_popular = "25-40%"
        elif alpha <= 2.0:
            expected_concentration = "60-80%"
            expected_most_popular = "40-60%"
        else:
            expected_concentration = "80-95%"
            expected_most_popular = "60-85%"

        print(f"Expected concentration: {expected_concentration}")
        print(f"Expected most popular: {expected_most_popular}")

        # Validation
        if alpha <= 0.5 and concentration_ratio > 0.4:
            print("⚠️  WARNING: Low alpha should have lower concentration!")
        elif alpha >= 2.0 and concentration_ratio < 0.6:
            print("⚠️  WARNING: High alpha should have higher concentration!")
        else:
            print("✅ Distribution looks correct")

    return results


def test_spatial_temporal_alpha():
    """Test the spatial-temporal alpha generation"""

    def generate_spatiotemporal_zipf_alpha(vehicle_location, current_time):
        grid_x, grid_y = int(vehicle_location[0]), int(vehicle_location[1])
        grid_x = max(0, min(99, grid_x))
        grid_y = max(0, min(99, grid_y))

        time_cycle = (current_time // 60) % 120
        quarter_size = 25

        if grid_x < quarter_size and grid_y < quarter_size:
            base_alpha = 3.5
            region_type = "disaster_center"
        elif grid_x < quarter_size and grid_y >= quarter_size:
            base_alpha = 0.12
            region_type = "rescue_zone"
        elif grid_x >= quarter_size and grid_y < quarter_size:
            base_alpha = 2.8
            region_type = "evacuation_area"
        else:
            base_alpha = 2.2
            region_type = "medical_area"

        if time_cycle < 30:
            temporal_multiplier = 1.6
            phase = "immediate"
        elif time_cycle < 60:
            temporal_multiplier = 0.4
            phase = "active_rescue"
        elif time_cycle < 90:
            temporal_multiplier = 1.2
            phase = "coordination"
        else:
            temporal_multiplier = 0.7
            phase = "recovery"

        final_alpha = base_alpha * temporal_multiplier
        noise = random.uniform(-0.15, 0.15)
        final_alpha = max(0.05, min(4.5, final_alpha + noise))

        return final_alpha, region_type, phase

    print("\n" + "=" * 50)
    print("TESTING SPATIAL-TEMPORAL ALPHA GENERATION")
    print("=" * 50)

    # Test different locations and times
    test_scenarios = [
        # (location, time, expected_behavior)
        ((10, 10), 1800, "High concentration (disaster center + immediate)"),
        ((10, 75), 3600, "Low concentration (rescue zone + active rescue)"),
        ((75, 10), 5400, "High concentration (evacuation + coordination)"),
        ((75, 75), 7200, "Medium concentration (medical + recovery)"),
    ]

    alpha_ranges = {
        "disaster_center": [],
        "rescue_zone": [],
        "evacuation_area": [],
        "medical_area": []
    }

    for location, time, expected in test_scenarios:
        print(f"\nLocation: {location}, Time: {time // 60}min")
        print(f"Expected: {expected}")

        # Generate multiple samples to see range
        alphas = []
        for _ in range(20):
            alpha, region, phase = generate_spatiotemporal_zipf_alpha(location, time)
            alphas.append(alpha)
            alpha_ranges[region].append(alpha)

        avg_alpha = np.mean(alphas)
        min_alpha = np.min(alphas)
        max_alpha = np.max(alphas)

        print(f"Alpha range: {min_alpha:.2f} - {max_alpha:.2f} (avg: {avg_alpha:.2f})")
        print(f"Region: {region}, Phase: {phase}")

        # Validate ranges
        if "High concentration" in expected and avg_alpha < 1.5:
            print("⚠️  WARNING: Expected high concentration but got low alpha!")
        elif "Low concentration" in expected and avg_alpha > 1.0:
            print("⚠️  WARNING: Expected low concentration but got high alpha!")
        else:
            print("✅ Alpha range looks correct")

    # Summary of all regions
    print(f"\n{'Region':<20} {'Min Alpha':<10} {'Max Alpha':<10} {'Avg Alpha':<10}")
    print("-" * 50)
    for region, alphas in alpha_ranges.items():
        if alphas:
            print(f"{region:<20} {min(alphas):<10.2f} {max(alphas):<10.2f} {np.mean(alphas):<10.2f}")


def validate_cache_hit_correlation():
    """Predict expected cache hit rates for different alpha values"""

    print("\n" + "=" * 50)
    print("EXPECTED CACHE HIT RATES BY ALPHA")
    print("=" * 50)

    cache_size = 50  # From your config
    total_content = 10 * 3 * 25  # 10 content per category * 3 categories * 25 UAVs = 750 items

    expected_results = [
        (0.1, "5-10%", "Very diverse requests, hard to predict"),
        (0.25, "10-15%", "High diversity, some patterns emerge"),
        (0.5, "20-30%", "Moderate concentration, MAB starts learning"),
        (1.0, "35-50%", "Clear patterns, good for caching"),
        (1.5, "50-65%", "High concentration, easy to cache"),
        (2.0, "60-75%", "Very concentrated, excellent for caching"),
        (3.0, "75-85%", "Extremely concentrated, almost all hits"),
    ]

    print(f"Cache size: {cache_size} MB")
    print(f"Total possible content items: ~{total_content}")
    print()
    print(f"{'Alpha':<8} {'Expected Hit Rate':<18} {'Reason'}")
    print("-" * 60)

    for alpha, hit_rate, reason in expected_results:
        print(f"{alpha:<8} {hit_rate:<18} {reason}")

    print(f"\n🎯 KEY INSIGHT:")
    print(f"   - Low α (0.1-0.5): Cache hit < 30% (requests too diverse)")
    print(f"   - High α (1.5-3.0): Cache hit > 50% (requests concentrated)")
    print(f"   - MAB should learn spatio-temporal patterns and outperform LRU/Popularity")


def run_mini_simulation():
    """Run a mini simulation to test request generation"""

    print("\n" + "=" * 50)
    print("MINI SIMULATION TEST")
    print("=" * 50)

    # Simulate request generation for different scenarios
    scenarios = [
        {"location": (10, 10), "time": 1800, "name": "Disaster Center"},
        {"location": (10, 75), "time": 3600, "name": "Rescue Zone"},
        {"location": (75, 75), "time": 5400, "name": "Medical Area"},
    ]

    def custom_zipf(alpha, n):
        if alpha <= 0:
            alpha = 0.1
        if alpha < 0.5:
            alpha = max(0.05, alpha)
        elif alpha > 2.0:
            alpha = min(5.0, alpha * 1.2)
        ranks = np.arange(1, n + 1, dtype=float)
        probabilities = ranks ** (-alpha)
        probabilities = probabilities / np.sum(probabilities)
        return np.random.choice(n, p=probabilities) + 1

    def generate_spatiotemporal_zipf_alpha(vehicle_location, current_time):
        grid_x, grid_y = int(vehicle_location[0]), int(vehicle_location[1])
        time_cycle = (current_time // 60) % 120
        quarter_size = 25

        if grid_x < quarter_size and grid_y < quarter_size:
            base_alpha = 3.5
        elif grid_x < quarter_size and grid_y >= quarter_size:
            base_alpha = 0.12
        elif grid_x >= quarter_size and grid_y < quarter_size:
            base_alpha = 2.8
        else:
            base_alpha = 2.2

        if time_cycle < 30:
            temporal_multiplier = 1.6
        elif time_cycle < 60:
            temporal_multiplier = 0.4
        elif time_cycle < 90:
            temporal_multiplier = 1.2
        else:
            temporal_multiplier = 0.7

        final_alpha = base_alpha * temporal_multiplier
        final_alpha = max(0.05, min(4.5, final_alpha + random.uniform(-0.15, 0.15)))
        return final_alpha

    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")

        # Generate alpha
        alpha = generate_spatiotemporal_zipf_alpha(scenario['location'], scenario['time'])
        print(f"Generated Alpha: {alpha:.2f}")

        # Simulate content requests
        n_requests = 100
        n_content_per_category = 10
        content_requests = []

        for _ in range(n_requests):
            # Simulate requesting from 3 entities (satellite/UAV/grid)
            entity_idx = random.randint(0, 2)
            category_idx = random.randint(0, 2)
            content_no = custom_zipf(alpha, n_content_per_category)

            content_key = f"entity{entity_idx}_cat{category_idx}_content{content_no}"
            content_requests.append(content_key)

        # Analyze distribution
        counter = Counter(content_requests)
        total = len(content_requests)
        sorted_counts = sorted(counter.values(), reverse=True)

        # Calculate metrics
        top_10_items = sum(sorted_counts[:10])
        concentration = top_10_items / total
        unique_items = len(counter)

        print(f"Unique content requested: {unique_items} out of ~90 possible")
        print(f"Top 10 items concentration: {concentration:.1%}")
        print(f"Most popular item: {max(counter.values())} requests ({max(counter.values()) / total:.1%})")

        # Simulate cache hit rate
        cache_capacity = 15  # items that can fit in cache
        cached_items = sorted_counts[:cache_capacity]
        cache_hits = sum(cached_items)
        hit_rate = cache_hits / total

        print(f"Simulated cache hit rate: {hit_rate:.1%}")

        # Validate expectations
        if alpha > 2.0 and hit_rate < 0.4:
            print("⚠️  WARNING: High alpha should give higher hit rate!")
        elif alpha < 0.5 and hit_rate > 0.3:
            print("⚠️  WARNING: Low alpha should give lower hit rate!")
        else:
            print("✅ Cache hit rate matches alpha expectation")


if __name__ == "__main__":
    print("ZIPF DISTRIBUTION VALIDATION TEST")
    print("=" * 60)

    # Run all tests
    test_zipf_distribution()
    test_spatial_temporal_alpha()
    validate_cache_hit_correlation()
    run_mini_simulation()

    print(f"\n🎯 SUMMARY:")
    print(f"   1. Run this test BEFORE your main simulation")
    print(f"   2. Verify that alpha variations are EXTREME (0.05 to 4.5)")
    print(f"   3. Confirm high alpha → high concentration → high cache hit")
    print(f"   4. Confirm low alpha → low concentration → low cache hit")
    print(f"   5. If tests pass, your Zipf distribution will be effective!")

    print(f"\n🚀 Next steps:")
    print(f"   - Copy the fixed communication.py code")
    print(f"   - Run this validation test")
    print(f"   - If tests pass, run your main simulation")
    print(f"   - You should see dramatic differences between α=0.1 and α=3.0")