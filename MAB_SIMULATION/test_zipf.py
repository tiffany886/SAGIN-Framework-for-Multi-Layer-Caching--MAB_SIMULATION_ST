#!/usr/bin/env python3
"""
COMPLETE ZIPF VERIFICATION TEST
Save this as 'test_zipf.py' and run it to verify your Zipf distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import random
import math


def test_your_current_zipf():
    """Test the EXACT Zipf implementation from your communication.py"""

    def custom_zipf(alpha, n):
        """YOUR CURRENT ZIPF IMPLEMENTATION"""
        if alpha <= 0:
            alpha = 0.01

        if alpha > 1:
            # Use numpy's zipf for alpha > 1
            return (np.random.zipf(alpha) % n) + 1
        else:
            # For alpha <= 1, use power-law distribution
            xk = np.arange(1, n + 1, dtype=float)
            pk = xk ** (-alpha)
            pk = pk / np.sum(pk)
            x = np.random.choice(xk, p=pk)
            return int(round(x))

    print("🔍 TESTING YOUR CURRENT ZIPF IMPLEMENTATION")
    print("=" * 60)

    # Test different alpha values
    test_alphas = [0.25, 0.5, 1.0, 2.0]
    n_content = 100
    n_samples = 10000

    for alpha in test_alphas:
        print(f"\n📊 Testing α = {alpha}")
        print("-" * 40)

        # Generate samples using YOUR current implementation
        samples = []
        for _ in range(n_samples):
            content_id = custom_zipf(alpha, n_content)
            samples.append(content_id)

        # Analyze distribution
        counter = Counter(samples)
        total_requests = len(samples)
        sorted_counts = sorted(counter.values(), reverse=True)

        # Calculate metrics
        top_20_percent_count = max(1, int(n_content * 0.2))  # Top 20 items
        top_20_requests = sum(sorted_counts[:top_20_percent_count])
        concentration_ratio = top_20_requests / total_requests

        most_popular_ratio = max(counter.values()) / total_requests
        unique_content = len(counter)

        # Expected values based on your paper
        if alpha <= 0.5:
            expected_concentration = 0.15  # 15%
            expected_popular = 0.08  # 8%
        elif alpha <= 1.0:
            expected_concentration = 0.35  # 35%
            expected_popular = 0.20  # 20%
        else:  # alpha >= 2.0
            expected_concentration = 0.70  # 70%
            expected_popular = 0.45  # 45%

        # Results
        print(f"   Top 20% concentration: {concentration_ratio:.1%}")
        print(f"   Expected concentration: {expected_concentration:.1%}")
        conc_status = "✅ PASS" if concentration_ratio >= expected_concentration * 0.8 else "❌ FAIL"
        print(f"   Concentration status: {conc_status}")

        print(f"   Most popular content: {most_popular_ratio:.1%}")
        print(f"   Expected most popular: {expected_popular:.1%}")
        pop_status = "✅ PASS" if most_popular_ratio >= expected_popular * 0.8 else "❌ FAIL"
        print(f"   Popularity status: {pop_status}")

        print(f"   Unique content accessed: {unique_content}/{n_content}")

        # Overall assessment
        if conc_status == "✅ PASS" and pop_status == "✅ PASS":
            print(f"   🎯 OVERALL: ✅ PASS - Zipf working for α={alpha}")
        else:
            print(f"   🎯 OVERALL: ❌ FAIL - Zipf NOT working for α={alpha}")
            print(f"      💡 FIX NEEDED: Your MAB can't learn without proper concentration!")


def test_hierarchical_zipf_simulation():
    """Test the 3-level hierarchical Zipf like in your actual simulation"""

    print("\n🏗️ TESTING HIERARCHICAL ZIPF (Your Full Simulation)")
    print("=" * 60)

    def custom_zipf(alpha, n):
        """Same as your implementation"""
        if alpha <= 0:
            alpha = 0.01
        if alpha > 1:
            return (np.random.zipf(alpha) % n) + 1
        else:
            xk = np.arange(1, n + 1, dtype=float)
            pk = xk ** (-alpha)
            pk = pk / np.sum(pk)
            x = np.random.choice(xk, p=pk)
            return int(round(x))

    # Your actual simulation parameters
    total_satellites = 3
    total_uavs = 25
    total_grid_cells = 100
    all_entities = total_satellites + total_uavs + total_grid_cells  # 128
    no_of_content_each_category = 5  # From your simulation

    alpha = 2.0  # High alpha should show strong concentration
    n_requests = 5000

    print(f"Simulating {n_requests} requests with α={alpha}")
    print(f"Total entities: {all_entities} (3 satellites + 25 UAVs + 100 grid)")
    print(f"Content per category: {no_of_content_each_category}")

    content_requests = defaultdict(int)
    entity_requests = defaultdict(int)

    for request_num in range(n_requests):
        # Level 1: Entity Selection (your hierarchical approach)
        element_index = custom_zipf(alpha, all_entities)

        # Determine entity type (matching your logic)
        if 1 <= element_index <= total_satellites:
            entity_type = 'satellite'
            entity_id = element_index
            categories = ['I', 'II', 'III']  # Satellite categories

        elif element_index <= total_satellites + total_uavs:
            entity_type = 'UAV'
            entity_id = element_index - total_satellites
            categories = ['II', 'III', 'IV']  # UAV categories

        else:
            entity_type = 'grid'
            entity_id = element_index - total_satellites - total_uavs
            categories = ['II', 'III', 'IV']  # Grid categories

        # Level 2: Category Selection
        category_index = custom_zipf(alpha, len(categories))
        content_category = categories[category_index - 1]

        # Level 3: Content Number Selection
        content_no = custom_zipf(alpha, no_of_content_each_category)

        # Track requests
        content_key = f"{entity_type}_{entity_id}_{content_category}_{content_no}"
        content_requests[content_key] += 1
        entity_requests[entity_type] += 1

    # Analyze results
    print(f"\n📈 HIERARCHICAL ZIPF RESULTS:")
    print("-" * 40)

    # Entity distribution
    total_entity_requests = sum(entity_requests.values())
    print("Entity Request Distribution:")
    for entity_type, count in sorted(entity_requests.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_entity_requests
        print(f"  {entity_type}: {count} ({percentage:.1%})")

    # Content concentration
    sorted_content_counts = sorted(content_requests.values(), reverse=True)
    total_content_requests = sum(sorted_content_counts)

    # Top 20% content concentration
    top_20_percent = max(1, int(len(sorted_content_counts) * 0.2))
    top_20_requests = sum(sorted_content_counts[:top_20_percent])
    concentration = top_20_requests / total_content_requests

    print(f"\nContent Concentration Analysis:")
    print(f"  Unique content requested: {len(content_requests)}")
    print(f"  Top 20% content gets: {concentration:.1%} of requests")
    print(
        f"  Most popular content: {max(sorted_content_counts)} requests ({max(sorted_content_counts) / total_content_requests:.1%})")

    # Assessment for α=2.0
    if concentration > 0.6:
        print(f"  🎯 ✅ EXCELLENT: Strong concentration for α={alpha}")
        print(f"     Your MAB should learn well with this pattern!")
    elif concentration > 0.4:
        print(f"  🎯 ⚠️  MODERATE: Some concentration for α={alpha}")
        print(f"     MAB learning will be slower but possible")
    else:
        print(f"  🎯 ❌ POOR: Low concentration for α={alpha}")
        print(f"     This explains why your MAB isn't learning!")
        print(f"     FIX: Check your Zipf implementation")

    return concentration


def create_quick_visualization():
    """Create a quick visualization to see the pattern"""

    def custom_zipf(alpha, n):
        if alpha <= 0:
            alpha = 0.01
        if alpha > 1:
            return (np.random.zipf(alpha) % n) + 1
        else:
            xk = np.arange(1, n + 1, dtype=float)
            pk = xk ** (-alpha)
            pk = pk / np.sum(pk)
            x = np.random.choice(xk, p=pk)
            return int(round(x))

    print("\n📊 CREATING ZIPF VISUALIZATION")
    print("=" * 40)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Zipf Distribution Test - Your Implementation', fontsize=16)

    test_alphas = [0.25, 0.5, 1.0, 2.0]
    n_content = 100
    n_samples = 5000

    for i, alpha in enumerate(test_alphas):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        # Generate samples
        samples = [custom_zipf(alpha, n_content) for _ in range(n_samples)]
        counter = Counter(samples)

        # Plot top 50 most popular items
        sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:50]
        content_ids, counts = zip(*sorted_items) if sorted_items else ([], [])

        ax.bar(range(len(content_ids)), counts, alpha=0.7)
        ax.set_title(f'α = {alpha}')
        ax.set_xlabel('Content Rank')
        ax.set_ylabel('Request Count')
        ax.grid(True, alpha=0.3)

        # Add concentration info
        if sorted_items:
            total = sum(counter.values())
            top_20_percent = max(1, int(len(counter) * 0.2))
            top_requests = sum(sorted(counter.values(), reverse=True)[:top_20_percent])
            concentration = top_requests / total
            ax.text(0.05, 0.95, f'Top 20%: {concentration:.1%}',
                    transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    plt.tight_layout()
    plt.savefig('zipf_verification_results.png', dpi=150, bbox_inches='tight')
    print("📁 Saved visualization as 'zipf_verification_results.png'")
    plt.show()


def main():
    """Run all Zipf tests"""
    print("🧪 ZIPF VERIFICATION TEST SUITE")
    print("=" * 60)
    print("This will test if your Zipf distribution is working correctly")
    print("If Zipf fails, your MAB cannot learn patterns!\n")

    # Test 1: Basic Zipf implementation
    test_your_current_zipf()

    # Test 2: Hierarchical Zipf (full simulation)
    concentration = test_hierarchical_zipf_simulation()

    # Test 3: Visualization
    create_quick_visualization()

    # Final assessment
    print("\n🎯 FINAL ASSESSMENT")
    print("=" * 40)

    if concentration > 0.6:
        print("✅ ZIPF WORKING CORRECTLY")
        print("   Your MAB should be able to learn with this concentration")
        print("   If MAB still not learning, check reward function and Q-value updates")
    elif concentration > 0.4:
        print("⚠️  ZIPF WORKING PARTIALLY")
        print("   Some concentration but not optimal")
        print("   MAB learning will be slower - consider tuning parameters")
    else:
        print("❌ ZIPF NOT WORKING")
        print("   This is why your MAB isn't learning!")
        print("   Fix Zipf distribution before proceeding")

    print(f"\n💡 NEXT STEPS:")
    if concentration > 0.5:
        print("   1. Zipf is working - focus on MAB reward function")
        print("   2. Add debug prints to track Q-value evolution")
        print("   3. Verify cache hit tracking is correct")
    else:
        print("   1. Fix Zipf implementation first!")
        print("   2. Ensure alpha > 1.0 creates strong concentration")
        print("   3. Re-test MAB after fixing Zipf")


if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)

    main()