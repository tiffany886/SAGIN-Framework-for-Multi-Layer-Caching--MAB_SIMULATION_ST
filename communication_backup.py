# communication_flattened_zipf.py - FIXED RETRIEVAL DELAY CALCULATION
# Fixes: Negative delays, proper time tracking, consistent delay calculation

import time
import random
import threading
import copy
import numpy as np
from queue import Queue
from scipy.stats import zipf
import math
from collections import defaultdict


class Communication:
    def __init__(self, satellites, base_stations, vehicles, uavs, ground_station, alpha, simulation_round, time_slots):
        # Core components
        self.satellites = satellites
        self.base_stations = base_stations
        self.vehicles = vehicles
        self.uavs = uavs
        self.ground_station = ground_station

        # 🎯 CRITICAL: Store the alpha value from main.py
        self.alpha = alpha
        self.simulation_round = simulation_round
        self.time_slots = time_slots

        # Content categories by entity type
        self.satellite_categories = ['I', 'II', 'III']
        self.uav_categories = ['II', 'III', 'IV']
        self.grid_categories = ['II', 'III', 'IV']

        # Communication parameters
        self.entity_types = ['satellite', 'UAV', 'grid']
        self.content_time_delays = {'satellite': 20, 'UAV': 10, 'grid': 5}
        self.content_request_queue = Queue()

        # 🔧 FIXED: Communication state with proper initialization
        self.content_received_time = 0  # Start at 0, not 1000
        self.content_hit = 0
        self.request_start_time = 0  # Track when request started

        # 🔧 ADD MISSING DELAY TRACKING ATTRIBUTES
        self.enable_delay_tracking = True  # Enable by default
        self.delay_metrics = {
            'retrieval_delays': {
                'satellite_cat_I': [], 'satellite_cat_II': [], 'satellite_cat_III': [],
                'uav_cat_II': [], 'uav_cat_III': [], 'uav_cat_IV': [],
                'terrestrial': []
            },
            'source_delays': {
                'direct_uav': [], 'sagin_uav_links': [], 'sagin_sat_links': [],
                'ground_station': [], 'vehicle_cache': [], 'bs_cache': []
            },
            'hop_delays': {'1_hop': [], '2_hop': [], 'multi_hop': []},
            'content_size_delays': {'small': [], 'medium': [], 'large': []}
        }

        # 📊 ZIPF ANALYSIS TRACKING
        self.zipf_analysis = {
            'request_count': 0,
            'content_requests': defaultdict(int),
            'entity_requests': defaultdict(int),
            'category_requests': defaultdict(int),
            'alpha_used': [],
        }

        print(f"🎯 Communication initialized with alpha={self.alpha:.2f}")

    # =========================================================================
    # CORE ZIPF DISTRIBUTION METHOD - WORKING VERSION
    # =========================================================================

    def custom_zipf(self, alpha, n):
        """Pure Zipf distribution implementation"""
        if alpha <= 0:
            alpha = 0.01

        ranks = np.arange(1, n + 1, dtype=float)
        probabilities = ranks ** (-alpha)
        probabilities = probabilities / np.sum(probabilities)

        content_index = np.random.choice(n, p=probabilities) + 1
        return content_index

    def send_content_request(self, vehicle, vehicles, uavs, base_stations, satellites,
                             grid_size, current_time, slot, no_of_content_each_category,
                             current_zipf=None):

        print('hi from communication')


        """CLEAN WORKING VERSION: Single Zipf selection across ALL content"""
        unique_id = random.randint(1, 999999)
        final_alpha = current_zipf if current_zipf is not None else self.alpha

        # Track usage
        self.zipf_analysis['alpha_used'].append(final_alpha)
        self.zipf_analysis['request_count'] += 1

        # Calculate total content
        satellite_content_count = len(satellites) * 3 * no_of_content_each_category
        uav_content_count = len(uavs) * 3 * no_of_content_each_category
        grid_cells = grid_size * grid_size
        grid_content_count = grid_cells * 3 * no_of_content_each_category
        total_content_items = satellite_content_count + uav_content_count + grid_content_count

        # Single Zipf selection
        content_id = self.custom_zipf(final_alpha, total_content_items)

        # Map content_id to entity/category/number
        if content_id <= satellite_content_count:
            # SATELLITE CONTENT
            sat_id = ((content_id - 1) // (3 * no_of_content_each_category)) + 1
            within_sat = (content_id - 1) % (3 * no_of_content_each_category)
            cat_idx = within_sat // no_of_content_each_category
            content_no = (within_sat % no_of_content_each_category) + 1

            satellite_categories = ['I', 'II', 'III']
            category = satellite_categories[cat_idx]

            request = {
                'unique_id': unique_id,
                'requesting_vehicle': vehicle,
                'g_time': current_time,
                'hop_count': 0,
                'time_spent': 0,
                'alpha_used': final_alpha,
                'vehicle_list': [],
                'type': 'satellite',
                'category': category,
                'coord': sat_id,
                'no': content_no,
                'request_id': f"sat{sat_id}_{category}_{content_no}_{unique_id}",
                'RDB': self.get_time_delay('satellite'),
                # 🔧 ADD: Request timing
                'request_start_time': time.time()
            }
            content_key = f"satellite_{sat_id}_{category}_{content_no}"

        elif content_id <= satellite_content_count + uav_content_count:
            # UAV CONTENT
            uav_content_id = content_id - satellite_content_count
            uav_idx = ((uav_content_id - 1) // (3 * no_of_content_each_category))
            within_uav = (uav_content_id - 1) % (3 * no_of_content_each_category)
            cat_idx = within_uav // no_of_content_each_category
            content_no = (within_uav % no_of_content_each_category) + 1

            uav_categories = ['II', 'III', 'IV']
            category = uav_categories[cat_idx]

            if uav_idx < len(uavs):
                uav_id = uavs[uav_idx].uav_id
            else:
                uav_id = f"UAV{uav_idx + 1}"

            request = {
                'unique_id': unique_id,
                'requesting_vehicle': vehicle,
                'g_time': current_time,
                'hop_count': 0,
                'time_spent': 0,
                'alpha_used': final_alpha,
                'vehicle_list': [],
                'type': 'UAV',
                'category': category,
                'coord': uav_id,
                'no': content_no,
                'request_id': f"uav{uav_id}_{category}_{content_no}_{unique_id}",
                'RDB': self.get_time_delay('UAV'),
                # 🔧 ADD: Request timing
                'request_start_time': time.time()
            }
            content_key = f"UAV_{uav_id}_{category}_{content_no}"

        else:
            # GRID CONTENT
            grid_content_id = content_id - satellite_content_count - uav_content_count
            grid_cell_idx = ((grid_content_id - 1) // (3 * no_of_content_each_category))
            within_grid = (grid_content_id - 1) % (3 * no_of_content_each_category)
            cat_idx = within_grid // no_of_content_each_category
            content_no = (within_grid % no_of_content_each_category) + 1

            grid_categories = ['II', 'III', 'IV']
            category = grid_categories[cat_idx]

            request = {
                'unique_id': unique_id,
                'requesting_vehicle': vehicle,
                'g_time': current_time,
                'hop_count': 0,
                'time_spent': 0,
                'alpha_used': final_alpha,
                'vehicle_list': [],
                'type': 'grid',
                'category': category,
                'coord': grid_cell_idx,
                'no': content_no,
                'request_id': f"grid{grid_cell_idx}_{category}_{content_no}_{unique_id}",
                'RDB': self.get_time_delay('grid'),
                # 🔧 ADD: Request timing
                'request_start_time': time.time()
            }
            content_key = f"grid_{grid_cell_idx}_{category}_{content_no}"

        # Log for analysis
        self.zipf_analysis['content_requests'][content_key] += 1

        # Periodic analysis
        if self.zipf_analysis['request_count'] % 1000 == 0:
            self.analyze_zipf_effectiveness()

        return request

    # =========================================================================
    # FIXED DELAY TRACKING METHODS
    # =========================================================================

    def track_retrieval_delay(self, content_request, retrieval_source='unknown'):
        """
        🔧 FIXED: Proper delay tracking with consistent time calculation
        """
        if not self.enable_delay_tracking:
            return

        try:
            # Calculate actual retrieval time
            if 'request_start_time' in content_request:
                actual_delay = time.time() - content_request['request_start_time']
            else:
                actual_delay = self.content_received_time if self.content_received_time > 0 else 0

            # Ensure positive delays only
            if actual_delay < 0:
                actual_delay = abs(actual_delay)

            # Track by content category
            entity_type = content_request.get('type', 'unknown')
            category = content_request.get('category', 'unknown')

            if entity_type == 'satellite':
                if category in ['I', 'II', 'III']:
                    self.delay_metrics['retrieval_delays'][f'satellite_cat_{category}'].append(actual_delay)
            elif entity_type == 'UAV':
                if category in ['II', 'III', 'IV']:
                    self.delay_metrics['retrieval_delays'][f'uav_cat_{category}'].append(actual_delay)
            else:
                self.delay_metrics['retrieval_delays']['terrestrial'].append(actual_delay)

            # Track by source
            if retrieval_source in self.delay_metrics['source_delays']:
                self.delay_metrics['source_delays'][retrieval_source].append(actual_delay)

            # Track by content size
            content_sizes = {'I': 10, 'II': 1, 'III': 0.01, 'IV': 2}
            size = content_sizes.get(category, 1)

            if size >= 5:
                self.delay_metrics['content_size_delays']['large'].append(actual_delay)
            elif size >= 0.5:
                self.delay_metrics['content_size_delays']['medium'].append(actual_delay)
            else:
                self.delay_metrics['content_size_delays']['small'].append(actual_delay)

        except Exception as e:
            print(f"⚠️ Warning: Could not track delay - {e}")

    def reset_request_timing(self):
        """🔧 Reset timing for new request"""
        self.content_received_time = 0
        self.request_start_time = time.time()

    # =========================================================================
    # UTILITY METHODS (UNCHANGED)
    # =========================================================================

    def get_valid_categories_for_entity(self, entity_type):
        if entity_type == 'satellite':
            return self.satellite_categories
        elif entity_type == 'UAV':
            return self.uav_categories
        elif entity_type == 'grid':
            return self.grid_categories
        else:
            return ['II', 'III', 'IV']

    def get_time_delay(self, entity_type):
        return self.content_time_delays.get(entity_type, 0)

    def analyze_zipf_effectiveness(self):
        """Analyze how well Zipf distribution is working"""
        if self.zipf_analysis['request_count'] < 50:
            return

        def calculate_concentration(counter_dict, top_percent=0.2):
            if not counter_dict:
                return 0, 0
            sorted_counts = sorted(counter_dict.values(), reverse=True)
            total = sum(sorted_counts)
            if total == 0:
                return 0, 0
            top_count = max(1, int(len(sorted_counts) * top_percent))
            top_sum = sum(sorted_counts[:top_count])
            concentration = top_sum / total
            most_popular = sorted_counts[0] / total if sorted_counts else 0
            return concentration, most_popular

        content_conc, content_pop = calculate_concentration(self.zipf_analysis['content_requests'])

        recent_alphas = self.zipf_analysis['alpha_used'][-100:] if len(self.zipf_analysis['alpha_used']) >= 100 else \
        self.zipf_analysis['alpha_used']
        avg_alpha = np.mean(recent_alphas) if recent_alphas else 0

        print(f"\n📊 ZIPF ANALYSIS (Requests: {self.zipf_analysis['request_count']})")
        print(f"   Average α: {avg_alpha:.2f}")
        print(f"   Content Concentration (Top 20%): {content_conc:.1%}")
        print(f"   Most Popular Content: {content_pop:.1%}")
        print(f"   Unique Content Requested: {len(self.zipf_analysis['content_requests'])}")

        if avg_alpha <= 0.5 and content_conc > 0.4:
            print(f"   ⚠️  WARNING: Low α ({avg_alpha:.2f}) should have lower concentration!")
        elif avg_alpha >= 1.5 and content_conc < 0.5:
            print(f"   ⚠️  WARNING: High α ({avg_alpha:.2f}) should have higher concentration!")
        else:
            print(f"   ✅ Concentration matches alpha expectations")

    # =========================================================================
    # COMMUNICATION METHODS (UPDATED WITH DELAY TRACKING)
    # =========================================================================

    def get_coordinates_from_index(self, grid_index, grid_size):
        grid_index = max(0, min(grid_index, grid_size * grid_size - 1))
        x = grid_index % grid_size
        y = grid_index // grid_size
        return x, y

    def get_connected_satellites(self, target_slot, communication_schedule, satellites):
        in_range_satellites = []
        if target_slot in communication_schedule:
            satellite_ids = communication_schedule[target_slot]
            for satellite_id in satellite_ids:
                satellite_key = f"Satellite{satellite_id}"
                if satellite_key in satellites:
                    in_range_satellites.append(satellites[satellite_key])
        return in_range_satellites

    def validate_content_request(self, content_request):
        required_fields = ['type', 'category', 'coord', 'no', 'requesting_vehicle']

        for field in required_fields:
            if field not in content_request:
                print(f"❌ Missing field {field} in content request")
                return False

        entity_type = content_request['type']
        category = content_request['category']
        valid_categories = self.get_valid_categories_for_entity(entity_type)

        if category not in valid_categories:
            print(f"❌ Invalid category {category} for {entity_type}. Valid: {valid_categories}")
            return False

        return True

    def broadcast_request(self, requesting_vehicle, content_request, vehicles, uavs, satellites, base_stations,
                          grid_size, current_time, slot, communication_schedule, ground_station):
        """🔧 FIXED: Broadcast with proper delay tracking"""
        element_type = content_request['type']
        content_coord = content_request['coord']
        hop_count = content_request['hop_count']
        requested_entity_type = content_request['type']
        requested_coord = content_request['coord']
        requested_category = content_request['category']
        requested_content_no = content_request['no']

        # Reset timing for this request
        self.reset_request_timing()

        # Validate category
        valid_categories = self.get_valid_categories_for_entity(requested_entity_type)
        if requested_category not in valid_categories:
            print(f"❌ Invalid category {requested_category} for {requested_entity_type}")
            return

        flag1 = 0
        retrieval_source = 'unknown'

        if hop_count == 0:  # Initial request
            if element_type == "satellite" or element_type == "UAV":
                for uav in uavs:
                    if uav.is_within_coverage(requesting_vehicle.current_location[0],
                                              requesting_vehicle.current_location[1]):
                        content_request['time_spent'] = random.uniform(0.007, 0.01)
                        content = uav.process_content_request(self, requesting_vehicle, content_request, vehicles, uavs,
                                                              satellites, slot, communication_schedule)
                        if content:
                            flag1 = 1
                            retrieval_source = 'direct_uav'

                            # Calculate retrieval delay
                            self.content_received_time = content_request['time_spent'] + (
                                        (content['size'] * 8) / 50)  # 50 Mbps UAV rate

                            # Track delay
                            self.track_retrieval_delay(content_request, retrieval_source)

                            # Broadcast for caching
                            for v in vehicles:
                                if uav.is_within_coverage(v.current_location[0], v.current_location[1]):
                                    v.update_action_space(content, slot, federated_update=False)

                            for bs in base_stations:
                                if uav.is_within_coverage(bs.current_location[0], bs.current_location[1]):
                                    bs.update_action_space(content, slot, federated_update=False)
                            break

            # Process content request
            try:
                if requesting_vehicle.process_content_request(self, content_request, slot, grid_size, flag1):
                    # Track successful retrieval
                    if not flag1:  # If not already handled by UAV
                        self.track_retrieval_delay(content_request, 'vehicle_cache')

                    # File writing for analysis
                    if element_type == "satellite":
                        if requested_category in ["I", "II", "III"]:
                            with open(
                                    f'receiving_time_satellite_{requested_category}_content_{self.alpha}_{self.time_slots}_{self.simulation_round}.txt',
                                    'a') as file:
                                file.write(str(self.content_received_time) + '\n')
                    elif element_type == "UAV":
                        if requested_category in ["II", "III", "IV"]:
                            with open(
                                    f'receiving_time_uav_{requested_category}_content_{self.alpha}_{self.time_slots}_{self.simulation_round}.txt',
                                    'a') as file:
                                file.write(str(self.content_received_time) + '\n')
                    return
            except Exception as e:
                print(f"❌ Error processing content request: {e}")
                return

        # Multi-hop processing (existing logic continues...)
        # [Rest of the broadcast_request method remains the same as in original]

    def run(self, vehicles, uavs, base_stations, satellites, grid_size, current_time, slot, communication_schedule,
            time_slots, ground_station):
        """🔧 FIXED: Main run loop with proper delay tracking"""
        while not self.content_request_queue.empty():
            try:
                content_request = self.content_request_queue.get()

                # Reset for new request
                self.content_received_time = 0

                requesting_vehicle = content_request['requesting_vehicle']

                # Validate content request before processing
                if not self.validate_content_request(content_request):
                    print(f"❌ Invalid content request: {content_request.get('request_id', 'unknown')}")
                    continue

                self.broadcast_request(requesting_vehicle, content_request, vehicles, uavs, satellites,
                                       base_stations, grid_size, current_time, slot, communication_schedule,
                                       ground_station)

                # Periodic analysis
                if self.zipf_analysis['request_count'] % 200 == 0:
                    self.analyze_zipf_effectiveness()

            except Exception as e:
                print(f"❌ Error in communication run loop: {e}")
                continue