# communication.py - HIERARCHICAL ZIPF IMPLEMENTATION
# Updated to use the old hierarchical Zipf logic while keeping all new features
# 通信控制核心
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
    # 🔧 HIERARCHICAL ZIPF IMPLEMENTATION - RESTORED FROM OLD VERSION
    # =========================================================================

    def custom_zipf(self, alpha, s, n):
        """
        ✅ RESTORED: Original hierarchical Zipf function from old implementation
        This creates the strong concentration patterns needed for good cache performance
        """
        if alpha <= 0:
            alpha = 0.01

        if alpha > 1:
            # Use numpy's zipf for alpha > 1 (creates strong concentration)
            return (np.random.zipf(alpha) % n) + 1
        else:
            # For alpha <= 1, use power-law distribution
            xk = np.arange(s, n + 1, dtype=float)
            #xk = np.arange(1, n + 1, dtype=float)
            pk = xk ** (-alpha) / np.sum(xk ** (-alpha))
            x = np.random.choice(xk, p=pk)
            return int(round(x))

    def send_content_request(self, vehicle, vehicles, uavs, base_stations, satellites,
                             grid_size, current_time, slot, no_of_content_each_category,
                             current_zipf=None):
        """
        ✅ HIERARCHICAL ZIPF IMPLEMENTATION - RESTORED FROM OLD VERSION
        This creates the 3-level hierarchy that gave good performance in your paper
        """
        unique_id = random.randint(1, 999999)
        final_alpha = current_zipf if current_zipf is not None else self.alpha

        # Track usage
        self.zipf_analysis['alpha_used'].append(final_alpha)
        self.zipf_analysis['request_count'] += 1

        # 🎯 STEP 1: ENTITY SELECTION (Hierarchical Zipf Level 1)
        # Total entities: satellites + UAVs + grid cells
        total_satellites = len(satellites)
        total_uavs = len(uavs)
        total_grid_cells = grid_size * grid_size
        #all_entities = total_satellites + total_uavs + total_grid_cells
        start=total_satellites + total_uavs
        all_entities = total_grid_cells
        #all_entities = total_satellites + total_uavs
        #all_entities = total_satellites

        # Select entity using Zipf distribution
        element_index = self.custom_zipf(final_alpha, start, all_entities)

        # 🎯 DETERMINE ENTITY TYPE AND SPECIFIC ENTITY
        if 1 <= element_index <= total_satellites:
            # ==================== SATELLITE CONTENT ====================
            entity_type = 'satellite'
            satellite_id = element_index
            satellite_key = f"Satellite{satellite_id}"

            if satellite_key in satellites:
                satellite = satellites[satellite_key]

                # 🎯 STEP 2: CATEGORY SELECTION (Hierarchical Zipf Level 2)
                category_index = self.custom_zipf(final_alpha, 1, len(satellite.content_categories))
                content_category = satellite.content_categories[category_index - 1]

                # 🎯 STEP 3: CONTENT NUMBER SELECTION (Hierarchical Zipf Level 3)
                content_no = self.custom_zipf(final_alpha, 1, no_of_content_each_category)

                request = {
                    'unique_id': unique_id,
                    'requesting_vehicle': vehicle,
                    'g_time': current_time,
                    'hop_count': 0,
                    'time_spent': 0,
                    'alpha_used': final_alpha,
                    'vehicle_list': [],
                    'type': 'satellite',
                    'category': content_category,
                    'coord': satellite_id,  # 🔥 INTEGER: 1, 2, 3
                    'no': content_no,       # 🔥 INTEGER: 1, 2, 3, ...
                    'request_id': f"sat{satellite_id}_{content_category}_{content_no}_{unique_id}",
                    'RDB': self.get_time_delay('satellite'),
                    'request_start_time': time.time()
                }
                content_key = f"satellite_{satellite_id}_{content_category}_{content_no}"
            else:
                # Fallback if satellite not found
                content_key = f"satellite_{satellite_id}_unknown_1"
                request = self._create_fallback_request(unique_id, vehicle, current_time, final_alpha)

        elif element_index <= total_satellites + total_uavs:
            # ==================== UAV CONTENT ====================
            entity_type = 'UAV'
            uav_index = element_index - total_satellites - 1
            uav_id = uav_index + 1  # 🔥 INTEGER: 1, 2, 3, ...

            if 0 <= uav_index < len(uavs):
                uav = uavs[uav_index]

                # 🎯 STEP 2: CATEGORY SELECTION (Hierarchical Zipf Level 2)
                category_index = self.custom_zipf(final_alpha, 1, len(uav.content_categories))
                content_category = uav.content_categories[category_index - 1]

                # 🎯 STEP 3: CONTENT NUMBER SELECTION (Hierarchical Zipf Level 3)
                content_no = self.custom_zipf(final_alpha, 1, no_of_content_each_category)

                request = {
                    'unique_id': unique_id,
                    'requesting_vehicle': vehicle,
                    'g_time': current_time,
                    'hop_count': 0,
                    'time_spent': 0,
                    'alpha_used': final_alpha,
                    'vehicle_list': [],
                    'type': 'UAV',
                    'category': content_category,
                    'coord': uav_id,        # 🔥 INTEGER: 1, 2, 3, ...
                    'no': content_no,       # 🔥 INTEGER: 1, 2, 3, ..
                    'request_id': f"uav{uav.uav_id}_{content_category}_{content_no}_{unique_id}",
                    'RDB': self.get_time_delay('UAV'),
                    'request_start_time': time.time()
                }
                content_key = f"UAV_{uav.uav_id}_{content_category}_{content_no}"
            else:
                # Fallback if UAV index out of range
                content_key = f"UAV_unknown_{content_category}_{content_no}"
                request = self._create_fallback_request(unique_id, vehicle, current_time, final_alpha)

        else:
            # ==================== GRID CONTENT ====================
            entity_type = 'grid'
            grid_cell_index = element_index - total_satellites - total_uavs-1
            grid_cell_index = max(0, grid_cell_index)  # Ensure non-negative

            # 🎯 STEP 2: CATEGORY SELECTION (Hierarchical Zipf Level 2)
            # Use vehicle's content categories for grid content
            grid_categories = vehicle.content_categories if hasattr(vehicle, 'content_categories') else ['II', 'III',
                                                                                                         'IV']
            category_index = self.custom_zipf(final_alpha, 1, len(grid_categories))
            content_category = grid_categories[category_index - 1]

            # 🎯 STEP 3: CONTENT NUMBER SELECTION (Hierarchical Zipf Level 3)
            content_no = self.custom_zipf(final_alpha, 1, no_of_content_each_category)

            request = {
                'unique_id': unique_id,
                'requesting_vehicle': vehicle,
                'g_time': current_time,
                'hop_count': 0,
                'time_spent': 0,
                'alpha_used': final_alpha,
                'vehicle_list': [],
                'type': 'grid',
                'category': content_category,
                'coord': grid_cell_index,  # 🔥 INTEGER: 0, 1, 2, ...
                'no': content_no,          # 🔥 INTEGER: 1, 2, 3, ...
                'request_id': f"grid{grid_cell_index}_{content_category}_{content_no}_{unique_id}",
                'RDB': self.get_time_delay('grid'),
                'request_start_time': time.time()
            }
            content_key = f"grid_{grid_cell_index}_{content_category}_{content_no}"

        # Log for analysis
        self.zipf_analysis['content_requests'][content_key] += 1
        self.zipf_analysis['entity_requests'][entity_type] += 1
        self.zipf_analysis['category_requests'][content_category] += 1

        # Periodic analysis
        if self.zipf_analysis['request_count'] % 1000 == 0:
            self.analyze_zipf_effectiveness()

        #print(f"request{request}")
        return request



    # def send_content_request(self, vehicle, vehicles, uavs, base_stations, satellites,
    #                          grid_size, current_time, slot, no_of_content_each_category,
    #                          current_zipf=None):
    #
    #     """CLEAN WORKING VERSION: Single Zipf selection across ALL content"""
    #     unique_id = random.randint(1, 999999)
    #     final_alpha = current_zipf if current_zipf is not None else self.alpha
    #
    #     # Track usage
    #     self.zipf_analysis['alpha_used'].append(final_alpha)
    #     self.zipf_analysis['request_count'] += 1
    #
    #     # Calculate total content
    #     satellite_content_count = len(satellites) * 3 * no_of_content_each_category
    #     uav_content_count = len(uavs) * 3 * no_of_content_each_category
    #     grid_cells = grid_size * grid_size
    #     grid_content_count = grid_cells * 3 * no_of_content_each_category
    #     total_content_items = satellite_content_count + uav_content_count + grid_content_count
    #
    #     # Single Zipf selection
    #     content_id = self.custom_zipf(final_alpha, total_content_items)
    #
    #     # Map content_id to entity/category/number
    #     if content_id <= satellite_content_count:
    #         # SATELLITE CONTENT
    #         sat_id = ((content_id - 1) // (3 * no_of_content_each_category)) + 1
    #         within_sat = (content_id - 1) % (3 * no_of_content_each_category)
    #         cat_idx = within_sat // no_of_content_each_category
    #         content_no = (within_sat % no_of_content_each_category) + 1
    #
    #         satellite_categories = ['I', 'II', 'III']
    #         category = satellite_categories[cat_idx]
    #
    #         request = {
    #             'unique_id': unique_id,
    #             'requesting_vehicle': vehicle,
    #             'g_time': current_time,
    #             'hop_count': 0,
    #             'time_spent': 0,
    #             'alpha_used': final_alpha,
    #             'vehicle_list': [],
    #             'type': 'satellite',
    #             'category': category,
    #             'coord': sat_id,
    #             'no': content_no,
    #             'request_id': f"sat{sat_id}_{category}_{content_no}_{unique_id}",
    #             'RDB': self.get_time_delay('satellite'),
    #             # 🔧 ADD: Request timing
    #             'request_start_time': time.time()
    #         }
    #         content_key = f"satellite_{sat_id}_{category}_{content_no}"
    #
    #     elif content_id <= satellite_content_count + uav_content_count:
    #         # UAV CONTENT
    #         uav_content_id = content_id - satellite_content_count
    #         uav_idx = ((uav_content_id - 1) // (3 * no_of_content_each_category))
    #         within_uav = (uav_content_id - 1) % (3 * no_of_content_each_category)
    #         cat_idx = within_uav // no_of_content_each_category
    #         content_no = (within_uav % no_of_content_each_category) + 1
    #
    #         uav_categories = ['II', 'III', 'IV']
    #         category = uav_categories[cat_idx]
    #
    #         if uav_idx < len(uavs):
    #             uav_id = uavs[uav_idx].uav_id
    #         else:
    #             uav_id = f"UAV{uav_idx + 1}"
    #
    #         request = {
    #             'unique_id': unique_id,
    #             'requesting_vehicle': vehicle,
    #             'g_time': current_time,
    #             'hop_count': 0,
    #             'time_spent': 0,
    #             'alpha_used': final_alpha,
    #             'vehicle_list': [],
    #             'type': 'UAV',
    #             'category': category,
    #             'coord': uav_id,
    #             'no': content_no,
    #             'request_id': f"uav{uav_id}_{category}_{content_no}_{unique_id}",
    #             'RDB': self.get_time_delay('UAV'),
    #             # 🔧 ADD: Request timing
    #             'request_start_time': time.time()
    #         }
    #         content_key = f"UAV_{uav_id}_{category}_{content_no}"
    #
    #     else:
    #         # GRID CONTENT
    #         grid_content_id = content_id - satellite_content_count - uav_content_count
    #         grid_cell_idx = ((grid_content_id - 1) // (3 * no_of_content_each_category))
    #         within_grid = (grid_content_id - 1) % (3 * no_of_content_each_category)
    #         cat_idx = within_grid // no_of_content_each_category
    #         content_no = (within_grid % no_of_content_each_category) + 1
    #
    #         grid_categories = ['II', 'III', 'IV']
    #         category = grid_categories[cat_idx]
    #
    #         request = {
    #             'unique_id': unique_id,
    #             'requesting_vehicle': vehicle,
    #             'g_time': current_time,
    #             'hop_count': 0,
    #             'time_spent': 0,
    #             'alpha_used': final_alpha,
    #             'vehicle_list': [],
    #             'type': 'grid',
    #             'category': category,
    #             'coord': grid_cell_idx,
    #             'no': content_no,
    #             'request_id': f"grid{grid_cell_idx}_{category}_{content_no}_{unique_id}",
    #             'RDB': self.get_time_delay('grid'),
    #             # 🔧 ADD: Request timing
    #             'request_start_time': time.time()
    #         }
    #         content_key = f"grid_{grid_cell_idx}_{category}_{content_no}"
    #
    #     # Log for analysis
    #     self.zipf_analysis['content_requests'][content_key] += 1
    #
    #     # Periodic analysis
    #     if self.zipf_analysis['request_count'] % 1000 == 0:
    #         self.analyze_zipf_effectiveness()
    #
    #     return request


    def _create_fallback_request(self, unique_id, vehicle, current_time, alpha):
        """Create a fallback request when entity selection fails"""
        return {
            'unique_id': unique_id,
            'requesting_vehicle': vehicle,
            'g_time': current_time,
            'hop_count': 0,
            'time_spent': 0,
            'alpha_used': alpha,
            'vehicle_list': [],
            'type': 'grid',
            'category': 'II',
            'coord': 0,
            'no': 1,
            'request_id': f"fallback_{unique_id}",
            'RDB': self.get_time_delay('grid'),
            'request_start_time': time.time()
        }

    # =========================================================================
    # EXISTING METHODS - KEEP ALL YOUR CURRENT FUNCTIONALITY
    # =========================================================================

    def get_time_delay(self, entity_type):
        return self.content_time_delays.get(entity_type, 0)

    def get_valid_categories_for_entity(self, entity_type):
        if entity_type == 'satellite':
            return self.satellite_categories
        elif entity_type == 'UAV':
            return self.uav_categories
        elif entity_type == 'grid':
            return self.grid_categories
        else:
            return ['II', 'III', 'IV']

    def track_retrieval_delay(self, content_request, retrieval_source='unknown'):
        """🔧 FIXED: Proper delay tracking with consistent time calculation"""
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

        except Exception as e:
            print(f"⚠️ Warning: Could not track delay - {e}")

    def reset_request_timing(self):
        """🔧 Reset timing for new request"""
        self.content_received_time = 0
        self.request_start_time = time.time()

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

       # print(f"\n📊 ZIPF ANALYSIS (Requests: {self.zipf_analysis['request_count']})")
        #print(f"   Average α: {avg_alpha:.2f}")
       # print(f"   Content Concentration (Top 20%): {content_conc:.1%}")
       # print(f"   Most Popular Content: {content_pop:.1%}")
       # print(f"   Unique Content Requested: {len(self.zipf_analysis['content_requests'])}")

        #if avg_alpha <= 0.5 and content_conc > 0.4:
           # print(f"   ⚠️  WARNING: Low α ({avg_alpha:.2f}) should have lower concentration!")
        #elif avg_alpha >= 1.5 and content_conc < 0.5:
         #   print(f"   ⚠️  WARNING: High α ({avg_alpha:.2f}) should have higher concentration!")
       # else:
           # print(f"   ✅ Concentration matches alpha expectations")

    # =========================================================================
    # KEEP ALL YOUR EXISTING COMMUNICATION METHODS UNCHANGED
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

    def write_delay_file(self, element_type, requested_category):
        """Write delay file for delay_analyzer.py"""
        try:
            algorithm = getattr(self, 'current_algorithm', 'unknown')
            valid_combos = {
                'satellite': ['I', 'II', 'III'],
                'UAV': ['II', 'III', 'IV'],
                'grid': ['II', 'III', 'IV']
            }

            if element_type in valid_combos and requested_category in valid_combos[element_type]:
                filename = f'delay_{algorithm}_{element_type}_{requested_category}_{self.alpha}_{self.time_slots}_{self.simulation_round}.txt'
                delay_value = max(0.001, self.content_received_time)
                with open(filename, 'a') as f:
                    f.write(f"{delay_value:.6f}\n")
        except:
            pass

    def broadcast_request(self, requesting_vehicle, content_request, vehicles, uavs, satellites, base_stations,
                          grid_size, current_time, slot, communication_schedule, ground_station):
        """🔧 FIXED: Broadcast with proper delay tracking"""

        # ADD THIS SINGLE LINE
        self.last_content_request = content_request

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
        if element_type == "satellite" or element_type == "UAV":
            for uav in uavs:
                if uav.is_within_coverage(requesting_vehicle.current_location[0],
                                          requesting_vehicle.current_location[1]):
                    content_request['time_spent'] = random.uniform(0.007, 0.01)
                    content = uav.process_content_request(self, requesting_vehicle, content_request, vehicles, uavs,
                                                          satellites, slot, communication_schedule)
                    if content:
                        flag1 = 1
                        for v in vehicles:
                            if uav.is_within_coverage(v.current_location[0], v.current_location[1]):
                                v.update_action_space(content, slot, federated_update=False)

                        for bs in base_stations:
                            if uav.is_within_coverage(bs.current_location[0], bs.current_location[1]):
                                bs.update_action_space(content, slot, federated_update=False)

        if requesting_vehicle.process_content_request(self, content_request, slot, grid_size, flag1):
            flag1= 1
        else:
            content_request['hop_count'] += 1
            for vehicle in vehicles:
                content_request['time_spent'] = random.uniform(0.003, 0.03)
                if (vehicle != requesting_vehicle and vehicle.is_within_range(requesting_vehicle.current_location) and
                        vehicle.vehicle_id not in content_request['vehicle_list']):
                    if vehicle.process_content_request(self, content_request, slot, grid_size, flag1):
                        flag1 = 1
                        break

            for bs in base_stations:
                content_request['time_spent'] = random.uniform(0.01, 0.03)
                if bs.check_within_range(requesting_vehicle.current_location):
                    if bs.process_content_request(self, content_request, slot, grid_size, ground_station, flag1):
                        flag1 = 1
                        break

        if flag1 == 1:
            self.write_delay_file(element_type, requested_category)



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

    # ADD THESE METHODS TO YOUR Communication CLASS in communication.py
    # Add them just before the last method of your Communication class

    def track_detailed_delay(self, content_request, retrieval_time, source_type='cache'):
        """Track detailed delay by content type and category - SAFE ADDITION"""
        try:
            entity_type = content_request.get('type', 'unknown')
            category = content_request.get('category', 'unknown')

            # Normalize entity type
            content_type = entity_type if entity_type in ['satellite', 'UAV', 'grid'] else 'unknown'

            # Validate category for content type
            valid_categories = {
                'satellite': ['I', 'II', 'III'],
                'UAV': ['II', 'III', 'IV'],
                'grid': ['II', 'III', 'IV']
            }

            if content_type in valid_categories and category in valid_categories[content_type]:
                # Initialize delay tracking if not exists
                if not hasattr(self, 'detailed_delay_data'):
                    self.detailed_delay_data = {}

                # Create algorithm-specific tracking
                algorithm = getattr(self, 'current_algorithm', 'unknown')
                if algorithm not in self.detailed_delay_data:
                    self.detailed_delay_data[algorithm] = {
                        'satellite': {'I': [], 'II': [], 'III': []},
                        'UAV': {'II': [], 'III': [], 'IV': []},
                        'grid': {'II': [], 'III': [], 'IV': []}
                    }

                # Store the delay
                if retrieval_time > 0:
                    self.detailed_delay_data[algorithm][content_type][category].append(retrieval_time)

        except Exception as e:
            # Silent fail to avoid breaking simulation
            pass

    def save_algorithm_delay_files(self, algorithm_name):
        """Save delay files for specific algorithm - SAFE ADDITION"""
        try:
            if hasattr(self, 'detailed_delay_data') and algorithm_name in self.detailed_delay_data:
                for content_type in self.detailed_delay_data[algorithm_name]:
                    for category in self.detailed_delay_data[algorithm_name][content_type]:
                        delays = self.detailed_delay_data[algorithm_name][content_type][category]

                        if delays:
                            filename = f'delay_{algorithm_name}_{content_type}_{category}_{self.alpha}_{self.time_slots}_{self.simulation_round}.txt'

                            with open(filename, 'w') as f:
                                for delay in delays:
                                    f.write(f"{delay:.6f}\n")

                            print(f"✅ Saved {len(delays)} delays for {algorithm_name}-{content_type}-{category}")
        except Exception as e:
            # Silent fail to avoid breaking simulation
            print(f"⚠️ Could not save delay files: {e}")
