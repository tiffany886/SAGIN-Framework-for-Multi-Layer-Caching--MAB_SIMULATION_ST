# uav_ccn.py

import random
import math
import time
import numpy as np


class UAV:
    def __init__(self, uav_id, grid_size, uav_grid_size, aggregator, algorithm="MAB_Contextual",
                 energy_lambda=0.0):
        # Existing initialization
        self.uav_id = uav_id
        self.grid_size = grid_size
        self.uav_grid_size = uav_grid_size
        self.content_categories = ['II', 'III', 'IV']
        self.content_size = {'I': 10, 'II': 1, 'III': 0.01, 'IV': 2}
        self.cache_size = 50
        self.coverage_area = self.calculate_coverage_area()
        self.transmission_rate = 50
        self.propagation_delay = 100
        self.content_cache = {}
        self.epsilon = 0.1
        self._period_seen_keys = set()

        # ✅ NEW: Algorithm selection
        self.algorithm = str(algorithm).strip()
        self.energy_lambda = energy_lambda
        self.max_energy_per_request = 100.0
        print(f"  📦 {self.uav_id} initialized with {self.algorithm} algorithm")

        # Action space and learning components
        self.num_actions = 0
        self.action_space = []
        self.generated_cache = {'II': [], 'III': [], 'IV': []}
        self.record = {}

        # Performance tracking
        self.num_context_parameters = 3
        self.request_receive_count = 0
        self.request_receive_from_other_source = 0
        self.global_request_receive_count = 0
        self.content_hit_from_source = 0
        self.content_hit_from_indirect_source = 0
        self.current_slot = None
        self.current_slot_content_requests = []
        self.content_hit = 0
        self.optimal_content_hit = 0
        self.avg_decision_latency = 0
        self.decision_count = 0
        self.aggregator = aggregator

        # Initialize neighbors list
        self.neighbors = []

        # ✅ ADD MISSING PERFORMANCE TRACKING ATTRIBUTES
        self.content_hit_for_direct_uav = 0
        self.content_hit_for_sagin_links = 0
        self.content_hit_source_uav_1st_hop = 0
        self.content_hit_source_uav_2nd_hop = 0
        self.content_hit_source_sat_2nd_hop = 0
        self.content_hit_cache_1st_hop = 0
        self.content_hit_cache_2nd_hop = 0
        self.total_request_for_caching = 0
        self.total_request_received = 0
        self.initialize_q_values_properly()


    def initialize_q_values_properly(self):
        """Initialize Q-values to 0, not 0.5"""

        for action in self.action_space:
            # Start with Q-value = 0, not 0.5
            action['q_value'] = 0.0

            content = action['content']
            s_type = content['content_type']
            s_coord = content['content_coord']
            s_category = content['content_category']
            s_no = content['content_no']

            # Initialize record structure
            if s_type not in self.record:
                self.record[s_type] = {}
            if s_coord not in self.record[s_type]:
                self.record[s_type][s_coord] = {}
            if s_category not in self.record[s_type][s_coord]:
                self.record[s_type][s_coord][s_category] = {}
            if s_no not in self.record[s_type][s_coord][s_category]:
                self.record[s_type][s_coord][s_category][s_no] = {
                    'q_value': 0.0,  # Start at 0
                    'avg_reward': 0.0,  # Start at 0
                    'content_hit': 0,  # Hit counter
                    'no_f_time_cached': 0,  # Selection counter
                    'total_requests': 0,  # Request counter
                    'energy_sum': 0.0,
                    'energy_count': 0,
                }

    def _ensure_record_structure(self, entity_type, coord, category, content_no):
        if entity_type not in self.record:
            self.record[entity_type] = {}
        if coord not in self.record[entity_type]:
            self.record[entity_type][coord] = {}
        if category not in self.record[entity_type][coord]:
            self.record[entity_type][coord][category] = {}
        if content_no not in self.record[entity_type][coord][category]:
            self.record[entity_type][coord][category][content_no] = {
                'weighted_request_tracking': 0,
                'request_tracking': 0,
                'content_hit': 0,
                'q_value': 0.0,
                'avg_reward': 0.0,
                'no_f_time_cached': 0,
                'slot': 0,
                'cache_path_reqs': 0,
                'cache_hit_delay_sum': 0.0,
                'cache_hit_delay_count': 0,
                'energy_sum': 0.0,
                'energy_count': 0,
            }

        entry = self.record[entity_type][coord][category][content_no]
        entry.setdefault('cache_path_reqs', 0)
        entry.setdefault('cache_hit_delay_sum', 0.0)
        entry.setdefault('cache_hit_delay_count', 0)
        entry.setdefault('energy_sum', 0.0)
        entry.setdefault('energy_count', 0)

    def _capture_energy_metrics(self, communication, content_request, retrieval_source):
        energy_joule, normalized_energy, effective_hops = communication.compute_retrieval_energy(
            content_request, retrieval_source
        )
        content_request['retrieval_source'] = retrieval_source
        content_request['energy_joule'] = energy_joule
        content_request['energy_cost_joule'] = energy_joule
        content_request['normalized_energy'] = normalized_energy
        content_request['effective_hops'] = effective_hops
        return energy_joule, normalized_energy, effective_hops

    def cache_cleanup(self, current_time):
        """
        UNIFIED cache cleanup - works with new structure
        """

        if not hasattr(self, 'content_cache'):
            return

        # Iterate through the 3-level structure: type -> coord -> category -> [contents]
        for content_type in list(self.content_cache.keys()):
            for coord in list(self.content_cache[content_type].keys()):
                for category in list(self.content_cache[content_type][coord].keys()):
                    # Now we have the list of contents
                    cached_contents = self.content_cache[content_type][coord][category]

                    # Filter out expired content
                    valid_contents = []
                    for content in cached_contents:
                        if isinstance(content, dict):  # Safety check
                            generation_time = content.get('generation_time', 0)
                            validity = content.get('content_validity', float('inf'))

                            if current_time < generation_time + validity:
                                valid_contents.append(content)

                    # Update the cache with only valid content
                    self.content_cache[content_type][coord][category] = valid_contents

                    # Clean up empty categories
                    if not valid_contents:
                        del self.content_cache[content_type][coord][category]

                # Clean up empty coordinates
                if coord in self.content_cache[content_type] and not self.content_cache[content_type][coord]:
                    del self.content_cache[content_type][coord]

            # Clean up empty types
            if content_type in self.content_cache and not self.content_cache[content_type]:
                del self.content_cache[content_type]

    def debug_reward_processing_detailed(self, slot):
        """Add this method to UAV class to check reward processing"""

        if slot % 100 == 0:
            print(f"\n🎁 DETAILED REWARD PROCESSING DEBUG - {self.uav_id} - Slot {slot}")
            print("=" * 70)

            total_hits = 0
            total_rewards_calculated = 0
            q_value_updates = 0

            for entity_type, coord_dict in self.record.items():
                for coord, category_dict in coord_dict.items():
                    for category, content_no_dict in category_dict.items():
                        for content_no, content_info in content_no_dict.items():

                            content_key = f"{entity_type}_{coord}_{category}_{content_no}"

                            # Check if this content is in cache
                            in_cache = self.is_content_in_cache(coord, category, content_no)

                            # Get all metrics
                            hits = content_info.get('content_hit', 0)
                            total_requests = content_info.get('total_requests', 0)
                            times_cached = content_info.get('no_f_time_cached', 0)
                            avg_reward = content_info.get('avg_reward', 0)
                            q_value = content_info.get('q_value', 0)

                            if hits > 0 or in_cache or times_cached > 0:
                                print(f"   📊 {content_key}:")
                                print(f"      In cache: {in_cache}")
                                print(f"      Hits: {hits}, Total requests: {total_requests}")
                                print(f"      Times cached: {times_cached}")
                                print(f"      Avg reward: {avg_reward:.4f}, Q-value: {q_value:.4f}")

                                total_hits += hits

                                if hits > 0:
                                    total_rewards_calculated += 1

                                if q_value != 0.5:  # Default Q-value is 0.5
                                    q_value_updates += 1

            print(f"\n📈 SUMMARY:")
            print(f"   Total cache hits: {total_hits}")
            print(f"   Content with rewards: {total_rewards_calculated}")
            print(f"   Q-values updated: {q_value_updates}")

            if total_hits > 0 and total_rewards_calculated == 0:
                print("   ❌ CRITICAL: Cache hits but no rewards calculated!")
                print("   🔧 Check get_reward() method is being called")
            elif total_rewards_calculated > 0 and q_value_updates == 0:
                print("   ❌ CRITICAL: Rewards calculated but Q-values not updated!")
                print("   🔧 Check Q-value update logic in get_reward()")
            elif total_hits == 0:
                print("   ❌ Still no cache hits - check content matching")
            else:
                print("   ✅ Reward processing pipeline working!")

    def sigmoid(self, x):
        """
        ✅ ROBUST SIGMOID - HANDLES EXTREME VALUES
        Replace the CONTENT of your existing sigmoid method with this
        """
        try:
            if x > 500:  # Prevent overflow
                return 1.0
            elif x < -500:  # Prevent underflow
                return 0.0
            else:
                return 1 / (1 + math.exp(-x))
        except (OverflowError, ZeroDivisionError):
            # Handle any remaining edge cases
            return 0.5  # Return neutral value


    def calculate_coverage_area(self):
        """
        ✅ ROBUST COVERAGE CALCULATION - WORKS WITH ANY GRID CONFIGURATION
        Handles all cases: grid_size < uav_grid_size, grid_size == uav_grid_size, etc.

        REPLACE YOUR EXISTING calculate_coverage_area METHOD WITH THIS
        """
        try:
            # Extract UAV index from ID (UAV1 -> 0, UAV2 -> 1, etc.)
            uav_index = int(self.uav_id.replace("UAV", "")) - 1  # Convert to 0-based

            # Ensure index is non-negative
            uav_index = max(0, uav_index)

            # 🔧 CASE 1: UAV grid size >= grid size (UAVs cover overlapping or full areas)
            if self.uav_grid_size >= self.grid_size:
                # Each UAV covers a portion of the total grid
                # Distribute UAVs evenly across the available space
                x1 = (uav_index * (max(1, self.grid_size // 4))) % self.grid_size
                y1 = ((uav_index // 4) * (max(1, self.grid_size // 4))) % self.grid_size
                x2 = min(x1 + min(self.uav_grid_size, self.grid_size), self.grid_size)
                y2 = min(y1 + min(self.uav_grid_size, self.grid_size), self.grid_size)

            # 🔧 CASE 2: Normal grid layout (uav_grid_size < grid_size)
            else:
                # Calculate how many UAVs fit per row/column
                uavs_per_row = max(1, self.grid_size // self.uav_grid_size)

                # Calculate UAV position in grid
                uav_row = uav_index // uavs_per_row
                uav_col = uav_index % uavs_per_row

                # Calculate coverage bounds
                x1 = uav_col * self.uav_grid_size
                x2 = min(x1 + self.uav_grid_size, self.grid_size)
                y1 = uav_row * self.uav_grid_size
                y2 = min(y1 + self.uav_grid_size, self.grid_size)

            # 🔧 BOUNDS CHECKING - Ensure coordinates are within grid
            x1 = max(0, min(x1, self.grid_size - 1))
            x2 = max(x1 + 1, min(x2, self.grid_size))
            y1 = max(0, min(y1, self.grid_size - 1))
            y2 = max(y1 + 1, min(y2, self.grid_size))

            coverage_area = ((x1, y1), (x2, y2))

            # Debug output.txt for troubleshooting
            print(
                f"  📍 {self.uav_id} coverage: ({x1},{y1}) to ({x2},{y2}) [grid={self.grid_size}, uav_grid={self.uav_grid_size}]")

            return coverage_area

        except Exception as e:
            print(f"⚠️  {self.uav_id} coverage calculation failed: {e}")
            # Fallback: Give each UAV a small area starting from origin
            fallback_size = min(5, max(1, self.grid_size // 2))
            x1 = (uav_index * 2) % max(1, (self.grid_size - fallback_size))
            y1 = ((uav_index // 2) * 2) % max(1, (self.grid_size - fallback_size))
            x2 = min(x1 + fallback_size, self.grid_size)
            y2 = min(y1 + fallback_size, self.grid_size)

            print(f"  🆘 {self.uav_id} using fallback coverage: ({x1},{y1}) to ({x2},{y2})")
            return ((x1, y1), (x2, y2))

    def add_neighbor(self, neighbor_uav):
        self.neighbors.append(neighbor_uav)

    # Key fixes needed in uav_ccn.py for the process_content_request method

    def process_content_request(self, communication, requesting_vehicle, content_request, vehicles, uavs, satellites,
                                slot, communication_schedule):

        #print("hi-uav1", content_request)

        requested_entity_type = content_request['type']
        requested_coord = content_request['coord']
        requested_category = content_request['category']
        requested_content_no = content_request['no']

        # ensure record structure exists
        if requested_entity_type not in self.record:
            self.record[requested_entity_type] = {}
        if requested_coord not in self.record[requested_entity_type]:
            self.record[requested_entity_type][requested_coord] = {}
        if requested_category not in self.record[requested_entity_type][requested_coord]:
            self.record[requested_entity_type][requested_coord][requested_category] = {}
        if requested_content_no not in self.record[requested_entity_type][requested_coord][requested_category]:
            self.record[requested_entity_type][requested_coord][requested_category][requested_content_no] = {
                'weighted_request_tracking': 0,
                'request_tracking': 0,
                'content_hit': 0,
                'q_value': 0.0,
                'avg_reward': 0.0,
                'no_f_time_cached': 0,
                'slot': 0,
                'cache_path_reqs': 0,  # NEW (attempts of cache path)
                'cache_hit_delay_sum': 0.0,
                'cache_hit_delay_count': 0,
                'energy_sum': 0.0,
                'energy_count': 0,
            }

        entry = self.record[requested_entity_type][requested_coord][requested_category][requested_content_no]
        entry['request_tracking'] += 1  # all requests
        self.global_request_receive_count += 1
        self.total_requests_tracked = getattr(self, 'total_requests_tracked', 0) + 1

        # ---------------- UAV-generated content (this UAV) ----------------
        if requested_entity_type == "UAV":
            if content_request['coord'] == int(self.uav_id.replace('UAV', '')):
                if requested_category in self.generated_cache:
                    category_cache = self.generated_cache[requested_category]
                    for content in category_cache:
                        if content['content_no'] == requested_content_no:
                            seg_delay_s = ((content['size'] * 8) / self.transmission_rate) + random.uniform(0.007, 0.01)
                            communication.content_received_time = content_request['time_spent'] + seg_delay_s

                            self.content_hit_for_direct_uav += 1
                            self.content_hit_source_uav_1st_hop += 1
                            use_federated = self.algorithm in [
                                "Federated_MAB",
                                "Federated_MAB_EnergyAware",
                                "Enhanced_Federated_MAB",
                                "Enhanced_Federated_MAB_EnergyAware",
                            ]
                            requesting_vehicle.update_action_space(content, slot, federated_update=use_federated)

                            # generated path: DO NOT count as cache-path attempt
                            energy_joule, _, _ = self._capture_energy_metrics(communication, content_request, 'uav')
                            self.track_content_hit(communication, content_request, content, slot,
                                                   observed_delay_ms=seg_delay_s * 1000.0,
                                                   from_cache=False, energy_joule=energy_joule)
                            return content
                else:
                    print(
                        f"❌ Category {requested_category} not found in generated_cache: {list(self.generated_cache.keys())}")
            else:
                for neighbor in self.neighbors:
                    if requested_coord == int(neighbor.uav_id.replace('UAV', '')):
                        if requested_category in neighbor.generated_cache:
                            category_cache = neighbor.generated_cache[requested_category]
                            for content in category_cache:
                                if content['content_no'] == requested_content_no:
                                    seg_delay_s = 2 * (
                                                (content['size'] * 8) / self.transmission_rate) + 2 * random.uniform(
                                        0.007, 0.01)
                                    communication.content_received_time = content_request['time_spent'] + seg_delay_s

                                    self.content_hit_for_sagin_links += 1
                                    self.content_hit_source_uav_2nd_hop += 1
                                    use_federated = self.algorithm in [
                                        "Federated_MAB",
                                        "Federated_MAB_EnergyAware",
                                        "Enhanced_Federated_MAB",
                                        "Enhanced_Federated_MAB_EnergyAware",
                                    ]
                                    requesting_vehicle.update_action_space(content, slot,
                                                                           federated_update=use_federated)

                                    # still generated path: DO NOT count as cache-path attempt
                                    energy_joule, _, _ = self._capture_energy_metrics(communication, content_request, 'uav')
                                    self.track_content_hit(communication, content_request, content, slot,
                                                           observed_delay_ms=seg_delay_s * 1000.0,
                                                           from_cache=False, energy_joule=energy_joule)
                                    return content

                # queue to fetch later if not found immediately (not a cache-path attempt)
                # (removed) was incorrectly used as cache requests; now counted at cache-path attempt
                self.current_slot_content_requests.append({
                    'type': requested_entity_type,
                    'coord': requested_coord,
                    'category': requested_category,
                    'content_no': requested_content_no,
                })

        # ------------- Cache path (local 1-hop and neighbor 2-hop) -------------
        # IMPORTANT: we attempt cache path for types where cache makes sense; count attempt ONCE

        cache_path_attempted = False

        if requested_entity_type in ("UAV", "satellite"):

            # mark one cache-path attempt before searching local/neighbor caches
            entry['cache_path_reqs'] += 1  # NEW/CHANGED: count attempt (success or failure)
            self.request_receive_from_other_source += 1  # count cache-path attempt
            cache_path_attempted = True

            # local cache 1-hop
            if requested_entity_type in self.content_cache:
                if requested_coord in self.content_cache[requested_entity_type]:
                    if requested_category in self.content_cache[requested_entity_type][requested_coord]:
                        category_cache = self.content_cache[requested_entity_type][requested_coord][requested_category]
                        for content in category_cache:
                            if content['content_no'] == requested_content_no:
                                seg_delay_s = ((content['size'] * 8) / self.transmission_rate) + random.uniform(0.007,
                                                                                                                0.01)
                                if communication.content_received_time > content_request['time_spent'] + seg_delay_s:
                                    communication.content_received_time = content_request['time_spent'] + seg_delay_s

                                # cache hit → count hit + delay for cached-only reward
                                energy_joule, _, _ = self._capture_energy_metrics(communication, content_request, 'uav')
                                self.track_content_hit(communication, content_request, content, slot,
                                                       observed_delay_ms=seg_delay_s * 1000.0,
                                                       from_cache=True, energy_joule=energy_joule)
                                communication.content_hit += 1
                                self.content_hit += 1
                                self.content_hit_cache_1st_hop += 1

                                use_federated = self.algorithm in [
                                    "Federated_MAB",
                                    "Federated_MAB_EnergyAware",
                                    "Enhanced_Federated_MAB",
                                    "Enhanced_Federated_MAB_EnergyAware",
                                ]
                                requesting_vehicle.update_action_space(content, slot, federated_update=use_federated)
                                return content

            # neighbor cache 2-hop (same attempt; do not increment again)
            for neighbor in self.neighbors:
                if requested_entity_type in getattr(neighbor, 'content_cache', {}):
                    if requested_coord in neighbor.content_cache[requested_entity_type]:
                        if requested_category in neighbor.content_cache[requested_entity_type][requested_coord]:
                            category_cache = neighbor.content_cache[requested_entity_type][requested_coord][
                                requested_category]
                            for content in category_cache:
                                if content['content_no'] == requested_content_no:
                                    seg_delay_s = 2 * (
                                                (content['size'] * 8) / self.transmission_rate) + 2 * random.uniform(
                                        0.007, 0.01)
                                    if communication.content_received_time > content_request[
                                        'time_spent'] + seg_delay_s:
                                        communication.content_received_time = content_request[
                                                                                  'time_spent'] + seg_delay_s

                                    energy_joule, _, _ = self._capture_energy_metrics(communication, content_request,
                                                                                     'uav')
                                    self.track_content_hit(communication, content_request, content, slot,
                                                           observed_delay_ms=seg_delay_s * 1000.0,
                                                           from_cache=True, energy_joule=energy_joule)
                                    communication.content_hit += 1
                                    self.content_hit += 1
                                    self.content_hit_cache_2nd_hop += 1
                                    use_federated = self.algorithm in [
                                        "Federated_MAB",
                                        "Federated_MAB_EnergyAware",
                                        "Enhanced_Federated_MAB",
                                        "Enhanced_Federated_MAB_EnergyAware",
                                    ]
                                    requesting_vehicle.update_action_space(content, slot,
                                                                           federated_update=use_federated)
                                    return content

        # ---------------- Satellite path (not cached) ----------------
        if requested_entity_type == "satellite":
            connected_satellites = communication.get_connected_satellites(slot, communication_schedule, satellites)
            for satellite in connected_satellites:
                if int(satellite.satellite_id.replace('Satellite', '')) == requested_coord:
                    if requested_category in satellite.generated_cache:
                        cat_cache = satellite.generated_cache[requested_category]
                        for content in cat_cache:
                            if content['content_no'] == requested_content_no:
                                seg_delay_s = 2 * ((content['size'] * 8) / self.transmission_rate) + 2 * random.uniform(
                                    0.007, 0.01)
                                communication.content_received_time = content_request['time_spent'] + seg_delay_s
                                self.content_hit_source_sat_2nd_hop += 1

                                # satellite → not cached path
                                energy_joule, _, _ = self._capture_energy_metrics(communication, content_request,
                                                                                 'satellite')
                                self.track_content_hit(communication, content_request, content, slot,
                                                       observed_delay_ms=seg_delay_s * 1000.0,
                                                       from_cache=False, energy_joule=energy_joule)
                                use_federated = self.algorithm in [
                                    "Federated_MAB",
                                    "Federated_MAB_EnergyAware",
                                    "Enhanced_Federated_MAB",
                                    "Enhanced_Federated_MAB_EnergyAware",
                                ]
                                requesting_vehicle.update_action_space(content, slot, federated_update=use_federated)
                                return content

        # nothing returned yet
        return None

    def track_content_request(self, requested_content):
        """Add this to track what content is being requested"""

        # Initialize request tracking
        if not hasattr(self, 'recent_requests'):
            self.recent_requests = []

        content_key = f"{requested_content['content_type']}_{requested_content['content_coord']}_{requested_content['content_category']}_{requested_content['content_no']}"

        # Track recent requests (keep last 100)
        self.recent_requests.append(content_key)
        if len(self.recent_requests) > 100:
            self.recent_requests.pop(0)

        # Update total request count
        s_type = requested_content['content_type']
        s_coord = requested_content['content_coord']
        s_category = requested_content['content_category']
        s_no = requested_content['content_no']

        if s_type in self.record and s_coord in self.record[s_type] and s_category in self.record[s_type][
            s_coord] and s_no in self.record[s_type][s_coord][s_category]:
            old_total = self.record[s_type][s_coord][s_category][s_no].get('total_requests', 0)
            self.record[s_type][s_coord][s_category][s_no]['total_requests'] = old_total + 1

    def track_content_hit(self, communication, content_request, content, slot, observed_delay_ms=None,
                          from_cache=False, energy_joule=None):
        """
        Increment hit counters and (optionally) cached-only delay accumulators.
        from_cache=True only for UAV local/neighbor cache hits.
        """
        content_type = content['content_type']
        content_coord = content['content_coord']
        content_category = content['content_category']
        content_no = content['content_no']

        self._ensure_record_structure(content_type, content_coord, content_category, content_no)

        e = self.record[content_type][content_coord][content_category][content_no]
        e['slot'] = slot

        if energy_joule is not None:
            e['energy_sum'] = e.get('energy_sum', 0.0) + float(energy_joule)
            e['energy_count'] = e.get('energy_count', 0) + 1
            content_request['energy_joule'] = float(energy_joule)
            content_request['energy_cost_joule'] = float(energy_joule)
            content_request['normalized_energy'] = min(
                1.0,
                float(energy_joule) / max(1e-6, float(getattr(self, 'max_energy_per_request', 100.0)))
            )

        if from_cache:
            e['content_hit'] = e.get('content_hit', 0) + 1
            self.cache_hits_tracked = getattr(self, '', 0) + 1  # ✅ correctly indented
            if observed_delay_ms is not None:
                e['cache_hit_delay_sum'] = e.get('cache_hit_delay_sum', 0.0) + float(observed_delay_ms)
                e['cache_hit_delay_count'] = e.get('cache_hit_delay_count', 0) + 1

    def generate_content(self, slot, current_time, no_of_content_each_category):
        content_validity = 0.5 * 60  # 15 minutes in seconds

        if (slot == 1):
            for category in self.content_categories:
                category_contents = []
                content_size = self.content_size.get(category, 0)
                num_contents = no_of_content_each_category
                for n in range(1, num_contents + 1):
                    content = {
                        'content_type': 'UAV',
                        'content_category': category,
                        'content_coord': int(self.uav_id.replace('UAV', '')),  # Convert "UAV1" → 1
                        'size': content_size,
                        'content_no': n,
                        'destination': None,
                        'generation_time': current_time,
                        'hop_count': 0,
                        'content_validity': content_validity,
                        'content_hit': 0,
                    }
                    category_contents.append(content)
                self.generated_cache[category] = category_contents
        else:
            for category, contents in self.generated_cache.items():
                for content in contents:
                    content['generation_time'] = current_time
        #print('hi66', self.generated_cache)

    def print_cache(self):
        for category, contents in self.generated_cache.items():
            for content in contents:
                print(f"Content Number: {content['content_no']}", end=' ')
            print()

    def cache_content(self, communication, communication_schedule, slot, uavs, satellites):
        target_slot = 1
        if slot > 10:
            target_slot = slot - 1
        # Determine connected satellites and non-neighbor UAVs
        satellite_list = []
        non_neighbor_uavs = []
        satellite_list = communication.get_connected_satellites(target_slot, communication_schedule, satellites)
        non_neighbor_uavs = [uav for uav in uavs if
                             uav.uav_id != self.uav_id and uav.uav_id not in [neighbor.uav_id for neighbor in
                                                                              self.neighbors]]

        # Update the action space based on connected satellites
        use_federated = self.algorithm in [
            "Federated_MAB",
            "Federated_MAB_EnergyAware",
            "Enhanced_Federated_MAB",
            "Enhanced_Federated_MAB_EnergyAware",
        ]
        self.update_action_space(satellite_list, non_neighbor_uavs, slot, federated_update=use_federated )

        # Select an action (which content to cache) based on the learned policy
        self.select_action()


    def update_action_space(self, satellite_list, non_neighbor_uavs, slot, federated_update=False):
        """
        UAV update_action_space with P, Q, S calculations and unified structure

        Args:
            satellite_list: List of available satellites for pre-caching
            non_neighbor_uavs: List of non-neighbor UAVs accessible via satellite
            slot: Current time slot
            federated_update: Whether to send federated update
        """
        #print(f"UAV update_content_cache {self.content_cache}")
        # Clear action space
        self.action_space = []

        # Initialize record if needed
        if not hasattr(self, 'record'):
            self.record = {}

        # Initialize request count if needed
        if not hasattr(self, 'request_receive_count'):
            self.request_receive_count = 1  # Avoid division by zero

        # PART 1: Process content already in cache (if using unified structure)
        if hasattr(self, 'content_cache'):
            # Use 4-level iteration for unified structure
            for content_type in self.content_cache:  # Level 1: TYPE
                for coord in self.content_cache[content_type]:  # Level 2: COORD
                    for category in self.content_cache[content_type][coord]:  # Level 3: CATEGORY
                        content_list = self.content_cache[content_type][coord][category]  # Level 4: LIST

                        for content in content_list:
                            if not isinstance(content, dict):
                                continue

                            # Initialize record structure if needed
                            if content_type not in self.record:
                                self.record[content_type] = {}
                            if coord not in self.record[content_type]:
                                self.record[content_type][coord] = {}
                            if category not in self.record[content_type][coord]:
                                self.record[content_type][coord][category] = {}

                            content_no = content.get('content_no')
                            if content_no not in self.record[content_type][coord][category]:
                                self.record[content_type][coord][category][content_no] = {
                                    'weighted_request_tracking': 0,
                                    'request_tracking': 0,
                                    'content_hit': 0,
                                    'q_value': 0.0,
                                    'avg_reward': 0.0,
                                    'no_f_time_cached': 0,
                                    'slot': 0,
                                    'cache_path_reqs': 0,
                                    'cache_hit_delay_sum': 0.0,
                                    'cache_hit_delay_count': 0,
                                }

                            # Calculate weighted popularity (YOUR ORIGINAL LOGIC)
                            record_entry = self.record[content_type][coord][category][content_no]

                            if self.request_receive_count > 0:
                                record_entry['weighted_request_tracking'] = self.sigmoid(
                                    (0.25 * record_entry['weighted_request_tracking']) +
                                    0.75 * (record_entry['request_tracking'] / self.request_receive_count)
                                )

                            # Get P, Q, S values (YOUR ORIGINAL CALCULATIONS)
                            p = record_entry['weighted_request_tracking']  # Popularity
                            s = record_entry['slot']  # Slot
                            q = self.sigmoid(record_entry['avg_reward'])  # Q-value

                            # Update Q-value in record
                            record_entry['q_value'] = q

                            # Add to action space
                            self.action_space.append({
                                'content': content,
                                'q_value': q,
                                'slot': s,
                                'popularity': p
                            })

                           # print(f"part 1 Content No: {content}")

        # PART 2: Add satellite content to action space (for pre-caching)
        for satellite in satellite_list:
            sat_id = satellite.satellite_id if hasattr(satellite, 'satellite_id') else str(satellite)
            sat_id = int(sat_id.replace('Satellite', ''))
           # print(f'Satellite- {sat_id} is now accessible')

            # Get satellite's content catalog
            if hasattr(satellite, 'generated_cache'):
                for category, content_list in satellite.generated_cache.items():
                        for content_item in content_list:
                            # Create unified content structure
                            content = {
                                'content_type': 'satellite',
                                'content_coord': sat_id,
                                'content_category': category,
                                'content_no': content_item.get('content_no', content_item.get('no', 0)),
                                'size': content_item.get('size', self.content_size.get(category, 10)),
                                'generation_time': content_item.get('generation_time', 0),
                                'content_validity': content_item.get('content_validity', 1000)
                            }

                            #print(f"part 4  Content No: {content}")

                            # Add other fields from original
                            for key, value in content_item.items():
                                if key not in content:
                                    content[key] = value

                            # Initialize record for satellite content
                            if 'satellite' not in self.record:
                                self.record['satellite'] = {}
                            if sat_id not in self.record['satellite']:
                                self.record['satellite'][sat_id] = {}
                            if category not in self.record['satellite'][sat_id]:
                                self.record['satellite'][sat_id][category] = {}

                            content_no = content['content_no']
                            if content_no not in self.record['satellite'][sat_id][category]:
                                self.record['satellite'][sat_id][category][content_no] = {
                                    'weighted_request_tracking': 0,
                                        'request_tracking': 0,
                                        'content_hit': 0,
                                        'q_value': 0.0,
                                        'avg_reward': 0.0,
                                        'no_f_time_cached': 0,
                                        'slot': 0,
                                        'cache_path_reqs': 0,
                                        'cache_hit_delay_sum': 0.0,
                                        'cache_hit_delay_count': 0,
                                }

                            # Calculate P, Q, S for satellite content
                            record_entry = self.record['satellite'][sat_id][category][content_no]

                            if self.request_receive_count > 0:
                                record_entry['weighted_request_tracking'] = self.sigmoid(
                                    (0.25 * record_entry['weighted_request_tracking']) +
                                    0.75 * (record_entry['request_tracking'] / self.request_receive_count)
                                )

                            p = record_entry['weighted_request_tracking']
                            s = record_entry['slot']
                            q = self.sigmoid(record_entry['avg_reward'])

                            record_entry['q_value'] = q

                            # Add to action space
                            self.action_space.append({
                                'content': content,
                                'q_value': q,
                                'slot': s,
                                'popularity': p
                            })

                          #  print(f" part 2 Content No: {content})")

        # PART 3: Add non-neighbor UAV content to action space
        for uav in non_neighbor_uavs:
            # Get UAV's generated content
            if hasattr(uav, 'generated_cache'):
                for category, content_list in uav.generated_cache.items():
                    if isinstance(content_list, list):
                        for content_item in content_list:
                            # Get UAV ID (just the number part)
                            uav_id = uav.uav_id if hasattr(uav, 'uav_id') else str(uav)
                            uav_num = int(uav_id.replace('UAV', '')) if 'UAV' in uav_id else uav_id

                            # Create unified content structure
                            content = {
                                'content_type': 'UAV',
                                'content_coord': uav_num,  # Store as integer
                                'content_category': category,
                                'content_no': content_item.get('content_no', content_item.get('no', 0)),
                                'size': content_item.get('size', self.content_size.get(category, 10)),
                                'generation_time': content_item.get('generation_time', 0),
                                'content_validity': content_item.get('content_validity', 1000)
                            }

                            # Add other fields
                            for key, value in content_item.items():
                                if key not in content:
                                    content[key] = value

                            # Initialize record for UAV content
                            if 'UAV' not in self.record:
                                self.record['UAV'] = {}
                            if uav_num not in self.record['UAV']:
                                self.record['UAV'][uav_num] = {}
                            if category not in self.record['UAV'][uav_num]:
                                self.record['UAV'][uav_num][category] = {}

                            content_no = content['content_no']
                            if content_no not in self.record['UAV'][uav_num][category]:
                                self.record['UAV'][uav_num][category][content_no] = {
                                'weighted_request_tracking': 0,
                                'request_tracking': 0,
                                'content_hit': 0,
                                'q_value': 0.0,
                                'avg_reward': 0.0,
                                'no_f_time_cached': 0,
                                'slot': 0,
                                'total_requests': 0,
                                'cache_path_reqs': 0,
                                'cache_hit_delay_sum': 0.0,
                                'cache_hit_delay_count': 0,
                                }

                            # Calculate P, Q, S for UAV content
                            record_entry = self.record['UAV'][uav_num][category][content_no]

                            if self.request_receive_count > 0:
                                record_entry['weighted_request_tracking'] = self.sigmoid(
                                    (0.25 * record_entry['weighted_request_tracking']) +
                                    0.75 * (record_entry['request_tracking'] / self.request_receive_count)
                                )

                            p = record_entry['weighted_request_tracking']
                            s = record_entry['slot']
                            q = self.sigmoid(record_entry['avg_reward'])

                            record_entry['q_value'] = q

                            # Add to action space
                            self.action_space.append({
                                'content': content,
                                'q_value': q,
                                'slot': s,
                                'popularity': p
                            })

                           # print(f" part 3 Content No: {content})")

        # Debug output (optional)
        #print(f"📊 UAV {self.uav_id}: Action space updated with {self.action_space} items")

        # Handle federated update if needed
        #if federated_update and hasattr(self, 'aggregator') and self.aggregator:
           #self.send_update_to_federated_server()



    def select_action(self):
        """
        ✅ UNIFIED ALGORITHM DISPATCHER
        Routes to appropriate caching algorithm based on self.algorithm
        """
        if self.algorithm == "LRU":
            return self.select_action_lru()
        elif self.algorithm == "Popularity":
            return self.select_action_popularity()
        elif self.algorithm == "MAB_Original":
            return self.select_action_mab_original()
        elif self.algorithm in ["MAB_Contextual", "MAB_Contextual_EnergyAware"]:
            return self.select_action_mab_contextual()
        elif self.algorithm in ["Federated_MAB", "Federated_MAB_EnergyAware"]:
            return self.select_action_federated_mab()
        elif self.algorithm in ["Enhanced_Federated_MAB", "Enhanced_Federated_MAB_EnergyAware"]:
            return self.select_action_enhanced_federated_mab()
        else:
            print(f"  ⚠️ Unknown algorithm {self.algorithm}, defaulting to MAB_Contextual")
            return self.select_action_mab_contextual()

    def select_action_lru(self):
        #print(f"🔄 LRU: Starting with {len(self.action_space)} actions, cache size: {self.cache_size}")

        self.clear_cache()
        remaining_cache_size = self.cache_size
        start_time = time.time()

        if not self.action_space:
            #print("   ❌ LRU: No actions in action_space!")
            return

        sorted_actions = sorted(self.action_space, key=lambda action: action.get('slot', 0), reverse=True)
        print(f"   📋 LRU: Sorted {len(sorted_actions)} actions by slot (recent first)")

        cached_count = 0
        for action in sorted_actions:
            selected_content = action['content']
            content_size = selected_content['size']
            slot = action.get('slot', 0)

            print(f"   🔍 LRU: Considering content slot={slot}, size={content_size}MB, remaining={remaining_cache_size}MB")

            if remaining_cache_size - content_size >= 0.0:
                self._cache_content_helper(selected_content)
                cached_count += 1
                remaining_cache_size -= content_size
                #print(f"   ✅ LRU: Cached content! Count={cached_count}, remaining={remaining_cache_size}MB")
            else:
                print(f"   ❌ LRU: Not enough space for content (need {content_size}MB, have {remaining_cache_size}MB)")
                break

        print(f"   📊 LRU: Final result - cached {cached_count} items")
        return

    def select_action_popularity(self):
        """POPULARITY ALGORITHM: Cache based on request frequency"""
        self.clear_cache()
        remaining_cache_size = self.cache_size
        start_time = time.time()

        if not self.action_space:
            return

        sorted_actions = sorted(self.action_space, key=lambda action: action.get('popularity', 0), reverse=True)

        cached_count = 0
        for action in sorted_actions:
            selected_content = action['content']
            content_size = selected_content['size']

            if remaining_cache_size - content_size >= 0.01:
                self._cache_content_helper(selected_content)
                cached_count += 1
                remaining_cache_size -= content_size
            else:
                break

        decision_latency = time.time() - start_time
        if decision_latency > 0:
            self.avg_decision_latency += decision_latency
            self.decision_count += 1

        return

    # 🔍 ADD THESE DEBUG PRINTS TO UAV select_action_mab_original()

    def select_action_mab_original(self):
        """DEBUG VERSION: Find why UAVs aren't caching at α=0.25"""

        print(f"\n🚁 UAV DEBUG - {self.uav_id}")

        self.clear_cache()
        remaining_cache_size = self.cache_size
        start_time = time.time()

        current_alpha = getattr(self, 'current_alpha', 1.0)

        # DEBUG: Basic parameters
        print(f"   🔍 Alpha: {current_alpha:.2f}")
        print(f"   🔍 Action space size: {len(self.action_space)}")
        print(f"   🔍 Cache size: {remaining_cache_size}")

        if not self.action_space:
            print("   ❌ NO ACTION SPACE - UAV can't cache anything!")
            return

        # Alpha-adaptive parameters (your fixed version)
        if current_alpha > 1.5:
            epsilon = 0.05
            cache_threshold = 0.30
            print(f"   🎯 HIGH ALPHA MODE: threshold={cache_threshold}")
        elif current_alpha > 0.8:
            epsilon = 0.15
            cache_threshold = 0.45
            print(f"   ⚖️ MEDIUM ALPHA MODE: threshold={cache_threshold}")
        else:
            epsilon = 0.25
            cache_threshold = 0.40  # Your reduced threshold
            print(f"   🔍 LOW ALPHA MODE: threshold={cache_threshold}")

        # Calculate UCB scores and find max
        max_ucb = 0
        qualifying_actions = 0

        for action in self.action_space:
            content = action['content']
            s_type = content['content_type']
            s_coord = content['content_coord']
            s_category = content['content_category']
            s_no = content['content_no']

            # Your existing UCB calculation...
            record_entry = self.record.get(s_type, {}).get(s_coord, {}).get(s_category, {}).get(s_no, {})

            selection_count = max(1, int(record_entry.get('no_f_time_cached', 0)))
            avg_reward = float(record_entry.get('avg_reward', record_entry.get('q_value', 0.0)))
            popularity_score = float(action.get('popularity', 0.0))
            enhanced_q_value = avg_reward + 0.15 * popularity_score

            current_slot = getattr(self, 'current_slot', 1)
            if current_slot <= 0:
                current_slot = 1

            if selection_count > 0 and current_slot > 1:
                exploration = np.sqrt((2 * np.log(current_slot)) / selection_count)

            else:
                exploration = 10.0

            if current_alpha > 1.5:
                ucb_score = enhanced_q_value + (0.1 * exploration)
            elif current_alpha > 0.8:
                ucb_score = enhanced_q_value + (0.3 * exploration)
            else:
                ucb_score = enhanced_q_value + (0.5 * exploration)

            action['ucb_score'] = ucb_score
            max_ucb = max(max_ucb, ucb_score)

            # Count how many actions qualify for caching
            if ucb_score > cache_threshold:
                qualifying_actions += 1

        # 🔍 CRITICAL DEBUG INFO
        print(f"   📊 Max UCB score: {max_ucb:.3f}")
        print(f"   📊 Cache threshold: {cache_threshold:.3f}")
        print(f"   📊 Actions above threshold: {qualifying_actions}/{len(self.action_space)}")

        if max_ucb <= cache_threshold:
            print(f"   ❌ PROBLEM FOUND: No actions meet cache threshold!")
            print(f"   💡 Suggestion: Lower threshold from {cache_threshold:.2f} to {max_ucb * 0.8:.2f}")
            return

        # Epsilon-greedy decision
        use_exploration = random.random() < epsilon
        print(f"   🎲 Using {'EXPLORATION' if use_exploration else 'EXPLOITATION'} (ε={epsilon:.2f})")

        if use_exploration:
            selected_actions = random.sample(self.action_space, min(len(self.action_space), 5))
        else:
            sorted_actions = sorted(self.action_space, key=lambda x: x.get('ucb_score', 0), reverse=True)
            selected_actions = sorted_actions

        # Caching attempts with detailed logging
        cached_count = 0
        cache_attempts = 0

        for action in selected_actions:
            selected_content = action['content']
            content_size = selected_content['size']
            ucb_score = action.get('ucb_score', 0)

            # Caching decision logic
            should_cache = False

            if current_alpha > 1.5:
                should_cache = (ucb_score > cache_threshold) or (action.get('popularity', 0) > 0.3)
            elif current_alpha > 0.8:
                should_cache = ucb_score > cache_threshold
            else:
                should_cache = ucb_score > cache_threshold and action.get('q_value', enhanced_q_value) > 0.1

            if should_cache and remaining_cache_size - content_size >= 0.01:
                cache_attempts += 1
                try:
                    self._cache_content_helper(selected_content)
                    cached_count += 1
                    remaining_cache_size -= content_size

                    # Update record structure (your existing pattern)
                    s_type = selected_content['content_type']
                    s_coord = selected_content['content_coord']
                    s_category = selected_content['content_category']
                    s_no = selected_content['content_no']

                    if s_type not in self.record:
                        self.record[s_type] = {}
                    if s_coord not in self.record[s_type]:
                        self.record[s_type][s_coord] = {}
                    if s_category not in self.record[s_type][s_coord]:
                        self.record[s_type][s_coord][s_category] = {}
                    if s_no not in self.record[s_type][s_coord][s_category]:
                        self.record[s_type][s_coord][s_category][s_no] = {
                            'weighted_request_tracking': 0,
                            'request_tracking': 0,
                            'content_hit': 0,
                            'q_value': 0.0,
                            'avg_reward': 0.0,
                            'no_f_time_cached': 0,
                            'slot': 0,
                            'cache_path_reqs': 0,
                            'cache_hit_delay_sum': 0.0,
                            'cache_hit_delay_count': 0,
                        }

                    self.record[s_type][s_coord][s_category][s_no]['no_f_time_cached'] += 1

                    print(f"   ✅ CACHED: {s_type}_{s_coord}_{s_category}_{s_no} (UCB: {ucb_score:.3f})")

                except Exception as e:
                    print(f"   ❌ CACHE FAILED: {e}")

            elif not should_cache:
                print(
                    f"   ⏭️  SKIPPED: {selected_content['content_type']} (UCB: {ucb_score:.3f} < threshold: {cache_threshold:.3f})")
            elif remaining_cache_size - content_size < 0.01:
                print(f"   💾 NO SPACE: Need {content_size:.2f}, have {remaining_cache_size:.2f}")
                break

        print(
            f"   📦 UAV RESULT: Cached {cached_count}/{cache_attempts} attempts, {len(self.action_space)} total actions")

        # Decision latency tracking
        decision_latency = time.time() - start_time
        if decision_latency > 0:
            self.avg_decision_latency += decision_latency
            self.decision_count += 1

        return

    def select_action_mab_contextual(self):
        """
        FIXED: Contextual MAB that focuses on performance like Federated MAB
        Removes UCB exploration dominance and treats context as enhancement
        """
        # Get node identifier for logging
        node_id = getattr(self, 'uav_id', getattr(self, 'vehicle_id', getattr(self, 'bs_id', 'Node')))
        print(f"\n🧠 FIXED CONTEXTUAL MAB - {node_id}")

        remaining_cache_size = self.cache_size
        start_time = time.time()

        if not self.action_space:
            print("   ❌ No action space!")
            return

        current_slot = getattr(self, 'current_slot', 1)
        if current_slot <= 0:
            current_slot = 1

        print(f"   📊 Slot: {current_slot}, Actions: {len(self.action_space)}")

        # Track already cached content (same as existing code)
        already_cached = set()
        if hasattr(self, 'content_cache'):
            for content_type in self.content_cache:
                for coord in self.content_cache[content_type]:
                    for category in self.content_cache[content_type][coord]:
                        for content in self.content_cache[content_type][coord][category]:
                            content_id = f"{content_type}_{coord}_{category}_{content.get('content_no', '')}"
                            already_cached.add(content_id)

        print(f"📦 Already cached: {len(already_cached)} unique items")

        # Calculate contextual Q-values (FIXED: No UCB dominance)
        for i, action in enumerate(self.action_space):
            content = action['content']
            s_type = content['content_type']
            s_coord = content['content_coord']
            s_category = content['content_category']
            s_no = content['content_no']

            # Get record entry (same as existing code)
            record_entry = self.record.get(s_type, {}).get(s_coord, {}).get(s_category, {}).get(s_no, {})
            selection_count = record_entry.get('no_f_time_cached', 0)
            hit_count = record_entry.get('content_hit', 0)
            request_count = record_entry.get('request_tracking', 0)

            # Base Q-value (same as MAB Original and Federated MAB)
            if selection_count > 0:
                local_q_value = hit_count / selection_count
            else:
                local_q_value = 0.0

            # CONTEXTUAL FEATURES (same calculations, but used differently)
            # 1. Popularity Index
            total_requests = max(1, sum(
                self.record.get(et, {}).get(ec, {}).get(ecat, {}).get(eno, {}).get('request_tracking', 0)
                for et in self.record for ec in self.record[et]
                for ecat in self.record[et][ec] for eno in self.record[et][ec][ecat]
            ))
            popularity_index = request_count / total_requests

            # 2. Freshness Index
            content_validity = content.get('content_validity', 3600)
            generation_time = content.get('generation_time', 0)
            current_time = current_slot * 60
            time_elapsed = max(0, current_time - generation_time)
            remaining_lifetime = max(0, content_validity - time_elapsed)
            freshness_index = remaining_lifetime / content_validity if content_validity > 0 else 0.0

            # 3. Cache Efficiency
            content_size = content.get('size', 1.0)
            cache_efficiency = min(1.0, self.cache_size / (content_size * 10.0))

            # Context vector
            context_vector = np.array([popularity_index, freshness_index, cache_efficiency])

            # Node-specific contextual weights (can be learned or tuned)
            if hasattr(self, 'lambda_weights'):
                lambda_weights = self.lambda_weights
            else:
                # Default weights based on node type
                if hasattr(self, 'uav_id'):
                    lambda_weights = np.array([0.5, 0.4, 0.1])  # UAV: popularity + freshness
                elif hasattr(self, 'vehicle_id'):
                    lambda_weights = np.array([0.6, 0.3, 0.1])  # Vehicle: popularity focused
                else:  # BS
                    lambda_weights = np.array([0.4, 0.3, 0.3])  # BS: balanced

            # FIXED: Small contextual enhancement (NOT dominance like UCB)
            contextual_bonus = np.dot(lambda_weights, context_vector) * 0.15  # Scale factor 0.15

            # NEW: Add small exploration for truly new content (much smaller than UCB)
            if selection_count == 0:
                exploration_bonus = 0.1  # Small boost for new content
            else:
                exploration_bonus = 0.0

            # FINAL: Contextual Q-value (similar to federated_q in Federated MAB)
            contextual_q_value = local_q_value + contextual_bonus + exploration_bonus

            # Store in action
            action['contextual_q_value'] = contextual_q_value
            action['base_q_value'] = local_q_value
            action['contextual_bonus'] = contextual_bonus
            action['popularity'] = popularity_index
            action['freshness'] = freshness_index
            action['cache_efficiency'] = cache_efficiency

            # Debug first few actions
            if i < 3:
                print(f"   Action {i}: {s_type}_{s_coord}_{s_category}_{s_no}")
                print(f"      Base Q: {local_q_value:.4f}, Context bonus: {contextual_bonus:.4f}")
                print(
                    f"      Features: Pop={popularity_index:.3f}, Fresh={freshness_index:.3f}, Cache={cache_efficiency:.3f}")
                print(f"      Final Contextual Q: {contextual_q_value:.4f}")

        # Sort by contextual Q-value (like Federated MAB sorts by federated_q_value)
        sorted_actions = sorted(self.action_space, key=lambda x: x.get('contextual_q_value', 0), reverse=True)

        print(f"🏆 Top 5 by Contextual Q-value:")
        for i, action in enumerate(sorted_actions[:5]):
            content = action['content']
            cq = action.get('contextual_q_value', 0)
            bq = action.get('base_q_value', 0)
            cb = action.get('contextual_bonus', 0)
            content_id = f"{content['content_type']}_{content['content_coord']}_{content['content_category']}_{content['content_no']}"
            print(f"   {i + 1}. {content_id}: CQ={cq:.4f} (Base={bq:.4f} + Context={cb:.4f})")

        # Cache content with duplicate prevention (same as existing logic)
        cached_count = 0
        cache_decisions = []

        for i, action in enumerate(sorted_actions):
            selected_content = action['content']
            content_size = selected_content['size']
            content_id = f"{selected_content['content_type']}_{selected_content['content_coord']}_{selected_content['content_category']}_{selected_content['content_no']}"

            # Skip if already cached
            if content_id in already_cached:
                if i < 10:  # Log first 10 for debugging
                    cache_decisions.append(f"⏭️  SKIP: {content_id} - Already cached")
                continue

            # Cache if space available
            if remaining_cache_size - content_size >= 0.01:
                self._cache_content_helper(selected_content)
                already_cached.add(content_id)
                cached_count += 1
                remaining_cache_size -= content_size

                if i < 10:  # Log first 10 for debugging
                    cq = action.get('contextual_q_value', 0)
                    cache_decisions.append(f"✅ CACHED: {content_id} (CQ: {cq:.4f}, Size: {content_size:.2f})")
            else:
                if i < 10:
                    cache_decisions.append(
                        f"💾 NO SPACE: {content_id} - Need {content_size:.2f}, have {remaining_cache_size:.2f}")
                break

        # Print caching decisions for debugging
        for decision in cache_decisions:
            print(f"   {decision}")

        # Summary statistics
        if cached_count > 0:
            avg_contextual_q = sum(
                action.get('contextual_q_value', 0) for action in sorted_actions[:cached_count]) / cached_count
            avg_base_q = sum(action.get('base_q_value', 0) for action in sorted_actions[:cached_count]) / cached_count
            print(f"   📊 SUMMARY:")
            print(f"      Cached: {cached_count} items")
            print(f"      Avg Contextual Q: {avg_contextual_q:.4f}")
            print(f"      Avg Base Q: {avg_base_q:.4f}")
            print(f"      Context Enhancement: {avg_contextual_q - avg_base_q:.4f}")
            print(f"      Remaining space: {remaining_cache_size:.2f}/{self.cache_size}")

        print(f"📦 Fixed Contextual MAB cached {cached_count} items (focusing on performance)")

        # Track decision latency (same as existing code)
        decision_latency = time.time() - start_time
        if decision_latency > 0:
            self.avg_decision_latency += decision_latency
            self.decision_count += 1

        return

    # ADD THIS TO DEBUG CACHE HITS
    # ===============================================

    def debug_cache_hits(self, slot):
        """Add this method to UAV class and call it every 50 slots"""
        if slot % 10 == 0:
            print(f"\n🎯 CACHE HIT DEBUG - {self.uav_id} - Slot {slot}")
            print("=" * 40)

            # Check if we have any content requests
            if hasattr(self, 'request_receive_count'):
                print(f"📥 Total requests received: {self.request_receive_count}")
            else:
                print("❌ No request_receive_count attribute!")

            # Check content hits
            total_hits = 0
            content_with_hits = 0

            for entity_type, coord_dict in self.record.items():
                for coord, category_dict in coord_dict.items():
                    for category, content_no_dict in category_dict.items():
                        for content_no, content_info in content_no_dict.items():
                            hits = content_info.get('content_hit', 0)
                            if hits > 0:
                                total_hits += hits
                                content_with_hits += 1
                                print(f"   🎯 HIT: {entity_type}_{coord}_{category}_{content_no} = {hits} hits")

            print(f"📊 Summary: {total_hits} total hits across {content_with_hits} content items")

            if total_hits == 0:
                print("❌ CRITICAL: NO CACHE HITS! This is why rewards are zero!")
                print("🔧 POSSIBLE CAUSES:")
                print("   1. Content not being requested")
                print("   2. Cached content doesn't match requested content")
                print("   3. Cache lookup failing")
                print("   4. Content expiring before being hit")
            else:
                print("✅ Cache hits are happening")

            print("=" * 40)



    def debug_algorithm_performance(self, slot):
        """Debug performance for current algorithm"""
        if slot % 20 == 0:
            print(f"\n📊 UAV {self.uav_id} - {self.algorithm} Performance (Slot {slot}):")

            if hasattr(self, 'content_hit') and hasattr(self, 'request_receive_from_other_source'):
                requests = self.request_receive_from_other_source
                hits = self.content_hit
                hit_ratio = hits / max(1, requests)
                print(f"   Cache Hit Ratio: {hits}/{requests} = {hit_ratio:.3f} ({hit_ratio * 100:.1f}%)")

            if self.algorithm in ['MAB_Original', 'MAB_Contextual', 'Federated_MAB']:
                # Show learning progress
                learned_content = 0
                for entity_type, coord_dict in self.record.items():
                    for coord, category_dict in coord_dict.items():
                        for category, content_no_dict in category_dict.items():
                            for content_no, content_info in content_no_dict.items():
                                if content_info.get('q_value', 0) > 0.1:
                                    learned_content += 1

                print(f"   Learning: {learned_content} content items with Q > 0.1")

            if self.algorithm in ['MAB_Contextual', 'Federated_MAB']:
                print(f"   Context: Using popularity, freshness, cache efficiency")

    def debug_content_requests_vs_cache(self, slot):
        """Debug why cached content isn't being hit"""

        if slot % 100 == 0:
            print(f"\n🔍 CONTENT REQUEST vs CACHE DEBUG - {self.uav_id}")
            print("=" * 60)

            # Show what's in cache
            print("📦 CACHED CONTENT:")
            cached_content = []
            for coord, coord_data in self.content_cache.items():
                for category, content_list in coord_data.items():
                    for content in content_list:
                        content_key = f"{content['content_type']}_{content['content_coord']}_{content['content_category']}_{content['content_no']}"
                        cached_content.append(content_key)
                        print(f"   {content_key}")

            print(f"   Total cached: {len(cached_content)} items")

            # Show what's being requested (add this to your request handling)
            print("📥 RECENT CONTENT REQUESTS:")
            if hasattr(self, 'recent_requests'):
                for req in self.recent_requests[-10:]:  # Last 10 requests
                    print(f"   {req}")
            else:
                print("   No recent_requests tracking - ADD THIS!")

            # Check for mismatches
            print("🔍 ANALYSIS:")
            if not cached_content:
                print("   ❌ Nothing in cache!")
            elif not hasattr(self, 'recent_requests'):
                print("   ❌ Not tracking requests - can't verify matches!")
            else:
                print(f"   Cache has {len(cached_content)} items")
                print(f"   Need to verify request patterns match")

    def select_action_federated_mab(self):
        """
        UNIFIED Federated MAB - Works for ALL node types (BS, Vehicle, UAV)
        """

        remaining_cache_size = self.cache_size
        start_time = time.time()

        if not self.action_space:
            return

        # Get node identifier for logging
        node_id = getattr(self, 'bs_id', getattr(self, 'vehicle_id', getattr(self, 'uav_id', 'Node')))

        print(f"\n🤖 FEDERATED MAB - {node_id}")
        print(f"📊 Action space: {len(self.action_space)} items")

        # 🚨 UNIFIED: Build cached content set - SAME CODE FOR ALL NODES!
        already_cached = set()

        if hasattr(self, 'content_cache'):
            for content_type in self.content_cache:
                for coord in self.content_cache[content_type]:
                    for category in self.content_cache[content_type][coord]:
                        for content in self.content_cache[content_type][coord][category]:
                            content_id = f"{content_type}_{coord}_{category}_{content.get('content_no', '')}"
                            already_cached.add(content_id)

        print(f"📦 Already cached: {len(already_cached)} unique items")

        # Calculate federated Q-values for each action
        for action in self.action_space:
            content = action['content']

            # Local Q-value
            local_q_value = self.get_local_q_value(content)

            # Aggregate neighbors' Q-values
            neighbor_q_values = []
            if hasattr(self, 'aggregator') and self.aggregator:
                neighbor_values = self.aggregator.get_neighbor_q_values(
                    node_id,
                    content['content_type'],
                    content['content_coord'],
                    content['content_category'],
                    content['content_no']
                )
                if neighbor_values:
                    neighbor_q_values = neighbor_values

            # Federated Q-value calculation
            if neighbor_q_values:
                federated_q = 0.7 * local_q_value + 0.3 * np.mean(neighbor_values)
            else:
                federated_q = local_q_value

            action['federated_q_value'] = federated_q

        # Sort by federated Q-value
        sorted_actions = sorted(
            self.action_space,
            key=lambda x: x['federated_q_value'],
            reverse=True
        )

        # Cache content with duplicate prevention
        cached_count = 0
        for action in sorted_actions:
            selected_content = action['content']
            content_size = selected_content['size']

            # Create unique content identifier - UNIFIED FORMAT
            content_id = f"{selected_content['content_type']}_{selected_content['content_coord']}_{selected_content['content_category']}_{selected_content['content_no']}"

            # Skip if already cached
            if content_id in already_cached:
                continue

            # Cache if space available
            if remaining_cache_size - content_size >= 0.01:
                # Use UNIFIED cache helper
                self._cache_content_helper(selected_content)
                already_cached.add(content_id)  # Track cached content
                cached_count += 1
                remaining_cache_size -= content_size
                print(f"   ✅ CACHED: {content_id} (Q: {action['federated_q_value']:.4f})")
            else:
                break

        print(f"📦 Federated MAB cached {cached_count} UNIQUE items")

        # Track decision latency
        end_time = time.time()
        decision_latency = end_time - start_time
        self.avg_decision_latency = ((self.avg_decision_latency * self.decision_count) + decision_latency) / (
                self.decision_count + 1)
        self.decision_count += 1

    def select_action_enhanced_federated_mab(self):
        """
        UNIFIED Enhanced Federated MAB - Works for ALL node types
        - Blends local Q + contextual signal with a federated prior.
        - Federation gets higher weight right after each epoch change.
        - No helpers used; everything is inlined here.
        """

        import time
        import numpy as np

        remaining_cache_size = float(getattr(self, "cache_size", 0.0))
        start_time = time.time()

        if not getattr(self, "action_space", None):
            return

        # Node identifier (works for BS / Vehicle / UAV)
        node_id = getattr(self, "bs_id",
                          getattr(self, "vehicle_id",
                                  getattr(self, "uav_id", "Node")))

        print(f"\n🤖 ENHANCED FEDERATED MAB - {node_id}")
        print(f"📊 Action space: {len(self.action_space)} items")

        # Build cached content set (duplicate prevention)
        already_cached = set()
        if hasattr(self, "content_cache") and isinstance(self.content_cache, dict):
            for ctype, coords in self.content_cache.items():
                for coord, cats in coords.items():
                    for cat, clist in cats.items():
                        for c in clist:
                            cno = c.get("content_no", "")
                            already_cached.add(f"{ctype}_{coord}_{cat}_{cno}")

        print(f"📦 Already cached: {len(already_cached)} unique items")

        # Epoch parameters (align these in main.py: node.epoch_len = communication.epoch_len)
        current_slot = int(getattr(self, "current_slot", 1) or 1)
        epoch_len = int(getattr(self, "epoch_len", 100) or 100)
        pos_in_epoch = (current_slot - 1) % max(1, epoch_len)
        progress = pos_in_epoch / float(max(1, epoch_len))  # 0..1 within epoch

        # Federation weight decreases over the epoch; local weight increases
        w_fed = 0.50 * (1.0 - progress) + 0.20 * progress  # 0.50 → 0.20
        w_loc = 1.0 - w_fed

        # Pre-fetch totals for contextual features
        total_req = float(getattr(self, "total_request", 0.0) or 0.0)

        # Calculate enhanced scores
        for action in self.action_space:
            content = action["content"]

            # --- Local Q (existing method in your code) ---
            local_q_value = float(self.get_local_q_value(content))

            # --- Federated prior (neighbors' Q) ---
            neighbor_q_values = []
            if hasattr(self, "aggregator") and self.aggregator:
                try:
                    nv = self.aggregator.get_neighbor_q_values(
                        node_id,
                        content.get("content_type"),
                        content.get("content_coord"),
                        content.get("content_category"),
                        content.get("content_no"),
                    )
                    # Normalize return type: scalar, list/tuple, or dict
                    if isinstance(nv, (int, float)):
                        neighbor_q_values = [float(nv)]
                    elif isinstance(nv, dict):
                        neighbor_q_values = [float(v) for v in nv.values()]
                    elif isinstance(nv, (list, tuple)):
                        neighbor_q_values = [float(v) for v in nv]
                    else:
                        neighbor_q_values = []
                except Exception:
                    neighbor_q_values = []
            neighbor_mean = float(np.mean(neighbor_q_values)) if neighbor_q_values else 0.5

            # --- Contextual signal (same ingredients you use in contextual MAB) ---
            req_count = float(action.get("request_count", content.get("request_count", 0)) or 0.0)
            selection_count = int(action.get("selection_count", content.get("selection_count", 0)) or 0)

            popularity_index = (req_count / max(1.0, total_req)) if total_req > 0 else 0.0

            remaining_lifetime = float(content.get("remaining_lifetime", 0.0) or 0.0)
            validity = float(content.get("content_validity",
                                         content.get("validity", 0.0)) or 0.0)
            freshness_index = (remaining_lifetime / validity) if validity > 0 else 0.0

            content_size = float(content.get("content_size_mb", content.get("size", 1.0)) or 1.0)
            cache_size = float(getattr(self, "cache_size", 1.0) or 1.0)
            cache_efficiency = min(1.0, cache_size / (content_size * 10.0))

            # Same weights you already use
            lambda_weights = np.array([0.4, 0.4, 0.2], dtype=float)
            ctx_vec = np.array([popularity_index, freshness_index, cache_efficiency], dtype=float)
            contextual_bonus = float(np.dot(lambda_weights, ctx_vec)) * 0.15

            exploration_bonus = 0.1 if selection_count == 0 else 0.0

            contextual_q = local_q_value + contextual_bonus + exploration_bonus

            # --- Final epoch-aware blend ---
            enhanced_score = (w_loc * contextual_q) + (w_fed * neighbor_mean)
            action["enhanced_score"] = float(enhanced_score)

        # Sort by enhanced score
        sorted_actions = sorted(self.action_space, key=lambda x: x["enhanced_score"], reverse=True)

        # Cache content with duplicate prevention
        cached_count = 0
        for action in sorted_actions:
            selected_content = action["content"]
            # Robust size field
            content_size = float(selected_content.get("size",
                                                      selected_content.get("content_size_mb", 0.0)) or 0.0)

            content_id = f"{selected_content.get('content_type')}_{selected_content.get('content_coord')}_{selected_content.get('content_category')}_{selected_content.get('content_no')}"

            # Skip if already cached
            if content_id in already_cached:
                continue

            # Cache if space available
            if remaining_cache_size - content_size >= 0.01:
                # Use your existing cache helper
                self._cache_content_helper(selected_content)
                already_cached.add(content_id)
                cached_count += 1
                remaining_cache_size -= content_size
                print(f"   ✅ CACHED: {content_id} (Score: {action['enhanced_score']:.4f})")
            else:
                break

        print(f"📦 Enhanced Federated MAB cached {cached_count} UNIQUE items")

        # Track decision latency
        end_time = time.time()
        decision_latency = end_time - start_time
        prev_avg = float(getattr(self, "avg_decision_latency", 0.0) or 0.0)
        prev_cnt = int(getattr(self, "decision_count", 0) or 0)
        self.avg_decision_latency = (prev_avg * prev_cnt + decision_latency) / (prev_cnt + 1)
        self.decision_count = prev_cnt + 1

    def _cache_content_helper(self, content):
        """
        UNIFIED cache helper - works for BS, Vehicle, and UAV
        Creates structure: content_cache[type][coord][category] = [contents]
        """

        # Extract content details (these already exist in content object)
        content_type = content.get('content_type')  # 'satellite', 'UAV', 'terrestrial'
        content_coord = content.get('content_coord')  # Sat ID, UAV ID, or grid coord
        # ALWAYS convert coord to string
        #content_coord = str(content.get('content_coord'))
        content_category = content.get('content_category')  # 'I', 'II', 'III', 'IV'

        # Initialize cache structure if needed (3-level nested dict)
        if not hasattr(self, 'content_cache'):
            self.content_cache = {}

        # Level 1: Content type
        if content_type not in self.content_cache:
            self.content_cache[content_type] = {}

        # Level 2: Coordinate/ID
        if content_coord not in self.content_cache[content_type]:
            self.content_cache[content_type][content_coord] = {}

        # Level 3: Category
        if content_category not in self.content_cache[content_type][content_coord]:
            self.content_cache[content_type][content_coord][content_category] = []

        # Add content to cache (checking for duplicates is done in the calling function)
        self.content_cache[content_type][content_coord][content_category].append(content)

        # Update record structure (same pattern)
        if not hasattr(self, 'record'):
            self.record = {}

        if content_type not in self.record:
            self.record[content_type] = {}
        if content_coord not in self.record[content_type]:
            self.record[content_type][content_coord] = {}
        if content_category not in self.record[content_type][content_coord]:
            self.record[content_type][content_coord][content_category] = {}

        content_no = content.get('content_no')
        if content_no not in self.record[content_type][content_coord][content_category]:
            self.record[content_type][content_coord][content_category][content_no] = {
                                'weighted_request_tracking': 0,
                                'request_tracking': 0,
                                'content_hit': 0,
                                'q_value': 0.0,
                                'avg_reward': 0.0,
                                'no_f_time_cached': 0,
                                'slot': 0,
                                'total_requests': 0,
                                'cache_path_reqs': 0,
                                'cache_hit_delay_sum': 0.0,
                                'cache_hit_delay_count': 0,
            }

        # Increment cache count
        self.record[content_type][content_coord][content_category][content_no]['no_f_time_cached'] += 1

        s_type = content['content_type']
        s_coord = content['content_coord']
        s_category = content['content_category']
        s_no = content['content_no']
        content_id = f"{s_type}_{s_coord}_{s_category}_{s_no}"

        print(f"🔧 DEBUG CACHE: {content_id} added to cache")

    def calculate_local_q_value(self, content, action):
        """Calculate local Q-value using contextual features"""
        try:
            avg_reward = self.record[content['content_type']][content['content_coord']][content['content_category']][
                content['content_no']]['avg_reward']
            weighted_request_tracking = \
                self.record[content['content_type']][content['content_coord']][content['content_category']][
                    content['content_no']]['weighted_request_tracking']
        except:
            avg_reward = 0
            weighted_request_tracking = 0

        time_spent = getattr(self, 'current_time', 0) - content.get('generation_time', 0)
        content_validity = content.get('content_validity', 1800)

        return self.sigmoid(
            avg_reward +
            (self.cache_size / content['size']) +
            ((content_validity - time_spent) / content_validity) +
            weighted_request_tracking
        )

    def clear_cache(self):
        """Clear the existing content cache"""
        self.content_cache = {}

    def debug_get_reward_calls(self):
        """Add this to your get_reward() method at the very beginning"""

        print(f"\n🔍 get_reward() CALLED - {self.uav_id}")

        # Check if there are any cache hits to process
        total_hits_found = 0
        content_with_hits = []

        for entity_type, coord_dict in self.record.items():
            for coord, category_dict in coord_dict.items():
                for category, content_no_dict in category_dict.items():
                    for content_no, content_info in content_no_dict.items():
                        hits = content_info.get('content_hit', 0)
                        if hits > 0:
                            total_hits_found += hits
                            content_key = f"{entity_type}_{coord}_{category}_{content_no}"
                            content_with_hits.append((content_key, hits))

        print(f"   Found {total_hits_found} total hits across {len(content_with_hits)} content items")

        if content_with_hits:
            print("   Content with hits:")
            for content_key, hits in content_with_hits[:5]:  # Show first 5
                print(f"      {content_key}: {hits} hits")
        else:
            print("   ❌ No content hits found in get_reward()")

        return total_hits_found > 0

    def get_reward(self):
        """
        ✅ UNIFIED REWARD DISPATCHER
        Routes to appropriate reward calculation based on self.algorithm
        """
        if self.algorithm == "LRU":
            return self.get_reward_lru()
        elif self.algorithm == "Popularity":
            return self.get_reward_popularity()
        elif self.algorithm == "MAB_Original":
            return self.get_reward_mab_original()
        elif self.algorithm in ["MAB_Contextual", "MAB_Contextual_EnergyAware"]:
            return self.get_reward_mab_contextual()
        elif self.algorithm in ["Federated_MAB", "Federated_MAB_EnergyAware"]:
            return self.get_reward_federated_mab()
        elif self.algorithm in ["Enhanced_Federated_MAB", "Enhanced_Federated_MAB_EnergyAware"]:
            return self.get_reward_enhanced_federated_mab()  # ✅ ADD THIS LINE
        else:
            print(f"  ⚠️ Unknown algorithm {self.algorithm}, defaulting to MAB_Contextual")
            return self.get_reward_mab_contextual()
    # ===============================
    # 2. ALGORITHM-SPECIFIC REWARD FUNCTIONS - ADD THESE NEW METHODS
    # ===============================

    def get_reward_lru(self):
        """LRU doesn't use learning rewards - just track recency"""
        # LRU is rule-based, no learning needed
        pass

    def get_reward_popularity(self):
        """Popularity-based reward - tracks request frequency"""
        for entity_type, coord_dict in self.record.items():
            for coord, category_dict in coord_dict.items():
                for category, content_no_dict in category_dict.items():
                    for content_no, content_info in content_no_dict.items():
                        requests = content_info.get('request_tracking', 0)
                        if requests > 0:
                            # Simple popularity tracking
                            total_requests = sum(
                                self.record.get(et, {}).get(ec, {}).get(ecat, {}).get(eno, {}).get('request_tracking',
                                                                                                   0)
                                for et in self.record for ec in self.record[et]
                                for ecat in self.record[et][ec] for eno in self.record[et][ec][ecat]
                            )
                            popularity_score = requests / max(1, total_requests)
                            content_info['popularity_score'] = popularity_score

    def get_reward_mab_original(self):
        lr = getattr(self, "learning_rate", 0.1)
        theta_ms_default = getattr(self, "theta_ms_default", 2000.0)
        cap = getattr(self, "delay_ratio_cap", 5.0)

        for ctype, d1 in self.record.items():
            for ccoord, d2 in d1.items():
                for ccat, d3 in d2.items():
                    for cno, e in d3.items():
                        attempts = int(e.get('cache_path_reqs', 0))
                        hits = int(e.get('content_hit', 0))
                        CH = (hits / float(max(1, attempts))) if attempts > 0 else 0.0

                        dsum = float(e.get('cache_hit_delay_sum', 0.0))
                        dcnt = int(e.get('cache_hit_delay_count', 0))
                        avg_ms = (dsum / dcnt) if dcnt > 0 else theta_ms_default

                        theta = self._get_theta_bound_ms_for_content(ctype, ccat, default_theta_ms=theta_ms_default)
                        delay_term = min(theta / max(1.0, avg_ms), cap)

                        reward = CH + delay_term

                        old = float(e.get('avg_reward', 0.0))
                        new = old + lr * (reward - old)
                        e['avg_reward'] = new
                        e['q_value'] = new

    def get_reward_mab_contextual(self):
        lr = getattr(self, "learning_rate", 0.1)
        theta_ms_default = getattr(self, "theta_ms_default", 2000.0)
        cap = getattr(self, "delay_ratio_cap", 5.0)

        w_base = getattr(self, "w_base", 1.0)
        w_ctx = getattr(self, "w_ctx", 0.3)
        w_beta = getattr(self, "w_beta", 1.0)
        w_rho = getattr(self, "w_rho", 1.0)
        w_s = getattr(self, "w_s", 1.0)

        for ctype, d1 in self.record.items():
            for ccoord, d2 in d1.items():
                for ccat, d3 in d2.items():
                    for cno, e in d3.items():
                        attempts = int(e.get('cache_path_reqs', 0))
                        hits = int(e.get('content_hit', 0))
                        CH = (hits / float(max(1, attempts))) if attempts > 0 else 0.0

                        dsum = float(e.get('cache_hit_delay_sum', 0.0))
                        dcnt = int(e.get('cache_hit_delay_count', 0))
                        avg_ms = (dsum / dcnt) if dcnt > 0 else theta_ms_default

                        theta = self._get_theta_bound_ms_for_content(ctype, ccat, default_theta_ms=theta_ms_default)
                        delay_term = min(theta / max(1.0, avg_ms), cap)
                        base = CH + delay_term

                        # contextual nudges
                        beta = self._get_popularity_index(e, ctype, ccoord, ccat, cno)
                        rho = self._get_freshness_ratio(ctype, ccoord, ccat, cno)
                        s = self._get_cache_size_ratio(ctype, ccat, cno)
                        s_norm = min(1.0, s)

                        avg_energy = float(e.get('energy_sum', 0.0)) / max(1, int(e.get('energy_count', 0)))
                        energy_penalty = getattr(self, 'energy_lambda', 0.0) * (
                            avg_energy / max(1e-6, float(getattr(self, 'max_energy_per_request', 100.0)))
                        )

                        reward = (w_base * base) + (w_ctx * ((w_beta * beta) + (w_rho * rho) + (w_s * s_norm))) - energy_penalty

                        old = float(e.get('avg_reward', 0.0))
                        new = old + lr * (reward - old)
                        e['avg_reward'] = new
                        e['q_value'] = new

    def _get_theta_bound_ms_for_content(self, content_type, content_category, default_theta_ms=2000.0):
        """
        Map (tier=content_type, category) -> Θ (ms).
        If you already store Θ somewhere, read it here.
        Fallbacks follow your manuscript Table III: Aerial ~2s, Satellite ~10s, Terrestrial ~5s.
        """
        # You can refine this with your actual lookup table:
        t = str(content_type).lower()
        c = str(content_category).lower()
        # defaults (ms)
        theta_sat_ms = 10_000.0
        theta_air_ms = 2_000.0
        theta_terr_ms = 5_000.0

        if "sat" in t:
            return theta_sat_ms
        if "uav" in t or "aerial" in t or "air" in t:
            return theta_air_ms
        if "veh" in t or "bs" in t or "ter" in t or "ground" in t:
            return theta_terr_ms
        return float(default_theta_ms)

    def _get_popularity_index(self, entry, content_type, content_coord, content_category, content_no):
        """
        β (0..1): simple, robust popularity index from your counters.
        You already track request_tracking (reqs) on this UAV for that content.
        Map to [0,1] via a saturating transform.
        """
        reqs = int(entry.get("request_tracking", 0))
        # Saturate after ~50 requests in a period
        beta = reqs / float(reqs + 50)
        return max(0.0, min(1.0, beta))

    def _get_freshness_ratio(self, content_type, content_coord, content_category, content_no):
        """
        ρ (0..1): remaining freshness window ratio.
        If you don’t track per-item generation time, return 1.0 (fresh).
        Hook this to your content metadata if available.
        """
        try:
            # If your content objects carry 'gen_time' and 'freshness_sec', you can look them up.
            # Here we default to "fresh" because UAV pre-caches periodically.
            return 1.0
        except Exception:
            return 1.0

    def _get_cache_size_ratio(self, content_type, content_category, content_no):
        """
        s: cache size ratio = (available capacity / item size).
        Larger items should contribute less (i.e., s smaller) so we bound to [0,1].
        """
        # Get item size (MB) from your caches if available; else use category defaults from your manuscript Table III
        size_mb = self._get_item_size_mb_fallback(content_category)
        # available capacity; if you keep fixed capacity, just use total capacity
        cap_mb = float(getattr(self, "cache_capacity_mb", 50.0))
        if size_mb <= 0.0:
            return 1.0
        raw = cap_mb / size_mb
        # normalize to ~[0,1] with a soft cap so huge ratios don’t dominate
        return min(1.0, raw)

    def _get_item_size_mb_fallback(self, content_category):
        """
        From manuscript Table III: Category I/II/III/IV => 10MB, 1MB, 0.1MB, 2MB (tweak if you have exact values in the object)
        """
        cat = str(content_category).lower()
        if cat in ("1", "i", "hdf"):
            return 10.0
        if cat in ("2", "ii", "image", "img"):
            return 1.0
        if cat in ("3", "iii", "sensor"):
            return 0.1
        if cat in ("4", "iv", "video", "vid"):
            return 2.0
        return 1.0

    def get_reward_federated_mab(self):
        """Federated MAB reward - includes global knowledge"""
        # Start with contextual reward
        self.get_reward_mab_contextual()

        # Add federated knowledge if available
        if hasattr(self, 'aggregator') and self.aggregator:
            for entity_type, coord_dict in self.record.items():
                for coord, category_dict in coord_dict.items():
                    for category, content_no_dict in category_dict.items():
                        for content_no, content_info in content_no_dict.items():
                            try:
                                content_key = f"{entity_type}_{coord}_{category}_{content_no}"
                                global_q = self.aggregator.get_global_q_value(content_key)
                                local_q = content_info.get('q_value', 0)

                                # Weighted combination
                                alpha = 0.6  # Local weight
                                federated_q = alpha * local_q + (1 - alpha) * global_q
                                content_info['q_value'] = federated_q
                                content_info['federated_q_value'] = federated_q
                            except:
                                pass

    def get_reward_enhanced_federated_mab(self):
        """
        Enhanced Federated MAB reward with performance-based weighting and trust factors
        """
        # Start with base contextual reward calculation
        self.get_reward_mab_contextual()

        # Enhanced federated integration if aggregator available
        if hasattr(self, 'aggregator') and self.aggregator:
            for entity_type, coord_dict in self.record.items():
                for coord, category_dict in coord_dict.items():
                    for category, content_no_dict in category_dict.items():
                        for content_no, content_info in content_no_dict.items():
                            try:
                                # Get enhanced neighbor information
                                content_key = f"{entity_type}_{coord}_{category}_{content_no}"

                                # Get node ID (works for Vehicle, UAV, BS)
                                node_id = getattr(self, 'vehicle_id',
                                                  getattr(self, 'uav_id',
                                                          getattr(self, 'bs_id', 'Unknown')))

                                # Get enhanced neighbor data with weights
                                neighbor_data = self.aggregator.get_enhanced_neighbor_values(
                                    node_id, entity_type, coord, category, content_no
                                )

                                local_q = content_info.get('q_value', 0.0)

                                if neighbor_data and neighbor_data['q_values']:
                                    # Enhanced weighted aggregation
                                    q_values = neighbor_data['q_values']
                                    weights = neighbor_data['weights']

                                    # Calculate weighted neighbor average
                                    weighted_sum = sum(q * w for q, w in zip(q_values, weights))
                                    total_weight = sum(weights)

                                    if total_weight > 0:
                                        weighted_neighbor_avg = weighted_sum / total_weight

                                        # Enhanced trust factor (based on node performance)
                                        trust_factor = min(0.4, 0.1 + (len(q_values) * 0.05))

                                        # Enhanced Q-value with performance weighting
                                        enhanced_q = (local_q * (1 - trust_factor)) + \
                                                     (weighted_neighbor_avg * trust_factor)

                                        # Apply stability bounds
                                        content_info['q_value'] = max(0.0, min(1.0, enhanced_q))

                                        # Update enhanced metrics for aggregator
                                        content_info['enhanced_update_count'] = content_info.get(
                                            'enhanced_update_count', 0) + 1
                                        content_info['trust_factor'] = trust_factor
                                        content_info['neighbor_influence'] = trust_factor * weighted_neighbor_avg

                            except Exception as e:
                                # Graceful fallback - continue with local Q-value
                                continue

        # Update enhanced statistics for aggregator
        if hasattr(self, 'aggregator') and hasattr(self.aggregator, 'update_node_statistics'):
            enhanced_stats = self.get_enhanced_statistics()
            node_id = getattr(self, 'vehicle_id',
                              getattr(self, 'uav_id',
                                      getattr(self, 'bs_id', 'Unknown')))
            self.aggregator.update_node_statistics(node_id, enhanced_stats)

    def get_enhanced_statistics(self):
        """
        Enhanced statistics for performance-based aggregation
        """
        if not hasattr(self, 'total_requests_tracked'):
            self.total_requests_tracked = 0
            self.cache_hits_tracked = 0

        # Calculate hit ratio
        hit_ratio = (self.content_hit / max(1, self.request_receive_from_other_source))

        # Calculate Q-value variance for stability assessment
        q_values = []
        content_counts = 0

        for entity_type, coord_dict in self.record.items():
            for coord, category_dict in coord_dict.items():
                for category, content_no_dict in category_dict.items():
                    for content_no, content_info in content_no_dict.items():
                        q_val = content_info.get('q_value', 0.0)
                        if q_val > 0:
                            q_values.append(q_val)
                            content_counts += 1

        # Q-value variance (stability metric)
        if len(q_values) > 1:
            mean_q = sum(q_values) / len(q_values)
            variance = sum((q - mean_q) ** 2 for q in q_values) / len(q_values)
            q_value_variance = variance
        else:
            q_value_variance = 0.0

        # Content type experience distribution
        content_type_experience = {}
        for entity_type in self.record:
            content_type_experience[entity_type] = len(self.record[entity_type])

        return {
            'total_requests': self.request_receive_from_other_source,
            'cache_hits': self.content_hit,
            'hit_ratio': hit_ratio,
            'q_value_variance': q_value_variance,
            'regional_importance': 0.5 + (hit_ratio * 0.5),  # Based on performance
            'coverage_quality': min(1.0, content_counts / 50.0),  # Content diversity
            'last_update_time': time.time(),
            'content_type_experience': content_type_experience,
            'trust_factor': 0.3 + (hit_ratio * 0.4),  # Performance-based trust
            'stability_score': max(0.0, 1.0 - q_value_variance),  # Lower variance = higher stability
            'content_diversity': len(content_type_experience)
        }

    def is_content_in_cache(self, coord, category, content_no):
        """
        CRITICAL METHOD - Add this to UAV class
        Check if specific content is in the cache
        """
        try:
            # Check if the content exists in cache structure
            if coord in self.content_cache:
                if category in self.content_cache[coord]:
                    for cached_content in self.content_cache[coord][category]:
                        if cached_content.get('content_no') == content_no:
                            return True
            return False
        except Exception as e:
            print(f"   ❌ Error in is_content_in_cache: {e}")
            return False

    def is_within_coverage(self, x, y):
        """
        ✅ ROBUST COVERAGE CHECK - WORKS WITH ANY CONFIGURATION
        Check if location falls in UAV's range with proper bounds checking

        REPLACE YOUR EXISTING is_within_coverage METHOD WITH THIS
        """
        try:
            (x1, y1), (x2, y2) = self.coverage_area

            # Ensure coordinates are valid
            x = max(0, min(x, self.grid_size - 1)) if hasattr(self, 'grid_size') else x
            y = max(0, min(y, self.grid_size - 1)) if hasattr(self, 'grid_size') else y

            # Check if point is within coverage area
            within_coverage = x1 <= x < x2 and y1 <= y < y2

            return within_coverage

        except Exception as e:
            print(f"⚠️  {self.uav_id} coverage check failed for ({x},{y}): {e}")
            # Fallback: assume it's not in coverage
            return False

    def send_update_to_federated_server(self):
        """Sends the UAV's Q-values to the federated server"""
        if hasattr(self, 'aggregator') and self.aggregator:
            self.aggregator.receive_update(self.uav_id, self.record)

    def get_local_q_value(self, content):
        """Get Q-value for content from record"""
        if not isinstance(content, dict):
            return 0.5

        content_type = content.get('content_type')
        content_coord = str(content.get('content_coord'))
        content_category = content.get('content_category')
        content_no = content.get('content_no')

        if (hasattr(self, 'record') and
                content_type in self.record and
                content_coord in self.record[content_type] and
                content_category in self.record[content_type][content_coord] and
                content_no in self.record[content_type][content_coord][content_category]):
            return self.record[content_type][content_coord][content_category][content_no].get('q_value', 0.5)

        return 0.5

    def update_enhanced_tracking(self):
        """OPTIONAL: Update enhanced tracking if enabled"""
        if hasattr(self, 'aggregator') and hasattr(self.aggregator, 'update_node_statistics'):
            stats = self.get_enhanced_statistics()
            self.aggregator.update_node_statistics(self.uav_id, stats)


    def run(self, current_time, communication, communication_schedule, slot, satellites, no_of_content_each_category,
            uav_content_generation_period, epsilon, uavs):
        self.current_slot = slot
        if (slot - 1) % uav_content_generation_period == 0:
            self.request_tracking = {}
            self.request_receive_count= 0
            self.generate_content(slot, current_time, no_of_content_each_category)  # Generate first
            self.cache_cleanup(current_time)  # Then cleanup
            if slot > 10:
                self.get_reward()
            self.cache_content(communication, communication_schedule, slot, uavs, satellites)
            self.send_update_to_federated_server()
        self.debug_cache_hits(slot)
        # if slot % 10 == 0:
        #     self.debug_content_requests_vs_cache(slot)
        #     self.debug_reward_processing_detailed(slot)
        #     cached_items = self.debug_cache_structure(slot)
        #     if cached_items == 0:
        #         print("   ❌ CRITICAL: Nothing in cache - check caching logic!")
