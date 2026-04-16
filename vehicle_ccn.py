# vehicle_ccn.py

import random
import math
import time
import copy
import numpy as np


class Vehicle:
    def __init__(self, vehicle_id, grid_size, vehicle_range, vehicle_speed, aggregator, algorithm="MAB_Contextual"):
        # Existing initialization...
        self.vehicle_id = vehicle_id
        self.grid_size = grid_size
        self.vehicle_speed = vehicle_speed
        self.v_range = vehicle_range
        self.content_categories = ['II', 'III', 'IV']
        self.content_size = {'I': 10, 'II': 1, 'III': 0.01, 'IV': 2}
        self.cache_size = 50
        self.transmission_rate = 6

        # ✅ NEW: Algorithm selection
        self.algorithm = algorithm
        print(f"  🚗 {self.vehicle_id} initialized with {self.algorithm} algorithm")

        # Initialize location
        self.current_location = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))

        # Initialize counters for requests generated per category and received content per category
        self.requests_generated_per_category = {category: 0 for category in self.content_categories}
        self.content_received_count_per_category = {category: 0 for category in self.content_categories}
        self.content_cache = {}  # Cache to store content received from satellites and other UAVs
        self.active_requests = set()  # Set to store active content requests
        self.cache_cleanup_interval = 60  # Cache cleanup interval in seconds
        self.epsilon = 0.1
        self.action_space = []
        self.record = {}
        self.aggregator = aggregator

        # Action space and learning components
        self.num_actions = 0  # Initialize the number of actions to zero
        self.generated_cache = {
            'II': [],
            'III': [],
            'IV': []  # Add more categories if needed
        }

        # Initialize the request tracking data structure
        self.request_receive = 0
        self.request_receive_cache = 0
        self.content_cache_hit = 0
        self.content_hit_from_source = 0
        self.global_request_receive_count = 0
        self.num_context_parameters = 3
        self.avg_decision_latency = 0
        self.decision_count = 0

        # ✅ ADD MISSING PERFORMANCE TRACKING ATTRIBUTES
        self.cache_hit = 0
        self.source_hit = 0
        self.total_request = 0
        self.total_request_for_caching = 0

    def cache_cleanup(self, current_time):
        """
        UNIFIED cache cleanup - works for ALL node types (BS, Vehicle, UAV)
        Removes expired content from the unified cache structure
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
                        # Check if content is still valid
                        if isinstance(content, dict):  # Safety check
                            generation_time = content.get('generation_time', 0)
                            validity = content.get('content_validity', float('inf'))

                            if current_time < generation_time + validity:
                                valid_contents.append(content)
                            else:
                                # Debug: log expired content
                                node_id = getattr(self, 'bs_id',
                                                  getattr(self, 'vehicle_id', getattr(self, 'uav_id', 'Node')))
                                content_no = content.get('content_no', 'unknown')
                                # print(f"🗑️ {node_id}: Expired {content_type}/{coord}/{category}/{content_no}")

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

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def move(self):
        # 0 = up, 1 = down, 2 = left, 3 = right
        direction = random.randint(0, 3)
        x, y = self.current_location
        if direction == 0:  # up
            y = max(0, y - self.vehicle_speed)
        elif direction == 1:  # down
            y = min(self.grid_size - 1, y + self.vehicle_speed)
        elif direction == 2:  # left
            x = max(0, x - self.vehicle_speed)
        else:  # right
            x = min(self.grid_size - 1, x + self.vehicle_speed)
        self.current_location = (x, y)

    def is_within_range(self, other_location):
        x1, y1 = self.current_location
        x2, y2 = other_location
        return abs(x1 - x2) <= self.v_range and abs(y1 - y2) <= self.v_range

    def get_content_size(self, category):
        return self.content_size.get(category, 0)

    def process_content_request(self, communication, content_request, slot, grid_size, flag):
        """Fixed version with proper record initialization"""
        # print(f"content request from vehilce:{content_request}")

        requested_entity_type = content_request['type']
        requested_coord = content_request['coord']
        requested_category = content_request['category']
        requested_content_no = content_request['no']

        # Validate category
        if requested_entity_type == 'satellite':
            valid_categories = ['I', 'II', 'III']
        elif requested_entity_type == 'UAV':
            valid_categories = ['II', 'III', 'IV']
        elif requested_entity_type == 'grid':
            valid_categories = ['II', 'III', 'IV']
        else:
            valid_categories = ['II', 'III', 'IV']

        if requested_category not in valid_categories:
            return False

        # Track requests
        self.request_receive += 1
        self.total_request += 1

        # 🔥 CRITICAL FIX: Initialize record structure with ALL required keys
        self._ensure_record_structure(requested_entity_type, requested_coord, requested_category, requested_content_no)

        # Track this request
        self.record[requested_entity_type][requested_coord][requested_category][requested_content_no][
            'request_tracking'] += 1
        self.record[requested_entity_type][requested_coord][requested_category][requested_content_no][
            'total_requests'] += 1

        # Initialize vehicle_list properly
        if 'vehicle_list' not in content_request:
            content_request['vehicle_list'] = []
        content_request['vehicle_list'].append(self.vehicle_id)

        # Rest of your existing process_content_request logic...
        if requested_entity_type == "satellite" or requested_entity_type == "UAV":
            self.request_receive_cache += 1
            self.total_request_for_caching += 1

            entry = self.record[requested_entity_type][requested_coord][requested_category][requested_content_no]
            entry['cache_path_reqs'] = entry.get('cache_path_reqs', 0) + 1

            content = self.find_cache(communication, content_request, slot)
            if content:
                self.cache_hit += 1
                if flag == 0:
                    communication.content_hit += 1

                if communication.content_received_time > (content_request['hop_count'] * random.uniform(0, 0.001)) + (
                        content_request['hop_count'] * ((content['size'] * 8) / self.transmission_rate)):
                    communication.content_received_time = (content_request['hop_count'] * random.uniform(0, 0.001)) + (
                            content_request['hop_count'] * ((content['size'] * 8) / self.transmission_rate))
                return True
            else:
                return False

        # Grid content processing...
        if requested_entity_type == "grid":
            try:
                grid_coords = communication.get_coordinates_from_index(requested_coord, grid_size)
                if self.is_within_range(grid_coords):
                    content = self.generate_content(communication, content_request, slot)
                    if content:
                        self.source_hit += 1
                        return True
                else:
                    self.request_receive_cache += 1
                    self.total_request_for_caching += 1
                    entry = self.record[requested_entity_type][requested_coord][requested_category][
                        requested_content_no]
                    entry['cache_path_reqs'] = entry.get('cache_path_reqs', 0) + 1
                    content = self.find_cache(communication, content_request, slot)
                    if content:
                        self.cache_hit += 1
                        if flag == 0:
                            communication.content_hit += 1
                        if communication.content_received_time > (
                                content_request['hop_count'] * random.uniform(0, 0.001)) + (
                                content_request['hop_count'] * ((content['size'] * 8) / self.transmission_rate)):
                            communication.content_received_time = (content_request['hop_count'] * random.uniform(0,
                                                                                                                 0.001)) + (
                                                                          content_request['hop_count'] * ((content[
                                                                                                               'size'] * 8) / self.transmission_rate))
                        return True
                    else:
                        return False
            except Exception as e:
                print(f"❌ Error in vehicle grid processing: {e}")
                return False

        return False

    def generate_content(self, communication, content_request, slot):
        """
        ✅ FIXED: Generate grid content with proper category validation
        """
        content_validity = 0.5 * 60  # 10 minutes in seconds
        # Get the content category and size based on the request
        content_category = content_request['category']

        # ✅ FIXED: Validate category for grid content
        valid_grid_categories = ['II', 'III', 'IV']
        if content_category not in valid_grid_categories:
            print(f"❌ Invalid grid category {content_category}, using default 'II'")
            content_category = 'II'  # Fallback to valid category

        content_size = self.content_size.get(content_category, 0)  # Default to 0 if category is not found
        request_recieve_time = content_request['g_time'] + (content_request['hop_count'] * 0.001)
        content_request['time_spent'] = request_recieve_time

        content = {
            'unique_id': content_request['unique_id'],
            'destination': content_request['requesting_vehicle'],
            'generation_time': request_recieve_time,
            'hop_count': 0,
            'content_type': 'grid',  # You can specify the content type here
            'content_category': content_category,
            'content_coord': content_request['coord'],  # Fixed typo: was 'content_cord'
            'content_no': content_request['no'],
            'content_validity': content_validity,
            'content_receive_time': 0,
            'size': content_size  # Include the content size based on category
        }
        self.content_hit_from_source += 1
        use_federated = self.algorithm in ["Federated_MAB", "Enhanced_Federated_MAB"]
        self.update_action_space(content, slot, federated_update=use_federated)  # cache the content
        return content

    def find_cache(self, communication, content_request, slot):
        # print('hi-v')
        type = content_request['type']
        coord = content_request['coord']
        category = content_request['category']

        if type in self.content_cache:
            # print('hi-v1')
            if coord in self.content_cache[type]:
                # print('hi-v2')
                if category in self.content_cache[type][coord]:
                    # print('hi-v3')
                    category_cache = self.content_cache[type][coord][category]
                    for content in category_cache:
                        if content['content_no'] == content_request['no']:
                            # print('hi-v4')
                            # compute a simple segment delay consistent with vehicle transfer model
                            seg_delay_s = (content_request['hop_count'] * random.uniform(0, 0.001)) + \
                                          (content_request['hop_count'] * (
                                                      (content['size'] * 8) / self.transmission_rate))
                            self.track_content_hit(content, slot, observed_delay_ms=seg_delay_s * 1000.0,
                                                   from_cache=True)
                            print(f" Vehicle {self.vehicle_id} VEHICLE CACHE HIT")
                            return content
        return None

    def track_content_hit(self, content, slot, observed_delay_ms=None, from_cache=False):
        """Enhanced content hit tracking for learning"""
        content_type = content['content_type']
        content_coord = content['content_coord']
        content_category = content['content_category']
        content_no = content['content_no']

        # Ensure record structure
        if content_type not in self.record:
            self.record[content_type] = {}
        if content_coord not in self.record[content_type]:
            self.record[content_type][content_coord] = {}
        if content_category not in self.record[content_type][content_coord]:
            self.record[content_type][content_coord][content_category] = {}
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

        e = self.record[content_type][content_coord][content_category][content_no]
        e['slot'] = slot
        if from_cache:
            e['content_hit'] = e.get('content_hit', 0) + 1
            if observed_delay_ms is not None:
                e['cache_hit_delay_sum'] = e.get('cache_hit_delay_sum', 0.0) + float(observed_delay_ms)
                e['cache_hit_delay_count'] = e.get('cache_hit_delay_count', 0) + 1

        print(f"   📊 Hit tracked for {content_type}_{content_coord}_{content_category}_{content_no}")
        print(
            f"      Total hits: {self.record[content_type][content_coord][content_category][content_no]['content_hit']}")

        return

    def update_action_space(self, content, slot, federated_update):
        # Initialize the Q-value for the content to 0
        # Add content and its associated Q-value
        if content['content_type'] not in self.record:
            self.record[content['content_type']] = {}
        if content['content_coord'] not in self.record[content['content_type']]:
            self.record[content['content_type']][content['content_coord']] = {}
        if content['content_category'] not in self.record[content['content_type']][content['content_coord']]:
            self.record[content['content_type']][content['content_coord']][content['content_category']] = {}
        if content['content_no'] not in self.record[content['content_type']][content['content_coord']][
            content['content_category']]:
            self.record[content['content_type']][content['content_coord']][content['content_category']][
                content['content_no']] = {
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
        time_spent = (slot * 60) - content['generation_time']
        if self.request_receive > 0:
            self.record[content['content_type']][content['content_coord']][content['content_category']][
                content['content_no']] \
                ['weighted_request_tracking'] \
                = self.sigmoid((0.25 *
                                self.record[content['content_type']][content['content_coord']][
                                    content['content_category']][content['content_no']] \
                                    ['weighted_request_tracking']) + 0.75 * (
                                       self.record[content['content_type']][content['content_coord']][
                                           content['content_category']][content['content_no']][
                                           'request_tracking'] / self.request_receive))

        p = self.record[content['content_type']][content['content_coord']][content['content_category']][
            content['content_no']] \
            ['weighted_request_tracking']
        s = self.record[content['content_type']][content['content_coord']][content['content_category']][
            content['content_no']]['slot']

        q = self.sigmoid(self.record[content['content_type']][content['content_coord']][
                             content['content_category']] \
                             [content['content_no']]['avg_reward'])

        self.record[content['content_type']][content['content_coord']][content['content_category']][
            content['content_no']]['q_value'] = q

        # Add content and associated Q-value to the action space
        self.action_space.append({'content': content, 'q_value': q, 'slot': s, 'popularity': p})
        return

    def append_action_space(self, slot, federated_update):
        # Add content from content_cache to the action space
        for type, c_type in self.content_cache.items():
            for coord, coord_data in c_type.items():
                for category, category_data in coord_data.items():
                    for content in category_data:
                        # Initialize the Q-value for the content to 0
                        if content['content_type'] not in self.record:
                            self.record[content['content_type']] = {}
                        if content['content_coord'] not in self.record[content['content_type']]:
                            self.record[content['content_type']][content['content_coord']] = {}
                        if content['content_category'] not in self.record[content['content_type']][
                            content['content_coord']]:
                            self.record[content['content_type']][content['content_coord']][
                                content['content_category']] = {}
                        if content['content_no'] not in self.record[content['content_type']][content['content_coord']][ \
                                content['content_category']]:
                            self.record[content['content_type']][content['content_coord']][content['content_category']][ \
                                content['content_no']] = {
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
                        time_spent = (slot * 60) - content['generation_time']
                        if self.request_receive > 0:
                            den = max(1, self.request_receive)   # << guard
                            self.record[content['content_type']][content['content_coord']][content['content_category']][
                                content['content_no']] \
                                ['weighted_request_tracking'] \
                                = self.sigmoid((0.25 *
                                                self.record[content['content_type']][content['content_coord']][
                                                    content['content_category']][content['content_no']] \
                                                    ['weighted_request_tracking']) + 0.75 * (
                                                       self.record[content['content_type']][content['content_coord']][
                                                           content['content_category']][content['content_no']][
                                                           'request_tracking'] / den))

                        p = self.record[content['content_type']][content['content_coord']][content['content_category']][
                            content['content_no']] \
                            ['weighted_request_tracking']
                        s = self.record[content['content_type']][content['content_coord']][content['content_category']][
                            content['content_no']]['slot']

                        q = self.sigmoid(self.record[content['content_type']][content['content_coord']][
                                             content['content_category']] \
                                             [content['content_no']]['avg_reward'])

                        self.record[content['content_type']][content['content_coord']][content['content_category']][
                            content['content_no']]['q_value'] = q

                        # Add content and associated Q-value to the action space
                        self.action_space.append({'content': content, 'q_value': q, 'slot': s, 'popularity': p})

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
        elif self.algorithm == "MAB_Contextual":
            return self.select_action_mab_contextual()
        elif self.algorithm == "Federated_MAB":
            return self.select_action_federated_mab()
        elif self.algorithm == "Enhanced_Federated_MAB":
            return self.select_action_enhanced_federated_mab()
        else:
            print(f"  ⚠️ Unknown algorithm {self.algorithm}, defaulting to MAB_Contextual")
            return self.select_action_mab_contextual()

    def _ensure_record_structure(self, entity_type, coord, category, content_no):
        """Ensure complete record structure with ALL required keys"""

        if entity_type not in self.record:
            self.record[entity_type] = {}
        if coord not in self.record[entity_type]:
            self.record[entity_type][coord] = {}
        if category not in self.record[entity_type][coord]:
            self.record[entity_type][coord][category] = {}
        if content_no not in self.record[entity_type][coord][category]:
            # Initialize with ALL required keys
            self.record[entity_type][coord][category][content_no] = {
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


        e = self.record[entity_type][coord][category][content_no]
        # ensure new delay-aware keys exist
        if 'cache_path_reqs' not in e:
            e['cache_path_reqs'] = 0
        if 'cache_hit_delay_sum' not in e:
            e['cache_hit_delay_sum'] = 0.0
        if 'cache_hit_delay_count' not in e:
            e['cache_hit_delay_count'] = 0


    def select_action_lru(self):
        """LRU ALGORITHM: Cache based on recency (slot access time)"""
        remaining_cache_size = self.cache_size
        start_time = time.time()

        if not self.action_space:
            return

        sorted_actions = sorted(self.action_space, key=lambda action: action.get('slot', 0), reverse=True)

        cached_count = 0
        for action in sorted_actions:
            selected_content = action['content']
            content_size = selected_content['size']

            if remaining_cache_size - content_size >= 0.0:
                self._cache_content_helper(selected_content)
                cached_count += 1
                remaining_cache_size -= content_size
            else:
                break

        # Handle items with slot = 0 (never accessed)
        if remaining_cache_size > 0.01:
            zero_slot_actions = [action for action in sorted_actions if action.get('slot', 0) == 0]
            if zero_slot_actions:
                random_action = random.choice(zero_slot_actions)
                selected_content = random_action['content']
                content_size = selected_content['size']

                if remaining_cache_size - content_size >= 0.01:
                    self._cache_content_helper(selected_content)
                    cached_count += 1

        decision_latency = time.time() - start_time
        if decision_latency > 0:
            self.avg_decision_latency += decision_latency
            self.decision_count += 1

        return


    def select_action_popularity(self):
        """POPULARITY ALGORITHM: Cache based on request frequency"""
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

        # Handle items with popularity = 0
        if remaining_cache_size > 0.01:
            zero_popularity_actions = [action for action in sorted_actions if action.get('popularity', 0) == 0]
            if zero_popularity_actions:
                random_action = random.choice(zero_popularity_actions)
                selected_content = random_action['content']
                content_size = selected_content['size']

                if remaining_cache_size - content_size >= 0.01:
                    self._cache_content_helper(selected_content)
                    cached_count += 1

        decision_latency = time.time() - start_time
        if decision_latency > 0:
            self.avg_decision_latency += decision_latency
            self.decision_count += 1

        return


    def select_action_mab_original(self):
        """
        🔍 COMPLETE DEBUG VERSION: Vehicle MAB_Original with detailed logging
        REPLACE your existing select_action_mab_original() in vehicle_ccn.py with this ENTIRE function
        """
        import random
        import time
        import numpy as np

        print(f"\n🚗 VEHICLE DEBUG - {self.vehicle_id}")

        self.clear_cache()
        remaining_cache_size = self.cache_size
        start_time = time.time()

        current_alpha = getattr(self, 'current_alpha', 1.0)
        current_slot = getattr(self, 'current_slot', 1)
        if current_slot <= 0:
            current_slot = 1

        # DEBUG: Basic parameters
        print(f"   🔍 Alpha: {current_alpha:.2f}")
        print(f"   🔍 Action space size: {len(self.action_space)}")
        print(f"   🔍 Cache size: {remaining_cache_size}")
        print(f"   🔍 Current slot: {current_slot}")

        if not self.action_space:
            print("   ❌ NO ACTION SPACE - Vehicle can't cache anything!")
            return

        # 🔥 TUNED PARAMETERS: More aggressive for vehicles
        if current_alpha > 1.5:
            epsilon = 0.05
            cache_threshold = 0.05  # 🔥 LOWERED from 0.25 to 0.15
            popularity_weight = 0.4
            learning_boost = 1.2
            print(f"   🎯 HIGH ALPHA MODE: ε={epsilon}, threshold={cache_threshold}")
        elif current_alpha > 0.8:
            epsilon = 0.15
            cache_threshold = 0.35  # 🔥 LOWERED from 0.40 to 0.35
            popularity_weight = 0.3
            learning_boost = 1.1
            print(f"   ⚖️ MEDIUM ALPHA MODE: ε={epsilon}, threshold={cache_threshold}")
        else:
            epsilon = 0.30
            cache_threshold = 0.55  # 🔥 LOWERED from 0.60 to 0.55
            popularity_weight = 0.2
            learning_boost = 1.0
            print(f"   🔍 LOW ALPHA MODE: ε={epsilon}, threshold={cache_threshold}")

        # Calculate UCB scores for all actions
        max_ucb = 0
        min_ucb = float('inf')
        qualifying_actions = 0
        total_popularity = 0
        total_q_values = 0

        for action in self.action_space:
            content = action['content']
            s_type = content['content_type']
            s_coord = content['content_coord']
            s_category = content['content_category']
            s_no = content['content_no']

            # Safe record access with defaults
            record_entry = self.record.get(s_type, {}).get(s_coord, {}).get(s_category, {}).get(s_no, {})
            selection_count = record_entry.get('no_f_time_cached', 0)
            hit_count = record_entry.get('content_hit', 0)

            # Q-value calculation (hit rate when cached)
            if selection_count > 0:
                base_q_value = hit_count / selection_count
            else:
                base_q_value = 0.0

            popularity_score = action.get('popularity', 0)
            total_popularity += popularity_score
            total_q_values += base_q_value

            # 🔥 ALPHA-ADAPTIVE Q-VALUE ENHANCEMENT
            if current_alpha > 1.5 and popularity_score > 0.3:
                # High concentration: boost popular content significantly
                enhanced_q_value = base_q_value * learning_boost + (popularity_weight * popularity_score)
            elif current_alpha > 0.8 and popularity_score > 0.1:
                # Medium concentration: moderate boost
                enhanced_q_value = base_q_value * learning_boost + (popularity_weight * popularity_score)
            else:
                # Low concentration: rely mainly on experience
                enhanced_q_value = base_q_value + (popularity_weight * popularity_score)

            # UCB exploration component
            if selection_count > 0 and current_slot > 1:
                exploration = np.sqrt((2 * np.log(current_slot)) / selection_count)
            else:
                exploration = 10.0  # High exploration for unselected items

            # 🔥 ALPHA-AWARE UCB CALCULATION
            if current_alpha > 1.5:
                # High concentration: reduce exploration, focus on exploitation
                ucb_score = enhanced_q_value + (0.1 * exploration)
            elif current_alpha > 0.8:
                # Medium concentration: balanced exploration
                ucb_score = enhanced_q_value + (0.3 * exploration)
            else:
                # Low concentration: high exploration
                ucb_score = enhanced_q_value + (0.5 * exploration)

            action['ucb_score'] = ucb_score
            action['enhanced_q'] = enhanced_q_value
            action['base_q'] = base_q_value
            action['exploration'] = exploration

            # Track statistics
            max_ucb = max(max_ucb, ucb_score)
            min_ucb = min(min_ucb, ucb_score)

            if ucb_score > cache_threshold:
                qualifying_actions += 1

        # 🔍 DETAILED STATISTICS
        avg_popularity = total_popularity / len(self.action_space) if self.action_space else 0
        avg_q_value = total_q_values / len(self.action_space) if self.action_space else 0

        print(f"   📊 UCB Score Stats:")
        print(f"      Max UCB: {max_ucb:.3f}")
        print(f"      Min UCB: {min_ucb:.3f}")
        print(f"      Cache threshold: {cache_threshold:.3f}")
        print(f"      Actions above threshold: {qualifying_actions}/{len(self.action_space)}")
        print(f"   📊 Content Stats:")
        print(f"      Avg popularity: {avg_popularity:.3f}")
        print(f"      Avg Q-value: {avg_q_value:.3f}")

        # Show top 3 actions for debugging
        sorted_actions = sorted(self.action_space, key=lambda x: x.get('ucb_score', 0), reverse=True)
        print(f"   🏆 Top 3 Actions:")
        for i, action in enumerate(sorted_actions[:3]):
            content = action['content']
            ucb = action.get('ucb_score', 0)
            q_val = action.get('base_q', 0)
            pop = action.get('popularity', 0)
            print(
                f"      {i + 1}. {content['content_type']}_{content['content_coord']}_{content['content_category']}_{content['content_no']}")
            print(f"         UCB: {ucb:.3f}, Q: {q_val:.3f}, Pop: {pop:.3f}")

        if max_ucb <= cache_threshold:
            print(f"   ❌ CRITICAL PROBLEM: No actions meet cache threshold!")
            print(f"   💡 SUGGESTION: Lower threshold from {cache_threshold:.3f} to {max_ucb * 0.9:.3f}")
            print(f"   🔧 Or increase learning boost/popularity weight")

            # Show what threshold would work
            sorted_scores = sorted([a.get('ucb_score', 0) for a in self.action_space], reverse=True)
            threshold_10pct = sorted_scores[
                min(len(sorted_scores) // 10, len(sorted_scores) - 1)] if sorted_scores else 0
            print(f"   📈 Threshold for top 10%: {threshold_10pct:.3f}")

            # Track decision latency and return
            decision_latency = time.time() - start_time
            if decision_latency > 0:
                self.avg_decision_latency += decision_latency
                self.decision_count += 1
            return

        # 🎯 EPSILON-GREEDY WITH ALPHA ADAPTATION
        use_exploration = random.random() < epsilon
        print(f"   🎲 Decision: {'EXPLORATION' if use_exploration else 'EXPLOITATION'} (ε={epsilon:.2f})")

        if use_exploration:
            selected_actions = random.sample(self.action_space, min(len(self.action_space), 10))
            print(f"   🔍 Selected {len(selected_actions)} random actions for exploration")
        else:
            selected_actions = sorted_actions
            print(f"   🎯 Using all {len(selected_actions)} actions sorted by UCB")

        # Detailed caching attempts with alpha-aware decisions
        cached_count = 0
        cache_attempts = 0
        cache_decisions = []
        total_cache_score = 0

        for i, action in enumerate(selected_actions):
            selected_content = action['content']
            content_size = selected_content['size']
            ucb_score = action.get('ucb_score', 0)
            enhanced_q = action.get('enhanced_q', 0)
            popularity = action.get('popularity', 0)

            # 🔥 ALPHA-ADAPTIVE CACHING DECISION
            should_cache = False
            decision_reason = ""

            if current_alpha > 1.5:
                # High concentration: cache if UCB > threshold OR popularity > 0.3
                should_cache = (ucb_score > cache_threshold) or (popularity > 0.3)
                decision_reason = f"High α: UCB>{cache_threshold:.2f} OR Pop>0.3"
            elif current_alpha > 0.8:
                # Medium concentration: balanced decision
                should_cache = ucb_score > cache_threshold
                decision_reason = f"Med α: UCB>{cache_threshold:.2f}"
            else:
                # Low concentration: conservative, only cache high-confidence items
                should_cache = ucb_score > cache_threshold and enhanced_q > 0.1
                decision_reason = f"Low α: UCB>{cache_threshold:.2f} AND Q>0.1"

            content_id = f"{selected_content['content_type']}_{selected_content['content_coord']}_{selected_content['content_category']}_{selected_content['content_no']}"

            # Cache if decision criteria met and space available
            if should_cache and remaining_cache_size - content_size >= 0.01:
                cache_attempts += 1
                try:
                    # Use vehicle's existing cache method
                    self._cache_content_helper(selected_content)
                    cached_count += 1
                    remaining_cache_size -= content_size
                    total_cache_score += ucb_score

                    # 🔥 UPDATE LEARNING: Ensure record structure exists
                    s_type = selected_content['content_type']
                    s_coord = selected_content['content_coord']
                    s_category = selected_content['content_category']
                    s_no = selected_content['content_no']

                    # Build record structure if missing
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
                            'total_requests': 0,
                            'cache_path_reqs': 0,
                            'cache_hit_delay_sum': 0.0,
                            'cache_hit_delay_count': 0,
                        }

                    # Increment selection count (critical for learning)
                    self.record[s_type][s_coord][s_category][s_no]['no_f_time_cached'] += 1

                    cache_decisions.append(f"✅ CACHED: {content_id} (UCB: {ucb_score:.3f})")

                except Exception as e:
                    cache_decisions.append(f"❌ CACHE FAILED: {content_id} - {str(e)}")

            elif not should_cache:
                reason = f"Failed: {decision_reason}"
                cache_decisions.append(f"⏭️ SKIPPED: {content_id} (UCB: {ucb_score:.3f}) - {reason}")
            elif remaining_cache_size - content_size < 0.01:
                cache_decisions.append(
                    f"💾 NO SPACE: {content_id} - Need {content_size:.2f}, have {remaining_cache_size:.2f}")
                break

            # Limit logging to first 10 decisions to avoid spam
            if i >= 10:
                if len(cache_decisions) == 10:
                    cache_decisions.append(f"... (showing first 10/{len(selected_actions)} decisions)")
                break

        # Print caching decisions for debugging
        for decision in cache_decisions:
            print(f"   {decision}")

        # Final summary statistics
        avg_cache_score = total_cache_score / max(cached_count, 1)
        cache_ratio = (cached_count / len(self.action_space)) * 100 if self.action_space else 0

        print(f"   📦 VEHICLE SUMMARY:")
        print(f"      Cached: {cached_count}/{cache_attempts} attempts ({len(self.action_space)} total actions)")
        print(f"      Cache ratio: {cache_ratio:.2f}%")
        print(f"      Avg cached UCB: {avg_cache_score:.3f}")
        print(f"      Remaining space: {remaining_cache_size:.2f}/{self.cache_size}")

        # Track decision latency (keep existing pattern)
        decision_latency = time.time() - start_time
        if decision_latency > 0:
            self.avg_decision_latency += decision_latency
            self.decision_count += 1

        return

    def _get_theta_bound_ms_for_content(self, content_type, content_category, default_theta_ms=2000.0):
        # Fallbacks per manuscript Table III
        t = str(content_type).lower()
        theta_sat_ms = 10_000.0
        theta_air_ms = 2_000.0
        theta_terr_ms = 5_000.0
        if "sat" in t:
            return theta_sat_ms
        if "uav" in t or "aerial" in t or "air" in t:
            return theta_air_ms
        if "veh" in t or "bs" in t or "ter" in t or "grid" in t or "ground" in t:
            return theta_terr_ms
        return float(default_theta_ms)

    def _get_popularity_index(self, entry, content_type, content_coord, content_category, content_no):
        reqs = int(entry.get("request_tracking", 0))
        return reqs / float(reqs + 50)

    def _get_freshness_ratio(self, content_type, content_coord, content_category, content_no):
        try:
            meta = self.content_cache.get(content_type, {}).get(content_coord, {}).get(content_category, [])
            # simple heuristic: recently cached -> fresh
            return 1.0 if meta else 0.8
        except Exception:
            return 1.0

    def _get_cache_size_ratio(self, content_type, content_category, content_no):
        cat = str(content_category).lower()
        size_map = {"1": 10.0, "i": 10.0, "2": 1.0, "ii": 1.0, "3": 0.1, "iii": 0.1, "4": 2.0, "iv": 2.0}
        size = size_map.get(cat, 1.0)
        cap = float(getattr(self, 'cache_size', 50.0))
        return min(1.0, size / max(1e-6, cap))


    def select_action_mab_contextual(self):
        """
        FIXED VEHICLE: Contextual MAB without UCB dominance
        """
        print(f"\n🧠 FIXED VEHICLE CONTEXTUAL MAB - {self.vehicle_id}")

        remaining_cache_size = self.cache_size
        start_time = time.time()

        if not self.action_space:
            print("   ❌ No action space!")
            return

        current_slot = getattr(self, 'current_slot', 1)
        if current_slot <= 0:
            current_slot = 1

        print(f"   📊 Slot: {current_slot}, Actions: {len(self.action_space)}")

        # Track already cached content
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

            # Get record entry
            record_entry = self.record.get(s_type, {}).get(s_coord, {}).get(s_category, {}).get(s_no, {})
            selection_count = record_entry.get('no_f_time_cached', 0)
            hit_count = record_entry.get('content_hit', 0)
            request_count = record_entry.get('request_tracking', 0)

            # Base Q-value
            if selection_count > 0:
                local_q_value = hit_count / selection_count
            else:
                local_q_value = 0.0

            # CONTEXTUAL FEATURES FOR VEHICLES (mobility-focused)
            # 1. Popularity Index
            total_requests = max(1, sum(
                self.record.get(et, {}).get(ec, {}).get(ecat, {}).get(eno, {}).get('request_tracking', 0)
                for et in self.record for ec in self.record[et]
                for ecat in self.record[et][ec] for eno in self.record[et][ec][ecat]
            ))
            popularity_index = request_count / total_requests

            # 2. Freshness Index (important for vehicles due to mobility)
            content_validity = content.get('content_validity', 3600)
            generation_time = content.get('generation_time', 0)
            current_time = current_slot * 60
            time_elapsed = max(0, current_time - generation_time)
            remaining_lifetime = max(0, content_validity - time_elapsed)
            freshness_index = remaining_lifetime / content_validity if content_validity > 0 else 0.0

            # 3. Cache Efficiency (important for vehicles with limited cache)
            content_size = content.get('size', 1.0)
            cache_efficiency = min(1.0, self.cache_size / (content_size * 10.0))

            # Context vector
            context_vector = np.array([popularity_index, freshness_index, cache_efficiency])

            # Vehicle-specific weights (popularity-focused due to mobility)
            lambda_weights = np.array([0.4, 0.4, 0.2])  # Prioritize popular content for vehicles

            # FIXED: Small contextual enhancement (NOT UCB dominance)
            contextual_bonus = np.dot(lambda_weights, context_vector) * 0.15  # Scale factor

            # Small exploration for new content
            exploration_bonus = 0.1 if selection_count == 0 else 0.0

            # FINAL: Contextual Q-value
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

        # Sort by contextual Q-value
        sorted_actions = sorted(self.action_space, key=lambda x: x.get('contextual_q_value', 0), reverse=True)

        print(f"🏆 Top 5 by Contextual Q-value:")
        for i, action in enumerate(sorted_actions[:5]):
            content = action['content']
            cq = action.get('contextual_q_value', 0)
            bq = action.get('base_q_value', 0)
            content_id = f"{content['content_type']}_{content['content_coord']}_{content['content_category']}_{content['content_no']}"
            print(f"   {i + 1}. {content_id}: CQ={cq:.4f} (Base={bq:.4f})")

        # Cache content
        cached_count = 0
        cache_decisions = []

        for i, action in enumerate(sorted_actions):
            selected_content = action['content']
            content_size = selected_content['size']
            content_id = f"{selected_content['content_type']}_{selected_content['content_coord']}_{selected_content['content_category']}_{selected_content['content_no']}"

            if content_id in already_cached:
                if i < 10:
                    cache_decisions.append(f"⏭️  SKIP: {content_id} - Already cached")
                continue

            if remaining_cache_size - content_size >= 0.01:
                self._cache_content_helper(selected_content)
                already_cached.add(content_id)
                cached_count += 1
                remaining_cache_size -= content_size

                if i < 10:
                    cq = action.get('contextual_q_value', 0)
                    cache_decisions.append(f"✅ CACHED: {content_id} (CQ: {cq:.4f})")
            else:
                if i < 10:
                    cache_decisions.append(f"💾 NO SPACE: {content_id}")
                break

        # Print decisions
        for decision in cache_decisions:
            print(f"   {decision}")

        print(f"📦 Fixed Vehicle Contextual MAB cached {cached_count} items")

        # Decision latency tracking
        decision_latency = time.time() - start_time
        if decision_latency > 0:
            self.avg_decision_latency += decision_latency
            self.decision_count += 1

        return


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

        time_spent = 0  # Calculate based on current time if available
        content_validity = content.get('content_validity', 1800)

        return self.sigmoid(
            avg_reward +
            (self.cache_size / content['size']) +
            ((content_validity - time_spent) / content_validity) +
            weighted_request_tracking
        )


    def debug_vehicle_performance(self, slot):
        """Debug vehicle performance every 20 slots"""
        if slot % 20 == 0:
            cache_hit_ratio = self.cache_hit / max(1, self.total_request_for_caching)
            source_hit_ratio = self.source_hit / max(1, self.total_request)
            overall_hit_ratio = (self.cache_hit + self.source_hit) / max(1, self.total_request)

            print(f"\n🚗 Vehicle {self.vehicle_id} - Slot {slot} Performance:")
            print(f"   Total requests: {self.total_request}")
            print(f"   Cache requests: {self.total_request_for_caching}")
            print(f"   Cache hits: {self.cache_hit} ({cache_hit_ratio:.3f})")
            print(f"   Source hits: {self.source_hit} ({source_hit_ratio:.3f})")
            print(f"   Overall hit ratio: {overall_hit_ratio:.3f} ({overall_hit_ratio * 100:.1f}%)")

            # Show learning progress
            total_q_value = 0
            content_with_q = 0

            for entity_type, coord_dict in self.record.items():
                for coord, category_dict in coord_dict.items():
                    for category, content_no_dict in category_dict.items():
                        for content_no, content_info in content_no_dict.items():
                            q_val = content_info.get('q_value', 0)
                            if q_val > 0:
                                total_q_value += q_val
                                content_with_q += 1

            avg_q = total_q_value / max(1, content_with_q)
            print(f"   Learning: {content_with_q} content items with Q > 0, avg Q = {avg_q:.4f}")


    def get_reward(self):
        """Unified reward processing - calls algorithm-specific reward function"""
        if self.algorithm == "LRU":
            return self.get_reward_lru()
        elif self.algorithm == "Popularity":
            return self.get_reward_popularity()
        elif self.algorithm == "MAB_Original":
            return self.get_reward_mab_original()
        elif self.algorithm == "MAB_Contextual":
            return self.get_reward_mab_contextual()
        elif self.algorithm == "Federated_MAB":
            return self.get_reward_federated_mab()
        else:
            return self.get_reward_mab_original()  # Default

        # ===============================
        # 2. ADD ALGORITHM-SPECIFIC REWARD METHODS
        # ===============================


    def get_reward_lru(self):
        """LRU doesn't use learning rewards"""
        pass


    def get_reward_popularity(self):
        """Popularity-based reward tracking"""
        for entity_type, coord_dict in self.record.items():
            for coord, category_dict in coord_dict.items():
                for category, content_no_dict in category_dict.items():
                    for content_no, content_info in content_no_dict.items():
                        requests = content_info.get('request_tracking', 0)
                        if requests > 0:
                            total_requests = sum(
                                self.record.get(et, {}).get(ec, {}).get(ecat, {}).get(eno, {}).get('request_tracking',
                                                                                                   0)
                                for et in self.record for ec in self.record[et]
                                for ecat in self.record[et][ec] for eno in self.record[et][ec][ecat]
                            )
                            popularity_score = requests / max(1, total_requests)
                            content_info['popularity_score'] = popularity_score


    def get_reward_mab_original(self):
        """Enhanced reward function for better learning"""

        print(f"\n🎁 Vehicle {self.vehicle_id} - Processing Rewards")

        total_rewards_calculated = 0

        for entity_type, coord_dict in self.record.items():
            for coord, category_dict in coord_dict.items():
                for category, content_no_dict in category_dict.items():
                    for content_no, content_info in content_no_dict.items():

                        # Check if content is currently in cache
                        in_cache = self.is_content_in_cache(coord, category, content_no)

                        if in_cache:
                            hits = content_info.get('content_hit', 0)
                            times_cached = content_info.get('no_f_time_cached', 0)

                            if hits > 0 and times_cached > 0:
                                # Calculate reward: hit rate when cached
                                reward = hits / times_cached

                                # Update average reward with learning rate
                                old_avg = content_info.get('avg_reward', 0)
                                learning_rate = 0.1
                                new_avg = old_avg + learning_rate * (reward - old_avg)
                                content_info['avg_reward'] = new_avg

                                # Update Q-value
                                new_q = min(1.0, max(0.0, new_avg))  # Keep Q-value in [0,1]
                                content_info['q_value'] = new_q

                                total_rewards_calculated += 1

                                print(f"   📊 {entity_type}_{coord}_{category}_{content_no}:")
                                print(f"      Hits: {hits}, Times cached: {times_cached}")
                                print(f"      Reward: {reward:.3f}, New Q: {new_q:.4f}")

                        # Reset hit counter for next period
                        content_info['content_hit'] = 0

        print(f"   ✅ Processed rewards for {total_rewards_calculated} content items")

        return


    def get_reward_mab_contextual(self):
        """Delay-aware Contextual MAB reward for Vehicles (compatible with UAV/BS style)"""
        lr = getattr(self, 'learning_rate', 0.1)
        cap = getattr(self, 'delay_ratio_cap', 5.0)
        theta_default = getattr(self, 'theta_ms_default', 5000.0)  # terrestrial bound as fallback
        w_base = getattr(self, 'w_base', 1.0)
        w_ctx = getattr(self, 'w_ctx', 0.3)
        w_beta = getattr(self, 'w_beta', 1.0)
        w_rho = getattr(self, 'w_rho', 1.0)
        w_s = getattr(self, 'w_s', 1.0)
        current_slot = getattr(self, 'current_slot', 1)

        for ctype, d1 in self.record.items():
            for ccoord, d2 in d1.items():
                for ccat, d3 in d2.items():
                    for cno, e in d3.items():
                        attempts = int(e.get('cache_path_reqs', 0))
                        hits = int(e.get('content_hit', 0))
                        CH = (hits / float(max(1, attempts))) if attempts > 0 else 0.0

                        dsum = float(e.get('cache_hit_delay_sum', 0.0))
                        dcnt = int(e.get('cache_hit_delay_count', 0))
                        avg_ms = (dsum / dcnt) if dcnt > 0 else theta_default

                        theta = self._get_theta_bound_ms_for_content(ctype, ccat, default_theta_ms=theta_default)
                        delay_term = min(theta / max(1.0, avg_ms), cap)
                        base = CH + delay_term

                        # contextual nudges
                        reqs = int(e.get('request_tracking', 0))
                        total_reqs = max(1, sum(
                            self.record.get(et, {}).get(ec, {}).get(ecat, {}).get(eno, {}).get('request_tracking', 0)
                            for et in self.record for ec in self.record[et]
                            for ecat in self.record[et][ec] for eno in self.record[et][ec][ecat]))
                        beta = reqs / total_reqs

                        age = max(0, current_slot - int(e.get('slot', current_slot)))
                        rho = max(0.0, min(1.0, 1.0 - (age / 200.0)))

                        s = self._get_cache_size_ratio(ctype, ccat, cno)
                        s_norm = min(1.0, s)

                        reward = (w_base * base) + (w_ctx * ((w_beta * beta) + (w_rho * rho) + (w_s * s_norm)))

                        old = float(e.get('avg_reward', 0.0))
                        new = old + lr * (reward - old)
                        e['avg_reward'] = new
                        e['q_value'] = new


    def get_reward_federated_mab(self):
        """NEW: Federated MAB reward"""
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

                                # Weighted combination (vehicles rely more on local knowledge)
                                alpha = 0.7  # Higher local weight for vehicles
                                federated_q = alpha * local_q + (1 - alpha) * global_q
                                content_info['q_value'] = federated_q
                            except:
                                pass


    def is_content_in_cache(self, coord, category, content_no):
        """Check if specific content is in the cache"""
        try:
            if coord in self.content_cache:
                if category in self.content_cache[coord]:
                    for cached_content in self.content_cache[coord][category]:
                        if cached_content.get('content_no') == content_no:
                            return True
            return False
        except Exception as e:
            return False


    def clear_cache(self):
        """Clear the existing content cache"""
        self.content_cache = {}


    def send_update_to_federated_server(self):
        """Sends the vehicle's Q-values to the federated server."""
        if hasattr(self, 'aggregator') and self.aggregator:
            self.aggregator.receive_update(self.vehicle_id, self.record)

    def get_enhanced_statistics(self):
        """
        Enhanced statistics for performance-based aggregation
        """
        if not hasattr(self, 'total_requests_tracked'):
            self.total_requests_tracked = 0
            self.cache_hits_tracked = 0

        # Calculate hit ratio
        hit_ratio = (self.cache_hit / max(1, self.request_receive_cache ))

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
            'total_requests': self.request_receive_cache ,
            'cache_hits': self.cache_hit,
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

    def update_enhanced_tracking(self):
        """OPTIONAL: Update enhanced tracking if enabled"""
        if hasattr(self, 'aggregator') and hasattr(self.aggregator, 'update_node_statistics'):
            stats = self.get_enhanced_statistics()
            self.aggregator.update_node_statistics(self.vehicle_id, stats)


    def run(self, current_time, slot, time_slots, vehicles, uavs, base_stations, satellites,
            communication, no_of_request_genertaed_in_each_timeslot, no_of_content_each_category):
        # Set current slot for UCB calculation
        self.current_slot = slot

        # Existing cache management
        self.cache_cleanup(current_time)
        if ((slot - 1) % 10 == 0):
            if slot > 10:
                # 🔧 FIX: Use federated_update based on algorithm
                use_federated = self.algorithm in ["Federated_MAB", "Enhanced_Federated_MAB"]
                self.append_action_space(slot, federated_update=use_federated)  # 🟢 FIXED!

                if len(self.action_space):
                    self.select_action()  # This will call the right algorithm
                    self.action_space = []
                self.send_update_to_federated_server()
            if slot > 20:
                self.get_reward()  # Process rewards for learning

        # Movement and request generation
        self.move()

        for _ in range(no_of_request_genertaed_in_each_timeslot):
            content_request = communication.send_content_request(
                self, vehicles, uavs, base_stations, satellites,
                self.grid_size, current_time, slot,
                no_of_content_each_category,
                current_zipf=communication.get_st_zipf(self, slot)
            )
            if (content_request['unique_id'] != 0):
                self.active_requests.add(content_request['unique_id'])
                communication.content_request_queue.put(content_request)

        # 🎯 ADD: Debug performance tracking
        self.debug_vehicle_performance(slot)


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