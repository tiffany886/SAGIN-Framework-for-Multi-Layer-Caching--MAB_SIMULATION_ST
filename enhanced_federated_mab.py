import numpy as np
from federated_mab import FederatedAggregator  # ADD THIS IMPORT!



class EnhancedFederatedAggregator(FederatedAggregator):
    """
    Enhanced federated aggregator with weighted aggregation based on node performance
    Works with content types: 'satellite', 'UAV', 'grid'
    """

    def __init__(self):
        super().__init__()
        self.node_performance = {}  # Track performance metrics per node

    def get_neighbor_q_values(self, node_id, content_type, content_coord, content_category, content_no):
        content_coord_str = str(content_coord)
        exact_key = f"{content_type}_{content_coord_str}_{content_category}_{content_no}"
        sem_key = f"{content_type}_*_{content_category}_{content_no}"

        vals = []
        if exact_key in self.global_q_values:
            vals = [float(q) for nid, q in self.global_q_values[exact_key].items() if nid != node_id]

        if not vals and sem_key in self.global_q_values:
            vals = [float(q) for nid, q in self.global_q_values[sem_key].items() if nid != node_id]

        return vals

    def receive_update(self, node_id, updates):
        if not hasattr(self, "global_q_values"):
            self.global_q_values = {}
        if not hasattr(self, "node_updates"):
            self.node_updates = {}
        if node_id not in self.node_updates:
            self.node_updates[node_id] = {}

        for content_key, metrics in updates.items():
            q_val = float(metrics.get("q_value", 0.5))
            self.global_q_values.setdefault(content_key, {})[node_id] = q_val
            self.node_updates[node_id][content_key] = metrics

            parts = content_key.split("_", 3)
            if len(parts) == 4:
                t, coord, cat, no = parts
                sem_key = f"{t}_*_{cat}_{no}"
                self.global_q_values.setdefault(sem_key, {})[node_id] = q_val
                self.node_updates[node_id][sem_key] = metrics

    def get_enhanced_neighbor_values(self, node_id, content_type, content_coord, content_category, content_no):
        """
        Get both Q-values and weights from neighbors (for enhanced version)

        Returns:
            Dictionary with 'q_values' and 'weights' lists, or None if no neighbors
        """

        content_coord_str = str(content_coord)
        content_key = f"{content_type}_{content_coord_str}_{content_category}_{content_no}"

        q_values = []
        weights = []

        if content_key in self.global_q_values:
            for other_node_id, q_value in self.global_q_values[content_key].items():
                if other_node_id != node_id:
                    q_values.append(q_value)

                    # Calculate weight based on node's hit ratio
                    if other_node_id in self.node_updates:
                        total_hits = sum(
                            metrics.get('content_hit', 0)
                            for metrics in self.node_updates[other_node_id].values()
                        )
                        total_requests = sum(
                            metrics.get('request_count', 1)
                            for metrics in self.node_updates[other_node_id].values()
                        )
                        hit_ratio = total_hits / max(total_requests, 1)
                        weight = 0.5 + hit_ratio  # Weight between 0.5 and 1.5
                    else:
                        weight = 1.0

                    weights.append(weight)

        if q_values:
            return {
                'q_values': q_values,
                'weights': weights
            }

        return None

    def aggregate_updates(self):
        """
        Enhanced aggregation with performance-based weighting
        Called periodically (every 10 slots)
        """

        if not self.global_q_values:
            return {}

        # Update node performance metrics
        for node_id, updates in self.node_updates.items():
            total_hits = sum(m.get('content_hit', 0) for m in updates.values())
            total_requests = sum(m.get('request_count', 1) for m in updates.values())
            hit_ratio = total_hits / max(total_requests, 1)

            self.node_performance[node_id] = {
                'hit_ratio': hit_ratio,
                'weight': 0.5 + hit_ratio,  # Weight between 0.5 and 1.5
                'total_hits': total_hits,
                'total_requests': total_requests
            }

        # Perform weighted aggregation
        aggregated_values = {}

        for content_key, node_values in self.global_q_values.items():
            if node_values:
                weighted_sum = 0
                total_weight = 0

                for node_id, q_value in node_values.items():
                    weight = self.node_performance.get(node_id, {}).get('weight', 1.0)
                    weighted_sum += q_value * weight
                    total_weight += weight

                if total_weight > 0:
                    aggregated_values[content_key] = weighted_sum / total_weight

                    # Update with weighted average (30% global, 70% local)
                    for node_id in node_values:
                        node_values[node_id] = (
                                aggregated_values[content_key] * 0.3 +
                                node_values[node_id] * 0.7
                        )

        # print(f"🔄 Enhanced Aggregator: Weighted aggregation complete")

        return aggregated_values

    def update_node_statistics(self, node_id, statistics):
        """
        Update enhanced statistics for a node
        """
        if 'node_statistics' not in dir(self):
            self.node_statistics = {}

        self.node_statistics[node_id] = statistics

        # Update node performance based on enhanced stats
        hit_ratio = statistics.get('hit_ratio', 0.0)
        trust_factor = statistics.get('trust_factor', 0.5)
        stability_score = statistics.get('stability_score', 0.5)

        # Enhanced weight calculation considering multiple factors
        base_weight = 0.5 + hit_ratio  # 0.5 to 1.5 based on hit ratio
        trust_bonus = trust_factor * 0.3  # Up to 0.3 bonus from trust
        stability_bonus = stability_score * 0.2  # Up to 0.2 bonus from stability

        enhanced_weight = min(2.0, base_weight + trust_bonus + stability_bonus)

        self.node_performance[node_id] = {
            'hit_ratio': hit_ratio,
            'weight': enhanced_weight,
            'trust_factor': trust_factor,
            'stability_score': stability_score,
            'total_hits': statistics.get('cache_hits', 0),
            'total_requests': statistics.get('total_requests', 1),
            'last_update': statistics.get('last_update_time', 0)
        }

    def get_enhanced_neighbor_info(self, node_id, content_type, content_coord, content_category, content_no):
        """
        Alias for get_enhanced_neighbor_values for backward compatibility
        """
        return self.get_enhanced_neighbor_values(node_id, content_type, content_coord, content_category, content_no)