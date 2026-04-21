import numpy as np


class FederatedAggregator:
    """
    Federated aggregator for coordinating Q-values across UAVs, BSs, and Vehicles
    Works with content types: 'satellite', 'UAV', 'grid'
    """

    def __init__(self):
        """Initialize the federated aggregator"""
        self.global_q_values = {}  # Stores Q-values from all nodes
        self.node_updates = {}  # Track updates from each node
        self.update_count = 0

    def _flatten_updates(self, updates):
        """Flatten nested node record structure to {content_key: {'q_value': ...}}."""
        flat = {}
        if not isinstance(updates, dict):
            return flat

        for content_type, coord_dict in updates.items():
            if not isinstance(coord_dict, dict):
                continue
            for coord, category_dict in coord_dict.items():
                if not isinstance(category_dict, dict):
                    continue
                for category, content_no_dict in category_dict.items():
                    if not isinstance(content_no_dict, dict):
                        continue
                    for content_no, metrics in content_no_dict.items():
                        if not isinstance(metrics, dict):
                            continue
                        key = f"{content_type}_{coord}_{category}_{content_no}"
                        flat[key] = {
                            "q_value": float(metrics.get("q_value", 0.5)),
                            "content_hit": float(metrics.get("content_hit", 0.0)),
                            "request_count": float(metrics.get("request_tracking", 0.0)),
                        }
        return flat

    def get_neighbor_q_values(self, node_id, content_type, content_coord, content_category, content_no):
        """
        Return neighbor Q-values for the specific content.
        Prefer exact (type_coord_category_no); if none, fall back to coord-less
        semantic key (type_*_category_no). Excludes the requesting node_id.
        """
        content_coord_str = str(content_coord)
        exact_key = f"{content_type}_{content_coord_str}_{content_category}_{content_no}"
        sem_key = f"{content_type}_*_{content_category}_{content_no}"

        # First: exact-coordinate sharing
        vals = []
        if exact_key in self.global_q_values:
            vals = [float(q) for nid, q in self.global_q_values[exact_key].items() if nid != node_id]

        # Fallback: coord-less semantic sharing if exact is empty
        if not vals and sem_key in self.global_q_values:
            vals = [float(q) for nid, q in self.global_q_values[sem_key].items() if nid != node_id]

        return vals

    def receive_update(self, node_id, updates):
        """
        Store per-node Q updates.
        Supports nested node record dict and flat key-value dict.
        """
        if not hasattr(self, "global_q_values"):
            self.global_q_values = {}
        if not hasattr(self, "node_updates"):
            self.node_updates = {}

        if node_id not in self.node_updates:
            self.node_updates[node_id] = {}

        flat_updates = self._flatten_updates(updates)
        # If caller already sends flat dict, keep compatibility.
        if not flat_updates and isinstance(updates, dict):
            for content_key, metrics in updates.items():
                q_val = 0.5
                if isinstance(metrics, dict):
                    q_val = float(metrics.get("q_value", 0.5))
                flat_updates[content_key] = {"q_value": q_val}

        for content_key, metrics in flat_updates.items():
            q_val = float(metrics.get("q_value", 0.5))
            self.global_q_values.setdefault(content_key, {})[node_id] = q_val
            self.node_updates[node_id][content_key] = metrics

            # coord-less semantic key for cross-coordinate transfer
            parts = content_key.split("_", 3)
            if len(parts) == 4:
                t, _coord, cat, no = parts
                sem_key = f"{t}_*_{cat}_{no}"
                self.global_q_values.setdefault(sem_key, {})[node_id] = q_val
                self.node_updates[node_id][sem_key] = metrics

    def get_global_q_value(self, *args):
        """
        Backward-compatible API:
        - get_global_q_value(content_key)
        - get_global_q_value(content_type, content_coord, content_category, content_no)
        """
        if len(args) == 1:
            content_key = str(args[0])
        elif len(args) == 4:
            content_type, content_coord, content_category, content_no = args
            content_key = f"{content_type}_{content_coord}_{content_category}_{content_no}"
        else:
            return 0.5

        if content_key in self.global_q_values:
            values = list(self.global_q_values[content_key].values())
            if values:
                return sum(values) / len(values)  # Return average

        return 0.5  # Default Q-value

    def aggregate_updates(self):
        """
        Basic federated averaging across nodes per content key.
        Keeps local personalization via EMA blend.
        """
        if not self.global_q_values:
            return {}

        aggregated_values = {}
        for content_key, node_values in self.global_q_values.items():
            if not node_values:
                continue

            avg_q = float(np.mean(list(node_values.values())))
            aggregated_values[content_key] = avg_q

            # 70% local + 30% global to reduce oscillation.
            for node_id, local_q in node_values.items():
                node_values[node_id] = 0.7 * float(local_q) + 0.3 * avg_q

        self.update_count += 1
        return aggregated_values

    def reset(self):
        """Reset aggregator state"""
        self.global_q_values = {}
        self.node_updates = {}
        self.update_count = 0
