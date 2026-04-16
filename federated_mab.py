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
    Also write-through to a coord-less 'semantic key' so knowledge
    can transfer across different coordinates for the same content.
    """
    if not hasattr(self, "global_q_values"):
        self.global_q_values = {}
    if not hasattr(self, "node_updates"):
        self.node_updates = {}

    if node_id not in self.node_updates:
        self.node_updates[node_id] = {}

    for content_key, metrics in updates.items():
        # Base store under exact (type_coord_category_no)
        q_val = float(metrics.get("q_value", 0.5))
        self.global_q_values.setdefault(content_key, {})[node_id] = q_val
        self.node_updates[node_id][content_key] = metrics

        # ALSO store under coord-less semantic key: type_*_category_no
        parts = content_key.split("_", 3)
        if len(parts) == 4:
            t, coord, cat, no = parts
            sem_key = f"{t}_*_{cat}_{no}"
            self.global_q_values.setdefault(sem_key, {})[node_id] = q_val
            self.node_updates[node_id][sem_key] = metrics


def get_global_q_value(self, content_type, content_coord, content_category, content_no):
        """
        Get the global aggregated Q-value for specific content

        Args:
            content_type: 'satellite', 'UAV', or 'grid'
            content_coord: Content coordinate/ID
            content_category: 'I', 'II', 'III', or 'IV'
            content_no: Content number
        """

        content_coord_str = str(content_coord)
        content_key = f"{content_type}_{content_coord_str}_{content_category}_{content_no}"

        if content_key in self.global_q_values:
            values = list(self.global_q_values[content_key].values())
            if values:
                return sum(values) / len(values)  # Return average

        return 0.5  # Default Q-value

def reset(self):
        """Reset aggregator state"""
        self.global_q_values = {}
        self.node_updates = {}
        self.update_count = 0
