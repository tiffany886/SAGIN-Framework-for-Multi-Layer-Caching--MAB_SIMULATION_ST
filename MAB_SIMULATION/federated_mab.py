import numpy as np

# 联邦MAB聚合器基类
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
        Get Q-values from neighbor nodes for specific content
        THIS IS THE METHOD THAT WAS MISSING!

        Args:
            node_id: ID of requesting node (e.g., 'UAV1', 'BS1', 'Vehicle1')
            content_type: Type of content ('satellite', 'UAV', 'grid')
            content_coord: Coordinate/ID of content source
                - For satellite: 'Satellite1', 'Satellite2', etc.
                - For UAV: integer (1, 2, 3...) or string
                - For grid: grid coordinates like '(5,7)' or grid cell number
            content_category: Category ('I', 'II', 'III', 'IV')
            content_no: Content number (integer)

        Returns:
            List of Q-values from neighbor nodes, or empty list if none found
        """

        # Convert coord to string for consistent key creation
        content_coord_str = str(content_coord)

        # Create content key
        content_key = f"{content_type}_{content_coord_str}_{content_category}_{content_no}"

        # Collect Q-values from other nodes
        neighbor_q_values = []

        if content_key in self.global_q_values:
            for other_node_id, q_value in self.global_q_values[content_key].items():
                # Don't include the requesting node's own value
                if other_node_id != node_id:
                    neighbor_q_values.append(q_value)

        return neighbor_q_values

    def receive_update(self, node_id, updates):
        """
        Receive Q-value updates from a node

        Args:
            node_id: ID of the sending node
            updates: Dictionary of content_key -> metrics
                     content_key format: "{type}_{coord}_{category}_{no}"
        """

        if node_id not in self.node_updates:
            self.node_updates[node_id] = {}

        # Update global Q-values
        for content_key, metrics in updates.items():
            if content_key not in self.global_q_values:
                self.global_q_values[content_key] = {}

            # Store Q-value from this node
            self.global_q_values[content_key][node_id] = metrics.get('q_value', 0.5)

            # Store full update for this node
            self.node_updates[node_id][content_key] = metrics

        self.update_count += 1

        # Debug output (optional)
        # print(f"📥 Aggregator: Received {len(updates)} updates from {node_id}")

    def aggregate_updates(self):
        """
        Aggregate Q-values across all nodes (called periodically every 10 slots)
        """

        if not self.global_q_values:
            return {}

        # Calculate aggregated Q-values
        aggregated_values = {}

        for content_key, node_values in self.global_q_values.items():
            if node_values:
                # Simple average aggregation
                avg_q_value = sum(node_values.values()) / len(node_values)
                aggregated_values[content_key] = avg_q_value

                # Update all nodes with aggregated value (weighted combination)
                for node_id in node_values:
                    # Combine global average with local value: 30% global, 70% local
                    node_values[node_id] = avg_q_value * 0.3 + node_values[node_id] * 0.7

        # print(f"🔄 Aggregator: Aggregated {len(aggregated_values)} Q-values across {len(self.node_updates)} nodes")

        return aggregated_values

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
