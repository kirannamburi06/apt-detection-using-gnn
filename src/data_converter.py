import torch
import networkx as nx
from torch_geometric.data import Data


def networkx_to_pyg(G, node_feature_dim=10):
    """
    Convert NetworkX graph to PyTorch Geometric Data object

    Args:
        G: NetworkX DiGraph
        node_feature_dim: Dimension of node feature vectors

    Returns:
        PyTorch Geometric Data object
    """
    if G.number_of_nodes() == 0:
        # Return empty graph
        return Data(x=torch.zeros((1, node_feature_dim)),
                    edge_index=torch.zeros((2, 0), dtype=torch.long))

    # Create node mapping (node name -> index)
    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Create edge index tensor [2, num_edges]
    edge_list = []
    for source, target in G.edges():
        src_idx = node_to_idx[source]
        tgt_idx = node_to_idx[target]
        edge_list.append([src_idx, tgt_idx])

    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Create node features
    node_features = []
    for node in node_list:
        features = create_node_features(G, node, node_feature_dim)
        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)
    data.node_names = node_list  # Store for later interpretation

    return data


def create_node_features(G, node, feature_dim=10):
    """
    Create feature vector for a node

    For now, simple features:
    - Node type (process=1, object=0)
    - Degree (number of connections)
    - In-degree
    - Out-degree
    - Rest: zeros (for now)
    """
    node_data = G.nodes[node]

    # Feature 1: Node type
    is_process = 1.0 if node_data.get('node_type') == 'process' else 0.0

    # Feature 2-4: Degree features (normalized)
    degree = G.degree(node) / 10.0  # Simple normalization
    in_degree = G.in_degree(node) / 10.0
    out_degree = G.out_degree(node) / 10.0

    # Create feature vector
    features = [
        is_process,
        degree,
        in_degree,
        out_degree,
    ]

    # Pad with zeros to reach feature_dim
    features.extend([0.0] * (feature_dim - len(features)))

    return features[:feature_dim]


def convert_temporal_graphs(nx_graphs, node_feature_dim=10):
    """
    Convert list of NetworkX graphs to PyG format

    Args:
        nx_graphs: List of NetworkX graphs
        node_feature_dim: Dimension of node features

    Returns:
        List of PyG Data objects
    """
    pyg_graphs = []

    print(f"\nConverting {len(nx_graphs)} graphs to PyTorch Geometric format...")

    for i, G in enumerate(nx_graphs, 1):
        pyg_graph = networkx_to_pyg(G, node_feature_dim)
        pyg_graphs.append(pyg_graph)

        print(f"  Graph {i}: {pyg_graph.num_nodes} nodes, {pyg_graph.num_edges} edges, "
              f"features: {pyg_graph.x.shape}")

    return pyg_graphs


if __name__ == "__main__":
    import pandas as pd
    from graph_builder import build_temporal_graphs

    print("=" * 60)
    print("GRAPH DATA CONVERTER TEST")
    print("=" * 60)

    # Load data and build graphs
    logs = pd.read_csv('../data/toy_audit_logs.csv')
    nx_graphs = build_temporal_graphs(logs, num_windows=5)

    # Convert to PyG format
    pyg_graphs = convert_temporal_graphs(nx_graphs, node_feature_dim=10)

    print("\n✅ Conversion successful!")
    print(f"\nExample graph (first window):")
    print(f"  Node features shape: {pyg_graphs[0].x.shape}")
    print(f"  Edge index shape: {pyg_graphs[0].edge_index.shape}")
    print(f"  Number of nodes: {pyg_graphs[0].num_nodes}")
    print(f"  Number of edges: {pyg_graphs[0].num_edges}")