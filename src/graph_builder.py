import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def build_simple_graph(logs_df):
    """
    Build a provenance graph from audit logs

    Args:
        logs_df: DataFrame with columns [timestamp, process, action, object, pid]

    Returns:
        NetworkX DiGraph
    """
    G = nx.DiGraph()

    for idx, row in logs_df.iterrows():
        process = row['process']
        obj = row['object']
        action = row['action']

        # Add nodes
        G.add_node(process, node_type='process', pid=row['pid'])
        G.add_node(obj, node_type='object')

        # Add edge
        G.add_edge(process, obj, action=action, timestamp=row['timestamp'])

    return G


def print_graph_stats(G):
    """Print graph statistics"""
    print(f"\n{'=' * 50}")
    print(f"GRAPH STATISTICS")
    print(f"{'=' * 50}")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Graph density: {nx.density(G):.4f}")

    print(f"\nTop 5 nodes by degree (most connections):")
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    for node, degree in sorted_nodes:
        print(f"  {node}: {degree} connections")


def visualize_graph(G, output_file='../results/graph_viz.png'):
    """Visualize the graph"""
    plt.figure(figsize=(14, 10))

    # Get node types
    process_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'process']
    object_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'object']

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=process_nodes,
                           node_color='lightblue', node_size=700,
                           label='Processes', alpha=0.9)
    nx.draw_networkx_nodes(G, pos, nodelist=object_nodes,
                           node_color='lightgreen', node_size=700,
                           label='Files/Network', alpha=0.9)

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True,
                           arrowsize=15, edge_color='gray')

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

    plt.legend(loc='upper left', fontsize=12)
    plt.title("Provenance Graph: System Activities", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Graph visualization saved to: {output_file}")
    plt.close()


def build_temporal_graphs(logs_df, num_windows=5):
    """
    Build multiple graphs, one per time window

    Args:
        logs_df: DataFrame with timestamp column
        num_windows: Number of time windows (default 5)

    Returns:
        List of NetworkX graphs
    """
    # Convert to datetime
    logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])

    # Sort by time
    logs_df = logs_df.sort_values('timestamp')

    # Get time range
    start_time = logs_df['timestamp'].min()
    end_time = logs_df['timestamp'].max()
    total_duration = (end_time - start_time).total_seconds()

    # Window size
    window_size = total_duration / num_windows

    print(f"\n{'=' * 50}")
    print(f"TEMPORAL GRAPH CREATION")
    print(f"{'=' * 50}")
    print(f"Total time span: {total_duration / 60:.1f} minutes")
    print(f"Number of windows: {num_windows}")
    print(f"Window size: {window_size / 60:.1f} minutes per window")

    # Build graphs for each window
    graphs = []
    current_time = start_time

    for i in range(num_windows):
        window_end = current_time + pd.Timedelta(seconds=window_size)

        # Filter logs in this window
        window_logs = logs_df[
            (logs_df['timestamp'] >= current_time) &
            (logs_df['timestamp'] < window_end)
            ]

        print(f"\nWindow {i + 1}: {current_time.strftime('%H:%M')} - {window_end.strftime('%H:%M')}")
        print(f"  Logs in window: {len(window_logs)}")

        # Build graph for this window
        if len(window_logs) > 0:
            G = build_simple_graph(window_logs)
            print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        else:
            G = nx.DiGraph()  # Empty graph if no logs
            print(f"  Empty window (no logs)")

        graphs.append(G)
        current_time = window_end

    return graphs


def visualize_temporal_graphs(graphs, output_prefix='../results/temporal_graph'):
    """Visualize all temporal graphs"""
    print(f"\n{'=' * 50}")
    print(f"VISUALIZING TEMPORAL GRAPHS")
    print(f"{'=' * 50}")

    for i, G in enumerate(graphs, 1):
        if G.number_of_nodes() == 0:
            print(f"Window {i}: Skipping (empty graph)")
            continue

        plt.figure(figsize=(10, 8))

        # Get node types
        process_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'process']
        object_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'object']

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw
        nx.draw_networkx_nodes(G, pos, nodelist=process_nodes,
                               node_color='lightblue', node_size=600, alpha=0.9)
        nx.draw_networkx_nodes(G, pos, nodelist=object_nodes,
                               node_color='lightgreen', node_size=600, alpha=0.9)
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=15)
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title(f"Time Window {i} - {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        output_file = f"{output_prefix}_{i}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved: {output_file}")
        plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("PROVENANCE GRAPH BUILDER")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading audit logs...")
    logs = pd.read_csv('../data/toy_audit_logs.csv')
    print(f"✅ Loaded {len(logs)} log entries")

    # Build single graph
    print("\n[2/4] Building single provenance graph...")
    G = build_simple_graph(logs)
    print_graph_stats(G)
    visualize_graph(G)

    # Build temporal graphs
    print("\n[3/4] Building temporal graphs...")
    temporal_graphs = build_temporal_graphs(logs, num_windows=5)
    print(f"\n✅ Created {len(temporal_graphs)} temporal graphs")

    # Visualize temporal graphs
    print("\n[4/4] Visualizing temporal graphs...")
    visualize_temporal_graphs(temporal_graphs)

    print("\n" + "=" * 60)
    print("🎉 ALL DONE!")
    print("=" * 60)
    print("\nCheck these files:")
    print("  - ../results/graph_viz.png (full graph)")
    print("  - ../results/temporal_graph_1.png to temporal_graph_5.png")
    print("\nNext: Build GAT model to process these graphs!")