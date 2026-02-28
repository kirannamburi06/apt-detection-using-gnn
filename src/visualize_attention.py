import torch
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from model import TemporalGAT
from graph_builder import build_temporal_graphs
from data_converter import convert_temporal_graphs


def visualize_attack_detection(model, sequence_data, sequence_id):
    """
    Visualize why a sequence was classified as attack/benign
    """
    model.eval()

    graphs = sequence_data['graphs']
    true_label = sequence_data['label']

    # Get prediction with attention
    with torch.no_grad():
        logits, attention_list = model(graphs, return_attention=True)

        # logits is [1, 2], get the class prediction
        pred = torch.argmax(logits, dim=1).item()
        probs = torch.softmax(logits, dim=1)
        confidence = probs[0, pred].item()

    # Print header
    print("\n" + "=" * 70)
    print(f"DETECTION ANALYSIS: {sequence_id}")
    print("=" * 70)

    true_label_str = "🚨 ATTACK" if true_label == 1 else "✅ BENIGN"
    pred_label_str = "🚨 ATTACK" if pred == 1 else "✅ BENIGN"
    correct = "✅ CORRECT" if pred == true_label else "❌ INCORRECT"

    print(f"\nTrue Label:      {true_label_str}")
    print(f"Predicted:       {pred_label_str}")
    print(f"Confidence:      {confidence * 100:.1f}%")
    print(f"Result:          {correct}")

    # Analyze each time window
    print(f"\n{'=' * 70}")
    print("TEMPORAL ANALYSIS (5 Time Windows)")
    print("=" * 70)

    for t, (graph, attn_data) in enumerate(zip(graphs, attention_list), 1):
        # Unpack attention data
        edge_index, attention = attn_data

        print(f"\n⏰ Time Window {t}:")
        print(f"   Nodes: {graph.num_nodes}, Edges: {edge_index.shape[1]}")

        if attention.shape[0] == 0 or edge_index.shape[1] == 0:
            print(f"   No activity in this window")
            continue

        # Get top suspicious edges
        top_k = min(3, attention.shape[0])
        top_values, top_indices = torch.topk(attention, top_k)

        print(f"   Top {top_k} Suspicious Activities:")

        for i in range(top_k):
            idx = top_indices[i].item()
            weight = top_values[i].item()

            # Get source and destination nodes
            src = edge_index[0, idx].item()
            dst = edge_index[1, idx].item()

            # Try to get node names if available
            if hasattr(graph, 'node_names'):
                src_name = graph.node_names[src]
                dst_name = graph.node_names[dst]
            else:
                src_name = f"Node_{src}"
                dst_name = f"Node_{dst}"

            # Color coding based on attention weight
            level = "🔴" if weight > 0.5 else "🟡" if weight > 0.3 else "🟢"

            print(f"      {i + 1}. {src_name} → {dst_name}")
            print(f"         Attention: {weight:.3f} {level}")

    # Create visualization
    create_attention_heatmap(attention_list, sequence_id)

    print(f"\n{'=' * 70}")
    print(f"✅ Visualization saved to: results/attention_{sequence_id}.png")
    print("=" * 70)


def create_attention_heatmap(attention_list, sequence_id):
    """Create heatmap of attention weights across time"""

    # Extract attention values from tuples
    attention_values = []
    for edge_index, attention in attention_list:
        if attention.shape[0] > 0:
            attention_values.append(attention.cpu().numpy())
        else:
            attention_values.append([])

    # Find max edges
    max_edges = max(len(attn) for attn in attention_values if len(attn) > 0)

    if max_edges == 0:
        print("Warning: No attention data to visualize")
        return

    # Pad attentions to same size
    attention_matrix = []
    for attn in attention_values:
        if len(attn) == 0:
            padded = [0] * max_edges
        else:
            padded = list(attn)
            # Pad if needed
            padded.extend([0] * (max_edges - len(padded)))
        attention_matrix.append(padded[:max_edges])

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(attention_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xlabel('Edge Index', fontsize=12)
    ax.set_ylabel('Time Window', fontsize=12)
    ax.set_title(f'Attention Weights Over Time - {sequence_id}',
                 fontsize=14, fontweight='bold')

    # Set ticks
    ax.set_yticks(range(len(attention_values)))
    ax.set_yticklabels([f'Window {i + 1}' for i in range(len(attention_values))])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(f'../results/attention_{sequence_id}.png', dpi=150, bbox_inches='tight')
    plt.close()


def analyze_all_predictions(model, test_data):
    """Analyze all test predictions"""
    print("\n" + "=" * 70)
    print("ANALYZING ALL TEST PREDICTIONS")
    print("=" * 70)

    for i, sample in enumerate(test_data):
        seq_id = f"test_seq_{i + 1}"
        visualize_attack_detection(model, sample, seq_id)


if __name__ == "__main__":
    print("=" * 70)
    print("ATTENTION VISUALIZATION FOR APT DETECTION")
    print("=" * 70)

    # Load model
    print("\n[1/3] Loading trained model...")
    model = TemporalGAT(node_features=10, gat_hidden=64, lstm_hidden=128, num_classes=2)

    try:
        model.load_state_dict(torch.load('../models/temporal_gat_model.pt'))
        print("✅ Model loaded")
    except FileNotFoundError:
        print("❌ Model not found! Please train the model first:")
        print("   python3 src/train.py")
        exit(1)

    model.eval()

    # Load test data
    print("\n[2/3] Loading test data...")
    from train import load_sequences, prepare_data

    sequences = load_sequences()
    data = prepare_data(sequences)

    # Use last 4 as test (same split as training)
    split_idx = int(0.8 * len(data))
    test_data = data[split_idx:]

    print(f"✅ Loaded {len(test_data)} test sequences")

    # Analyze
    print("\n[3/3] Analyzing predictions with attention...")
    analyze_all_predictions(model, test_data)

    print("\n" + "=" * 70)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nCheck ../results/ folder for attention visualizations:")
    for i in range(len(test_data)):
        print(f"  - attention_test_seq_{i + 1}.png")