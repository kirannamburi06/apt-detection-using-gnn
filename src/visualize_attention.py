import torch
import matplotlib.pyplot as plt
from model import TemporalGAT


def visualize_attack_detection(model, sequence_data, sequence_id):

    model.eval()

    graphs = sequence_data['graphs']
    true_label = sequence_data['label']

    with torch.no_grad():
        logits, attention_list = model(graphs, return_attention=True)

        pred = torch.argmax(logits, dim=1).item()

        probs = torch.softmax(logits, dim=1)
        confidence = probs[0, pred].item()

    print("\n" + "=" * 70)
    print(f"DETECTION ANALYSIS: {sequence_id}")
    print("=" * 70)

    true_label_str = "🚨 ATTACK" if true_label == 1 else "✅ BENIGN"
    pred_label_str = "🚨 ATTACK" if pred == 1 else "✅ BENIGN"
    correct = "✅ CORRECT" if pred == true_label else "❌ INCORRECT"

    print(f"\nTrue Label:  {true_label_str}")
    print(f"Predicted:   {pred_label_str}")
    print(f"Confidence:  {confidence * 100:.2f}%")
    print(f"Result:      {correct}")

    print("\n" + "=" * 70)
    print("TEMPORAL ANALYSIS")
    print("=" * 70)

    for t, (graph, attn_data) in enumerate(zip(graphs, attention_list), 1):

        edge_idx, attention = attn_data

        print(f"\n⏰ Time Window {t}:")
        print(f"   Nodes: {graph.num_nodes}, Edges: {edge_idx.shape[1]}")

        if attention.shape[0] == 0:
            print("   No activity")
            continue

        top_k = min(3, attention.shape[0])
        top_values, top_indices = torch.topk(attention, top_k)

        print(f"   Top {top_k} Suspicious Activities:")

        for i in range(top_k):
            idx = top_indices[i]
            weight = top_values[i]

            src = edge_idx[0, idx].item()
            dst = edge_idx[1, idx].item()

            src_name = f"Node_{src}"
            dst_name = f"Node_{dst}"

            level = "🔴" if weight > 0.5 else "🟡" if weight > 0.3 else "🟢"

            print(f"      {i+1}. {src_name} → {dst_name}")
            print(f"         Attention: {weight.item():.3f} {level}")

    print("\n" + "=" * 70)