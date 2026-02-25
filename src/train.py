import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
from model import TemporalGAT
from graph_builder import build_temporal_graphs
from data_converter import convert_temporal_graphs


def load_sequences():
    """Load labeled sequences"""
    logs = pd.read_csv('../data/labeled_audit_logs.csv')
    labels_df = pd.read_csv('../data/sequence_labels.csv')

    sequences = []

    for _, row in labels_df.iterrows():
        seq_id = row['sequence_id']
        label = row['label']

        # Get logs for this sequence
        seq_logs = logs[logs['sequence_id'] == seq_id].copy()
        seq_logs = seq_logs.drop('sequence_id', axis=1)

        sequences.append({
            'logs': seq_logs,
            'label': label
        })

    return sequences


def prepare_data(sequences):
    """Convert sequences to graph format"""
    data = []

    print("Preparing data...")
    for i, seq in enumerate(sequences):
        # Build temporal graphs
        nx_graphs = build_temporal_graphs(seq['logs'], num_windows=5)

        # Convert to PyG format
        pyg_graphs = convert_temporal_graphs(nx_graphs, node_feature_dim=10)

        # Store
        data.append({
            'graphs': pyg_graphs,
            'label': seq['label']
        })

        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(sequences)} sequences")

    return data


def train_model(model, train_data, epochs=50, lr=0.001):
    """Train the model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining for {epochs} epochs...")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        random.shuffle(train_data)

        for sample in train_data:
            graphs = sample['graphs']

            # Target must be shape [batch_size]
            label = torch.tensor([sample['label']], dtype=torch.long)

            optimizer.zero_grad()

            # Forward pass
            output = model(graphs)   # shape: [1, 2]

            # Compute loss directly (NO unsqueeze!)
            loss = criterion(output, label)

            # Backprop
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Prediction
            pred = torch.argmax(output, dim=1)  # shape: [1]
            correct += (pred == label).sum().item()
            total += 1

        avg_loss = total_loss / total
        accuracy = 100 * correct / total

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")

    print("=" * 60)
    print("✅ Training complete!")

    return model


def evaluate_model(model, test_data):
    """Evaluate the model"""
    model.eval()
    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for sample in test_data:
            graphs = sample['graphs']
            label = torch.tensor([sample['label']], dtype=torch.long)

            output = model(graphs)  # shape [1, 2]
            pred = torch.argmax(output, dim=1)

            correct += (pred == label).sum().item()
            total += 1

            predictions.append({
                'true_label': label.item(),
                'predicted': pred.item(),
                'correct': pred.item() == label.item()
            })

    accuracy = 100 * correct / total

    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")

    print("\nSample predictions:")
    for i, pred in enumerate(predictions[:5]):
        status = "✅" if pred['correct'] else "❌"
        true_label = "Attack" if pred['true_label'] == 1 else "Benign"
        pred_label = "Attack" if pred['predicted'] == 1 else "Benign"
        print(f"  {status} True: {true_label}, Predicted: {pred_label}")

    return accuracy

if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING TEMPORAL GAT MODEL")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Load sequences
    print("\n[1/5] Loading sequences...")
    sequences = load_sequences()
    print(f"✅ Loaded {len(sequences)} sequences")

    # Prepare data
    print("\n[2/5] Preparing data...")
    data = prepare_data(sequences)

    # Split train/test (80/20)
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    print(f"✅ Train: {len(train_data)}, Test: {len(test_data)}")

    # Create model
    print("\n[3/5] Creating model...")
    model = TemporalGAT(
        node_features=10,
        gat_hidden=64,
        lstm_hidden=128,
        num_classes=2
    )
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train
    print("\n[4/5] Training model...")
    model = train_model(model, train_data, epochs=50, lr=0.001)

    # Evaluate
    print("\n[5/5] Evaluating model...")
    accuracy = evaluate_model(model, test_data)

    # Save model
    torch.save(model.state_dict(), '../models/temporal_gat_model.pt')
    print(f"\n✅ Model saved to: ../models/temporal_gat_model.pt")

    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Final Test Accuracy: {accuracy:.2f}%")