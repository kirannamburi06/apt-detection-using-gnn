import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class SimpleGAT(nn.Module):
    """
    Simple Graph Attention Network for graph classification
    """

    def __init__(self, node_features=10, hidden_dim=64, num_heads=4, num_classes=2):
        super(SimpleGAT, self).__init__()

        self.gat1 = GATConv(
            in_channels=node_features,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=0.6
        )

        self.gat2 = GATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            dropout=0.6
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch=None, return_attention=False):

        # First GAT layer
        x, _ = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)

        # Second GAT layer
        x, (edge_index_2, attention_2) = self.gat2(
            x, edge_index, return_attention_weights=True
        )
        x = F.elu(x)

        # Pooling
        if batch is None:
            x = torch.mean(x, dim=0, keepdim=True)
        else:
            x = global_mean_pool(x, batch)

        logits = self.classifier(x)

        if return_attention:
            # Average heads if needed
            if attention_2.dim() > 1:
                attn = attention_2.mean(dim=1)
            else:
                attn = attention_2

            return logits, (edge_index_2, attn)

        return logits


class TemporalGAT(nn.Module):
    """
    Temporal Graph Attention Network with LSTM
    """

    def __init__(self, node_features=10, gat_hidden=64, lstm_hidden=128, num_classes=2):
        super(TemporalGAT, self).__init__()

        self.gat = SimpleGAT(
            node_features=node_features,
            hidden_dim=gat_hidden,
            num_heads=4,
            num_classes=gat_hidden
        )

        self.lstm = nn.LSTM(
            input_size=gat_hidden,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )

        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, temporal_graphs, return_attention=False):

        graph_embeddings = []
        attention_outputs = []

        for graph in temporal_graphs:

            if return_attention:
                embedding, (edge_idx, attn) = self.gat(
                    graph.x, graph.edge_index, return_attention=True
                )
                attention_outputs.append((edge_idx, attn))
            else:
                embedding = self.gat(graph.x, graph.edge_index)

            graph_embeddings.append(embedding.squeeze(0))

        # Stack sequence
        sequence = torch.stack(graph_embeddings).unsqueeze(0)

        lstm_out, (h_n, _) = self.lstm(sequence)

        final_hidden = h_n[-1]  # shape [1, hidden]
        logits = self.classifier(final_hidden)  # shape [1, 2]

        if return_attention:
            return logits, attention_outputs

        return logits