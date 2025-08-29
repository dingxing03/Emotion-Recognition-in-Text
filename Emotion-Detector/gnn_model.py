import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNRefiner(nn.Module):
    def __init__(self, edge_index, edge_weight=None, hidden=64):
        super().__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.gcn1 = GCNConv(1, hidden)
        self.gcn2 = GCNConv(hidden, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (batch, num_emotions) logits
        outs = []
        for i in range(x.size(0)):
            xi = x[i].unsqueeze(1)  # (num_emotions, 1)
            h = F.relu(self.gcn1(xi, self.edge_index, self.edge_weight))
            h = self.dropout(h)
            delta = self.gcn2(h, self.edge_index, self.edge_weight).squeeze(1)  # (num_emotions,)
            outs.append(x[i] + 0.5 * delta)  # residual with small step
        return torch.stack(outs, dim=0)