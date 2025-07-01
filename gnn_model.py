import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNBinaryClassifier(torch.nn.Module):
    """
    A simple 2-layer GCN for binary classification.
    """
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GNNBinaryClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # First graph convolution + ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Second graph convolution (outputs raw logits per node)
        x = self.conv2(x, edge_index)
        return x
