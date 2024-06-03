import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.nn import GCNConv

class GCNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_features, out_features)

    def forward(self, x, edge_index, active=True):
        x = self.conv(x, edge_index)

        if active:
            x = F.leaky_relu(x, negative_slope=0.2)
        return x
