from torch.nn import Module
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
import warnings

from opt import args

warnings.filterwarnings("ignore")

class GraphTransformerLayer(Module):
    def __init__(self, in_features, out_features, heads=1):
        super(GraphTransformerLayer, self).__init__()
        self.conv = TransformerConv(in_features, out_features, heads=heads)

    def forward(self, x, edge_index, active=True):
        x = self.conv(x, edge_index)

        if active:
            x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        return x