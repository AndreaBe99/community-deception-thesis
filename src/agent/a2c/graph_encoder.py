from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_feature, 
        # embdedding_size,
        num_layers=2):
        super(GraphEncoder, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(in_feature, in_feature))
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(in_feature, in_feature))
    
    #NOTE Torch Geometric MessagePassing, it takes as input the edge list 
    def forward(self, graph: Data)-> torch.Tensor:
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
        # embedding = global_mean_pool(x, batch)
        embedding = x + graph.x
        return embedding