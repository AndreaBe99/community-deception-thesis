from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

import torch
import torch.nn as nn
import networkx as nx
import scipy.sparse as sp

class GraphEncoder(nn.Module):
    def __init__(self,in_feature, hidden_feature, out_feature):
        super(GraphEncoder, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.out_feature = out_feature
        
        #Torch Geometric GCNConv
        self.pyg_conv = GCNConv(in_feature, hidden_feature)
        self.linear1 = nn.Linear(hidden_feature,out_feature)
        self.tanh = nn.Tanh()
        self.relu = torch.relu
    
    #NOTE Torch Geometric MessagePassing, it takes as input the edge list 
    def forward(self, graph: Data)-> torch.Tensor:
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        
        x = self.pyg_conv(x, edge_index)
        x = self.relu(x)
        embedding = global_mean_pool(x, batch)
        embedding = self.linear1(embedding)
        embedding = self.tanh(embedding)
        return embedding
