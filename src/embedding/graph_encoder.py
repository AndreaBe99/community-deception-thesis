from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # TEST Use GATConv
        self.in_head = self.hidden_feature
        self.conv1 = GATConv(
            self.in_feature, 
            self.hidden_feature, 
            heads=self.in_head, 
            dropout=0.6)
        self.conv2 = GATConv(
            self.hidden_feature * self.in_head, 
            self.out_feature, 
            concat=False,
            heads=self.out_feature,
            dropout=0.6)
    
    #NOTE Torch Geometric MessagePassing, it takes as input the edge list 
    def forward(self, graph: Data)-> torch.Tensor:
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.pyg_conv(x, edge_index)
        x = self.relu(x)
        embedding = global_mean_pool(x, batch)
        embedding = self.linear1(embedding)
        embedding = self.tanh(embedding)
        return embedding
    
    # TEST Use GATConv
    # def forward(self, graph: Data)-> torch.Tensor:
    #     x, edge_index, batch = graph.x, graph.edge_index, graph.batch
    #     # Dropout before the GAT layer is used to avoid overfitting in small datasets like Cora.
    #     # One can skip them if the dataset is sufficiently large.
    #     x = F.dropout(x, p=0.6, training=self.training)
    #     x = self.conv1(x, edge_index)
    #     x = F.elu(x)
    #     x = F.dropout(x, p=0.6, training=self.training)
    #     x = self.conv2(x, edge_index)
    #     x = self.relu(x)
    #     embedding = global_mean_pool(x, batch)
    #     embedding = self.linear1(embedding)
    #     embedding = self.tanh(embedding)
    #     return embedding
    
        #return F.log_softmax(x, dim=1)
        # return torch.flatten(x).T