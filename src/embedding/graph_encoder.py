import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GraphEncoder(nn.Module):
    def __init__(self,in_feature, hidden_feature, out_feature):
        super(GraphEncoder, self).__init__()

        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.out_feature = out_feature


        self.conv1 = GCNConv(in_feature, hidden_feature)
        self.linear1 = nn.Linear(hidden_feature,out_feature)
        self.tanh = nn.Tanh()
        self.relu = torch.relu

    def forward(self,  Graph):

        x, edge_index,batch = Graph.x, Graph.edge_index,Graph.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        embedding = global_mean_pool(x,batch)
        embedding = self.linear1(embedding)
        embedding = self.tanh(embedding)

        return embedding
