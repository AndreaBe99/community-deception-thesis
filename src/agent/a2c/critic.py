from src.utils.utils import FilePaths, HyperParams
from src.agent.a2c.graph_encoder import GraphEncoder
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import functional as F
from torch import nn

import torch
import os

class CriticNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_size_1: int,
        hidden_size_2: int):
        super(CriticNetwork, self).__init__()
        
        self.graph_encoder = GraphEncoder(state_dim)
        self.linear1 = nn.Linear(state_dim, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, 1)
        
        self.relu = nn.LeakyReLU()
        # self.relu = nn.ReLU()
        # self.relu = F.relu
        # self.tanh = nn.Tanh()

    def forward(self, state: Data):
        embedding = self.graph_encoder(state)
        embedding = torch.sum(embedding, dim=0)
        value = self.relu(self.linear1(embedding))
        value = self.relu(self.linear2(value))
        value = self.linear3(value)
        return value
