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

        # self.graph_encoder = GraphEncoder(state_dim)
        # self.conv1 = GCNConv(state_dim, state_dim)
        
        self.lin1 = nn.Linear(state_dim, hidden_size_1)
        self.lin2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.lin3 = nn.Linear(hidden_size_2, 1)

        # self.relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # self.relu = F.relu
        # self.tanh = nn.Tanh()

    def forward(self, data: torch.Tensor)->torch.Tensor:
        # out = F.relu(self.conv1(data.x, data.edge_index))
        # x = out + data.x
        # x = torch.sum(x, dim=0)
        x = self.relu(self.lin1(data))
        # x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.lin3(x)
        return x

    """
    def forward(self, state: Data):
        embedding = self.graph_encoder(state)
        embedding = torch.sum(embedding, dim=0)
        value = self.relu(self.linear1(embedding))
        value = self.relu(self.linear2(value))
        value = self.linear3(value)
        return value
    """
