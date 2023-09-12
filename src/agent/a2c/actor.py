"""Module for the Actor Network"""
from src.utils.utils import FilePaths, HyperParams
from src.agent.a2c.graph_encoder import GraphEncoder
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch import nn

import torch
import os


class ActorNetwork(nn.Module):
    """Actor Network"""

    def __init__(
            self,
            state_dim: int,
            hidden_size_1: int,
            hidden_size_2: int,
            action_dim: int):
        super(ActorNetwork, self).__init__()

        # self.graph_encoder = GraphEncoder(state_dim)
        # self.conv1 = GCNConv(state_dim, state_dim)
        
        self.lin1 = nn.Linear(state_dim, hidden_size_1)
        self.lin2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.lin3 = nn.Linear(hidden_size_2, action_dim)
        
        # self.relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()

    def forward(self, data: torch.Tensor)->torch.Tensor:
        # out = F.relu(self.conv1(data.x, data.edge_index))
        # x = out + data.x
        x = F.relu(self.lin1(data))
        # x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

    """
    def forward(self, state: Data):
        embedding = self.graph_encoder(state)
        actions = self.relu(self.linear1(embedding))
        actions = self.relu(self.linear2(actions))
        actions = self.linear3(actions)
        return actions
    """
