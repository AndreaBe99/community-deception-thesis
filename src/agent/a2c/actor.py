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
        
        self.graph_encoder = GraphEncoder(state_dim)
        self.linear1 = nn.Linear(state_dim, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, action_dim)
        
        self.relu = nn.LeakyReLU()
        # self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()

    def forward(self, state: Data):
        embedding = self.graph_encoder(state)
        actions = self.relu(self.linear1(embedding))
        actions = self.relu(self.linear2(actions))
        actions = self.linear3(actions)
        return actions
    
    def is_nan(self, x, label):
        """Debugging function to check if there are NaN values in the tensor"""
        if torch.isnan(x).any():
            print(label, ":", x)
            raise ValueError(label, "is NaN")
