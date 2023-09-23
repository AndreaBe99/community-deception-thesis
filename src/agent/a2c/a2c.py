"""Module for the ActorCritic class"""
from src.agent.a2c.actor import ActorNetwork
from src.agent.a2c.critic import CriticNetwork
from src.utils.utils import HyperParams, FilePaths, Utils

from torch_geometric.utils.convert import from_networkx
from torch.distributions import MultivariateNormal
from torch_geometric.data import Data
from torch.nn import functional as F
from torch import nn

from typing import Tuple, List
from collections import namedtuple

import networkx as nx

import torch


class ActorCritic(nn.Module):
    """ActorCritic Network"""

    def __init__(
        self, 
        state_dim: int, 
        hidden_size_1: int, 
        hidden_size_2: int, 
        action_dim: int,
        dropout: float):
        super(ActorCritic, self).__init__()
        self.actor = ActorNetwork(
            state_dim=state_dim,
            hidden_size_1=hidden_size_1,
            hidden_size_2=hidden_size_2,
            action_dim=action_dim,
            dropout=dropout,
        )
        self.critic = CriticNetwork(
            state_dim=state_dim,
            hidden_size_1=hidden_size_1,
            hidden_size_2=hidden_size_2,
            dropout=dropout,
        )
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        

    def forward(self, graph: nx.Graph, jitter=1e-20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass, computes action and value

        Parameters
        ----------
        graph : nx.Graph
            Graph state
        jitter : float, optional
            Jitter value, by default 1e-20

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of concentration and value
        """
        # Convert graph to torch_geometric.data.Data
        state = from_networkx(graph).to(self.device)

        # Actor
        probs = self.actor(state)        
        # Use softplus to ensure concentration is positive, then add jitter to 
        # ensure numerical stability
        concentration = F.softplus(probs).reshape(-1) + jitter

        # Critic
        value = self.critic(state)
        return concentration, value
