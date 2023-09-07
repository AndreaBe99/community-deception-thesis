"""Module for the ActorCritic class"""
from src.agent.a2c.actor import ActorNetwork
from src.agent.a2c.critic import CriticNetwork
from src.agent.a2c.memory import Memory
from src.utils.utils import HyperParams, FilePaths, Utils

from torch.distributions import MultivariateNormal
from torch_geometric.data import Data
from torch.nn import functional as F
from torch import nn
from collections import namedtuple

from typing import Tuple, List

import torch


class ActorCritic(nn.Module):
    """ActorCritic Network"""

    def __init__(self, state_dim, hidden_size_1, hidden_size_2, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = ActorNetwork(
            state_dim=state_dim,
            hidden_size_1=hidden_size_1,
            hidden_size_2=hidden_size_2,
            action_dim=action_dim
        )
        self.critic = CriticNetwork(
            state_dim=state_dim,
            hidden_size_1=hidden_size_1,
            hidden_size_2=hidden_size_2
        )
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, state: Data, jitter=1e-20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass, computes action and value
        
        Parameters
        ----------
        state : Data
            Graph state
        jitter : float, optional
            Jitter value, by default 1e-20
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of concentration and value
        """
        state = state.to(self.device)
        # Actor
        probs = self.actor(state)
        # Adds jitter to ensure numerical stability
        # Use softplus to ensure concentration is positive
        concentration = F.softplus(probs).reshape(-1) + jitter
        # Critic
        value = self.critic(state)
        return concentration, value
    