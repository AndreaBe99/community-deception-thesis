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

    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        actor_cfg = {
            # Encoder
            'g_in_size': state_dim,
            'g_embedding_size': HyperParams.G_EMBEDDING_SIZE.value,

            # Actor
            'hidden_size_1': HyperParams.HIDDEN_SIZE_1.value,
            'hidden_size_2': HyperParams.HIDDEN_SIZE_2.value,
            'nb_actions': action_dim,
        }
        critic_cfg = {
            # Encoder
            'g_in_size': state_dim,
            'g_embedding_size': HyperParams.G_EMBEDDING_SIZE.value,

            # Critic
            'hidden_size_1': HyperParams.HIDDEN_SIZE_1.value,
        }
        self.actor = ActorNetwork(**actor_cfg)
        self.critic = CriticNetwork(**critic_cfg)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, state: Data, jitter=1e-20) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
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
        # Actor
        probs = self.actor(state)
        concentration = F.softplus(probs).reshape(-1) + jitter
        # Critic
        value = self.critic(state)
        return concentration, value
    