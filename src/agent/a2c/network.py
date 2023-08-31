"""Module for the ActorCritic class"""
from src.agent.a2c.actor import ActorNetwork
from src.agent.a2c.critic import CriticNetwork
from src.agent.a2c.memory import Memory
from src.utils.utils import HyperParams

from torch.distributions import MultivariateNormal
from torch_geometric.data import Data

from typing import Tuple
from torch import nn

import torch


class ActorCritic(nn.Module):
    """ActorCritic Network"""
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        actor_cfg = {
            'g_in_size': state_dim,
            'g_hidden_size': HyperParams.G_HIDDEN_SIZE.value,
            'g_embedding_size': HyperParams.G_EMBEDDING_SIZE.value,
            'hidden_size': HyperParams.HIDDEN_SIZE.value,
            'nb_actions': action_dim,
        }
        critic_cfg = {
            'g_in_size': state_dim,
            'g_hidden_size': HyperParams.G_HIDDEN_SIZE.value,
            'g_embedding_size': HyperParams.G_EMBEDDING_SIZE.value,
        }
        self.actor = ActorNetwork(**actor_cfg)
        self.critic = CriticNetwork(**critic_cfg)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.action_var = torch.full(
            (action_dim,), action_std*action_std).to(self.device)

    def forward(self):
        """Forward pass"""
        raise NotImplementedError

    def act(
        self, 
        state: Data,
        memory: Memory)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the action to take given the current state

        Parameters
        ----------
        state : torch.Tensor
            Current state
        memory : Memory
            Memory object

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The action to take and the log probability of the action
        """
        action_mean = self.actor(state)
        
        cov_mat = torch.diag(self.action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()
        #return action.detach(), action_logprob.detach()
    
    def evaluate(self, state: torch.Tensor, action)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the current state and action

        Parameters
        ----------
        state : _type_
            _description_
        action : _type_
            _description_

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The log probability of the action, the state value, and the entropy
        """
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, state_value, dist_entropy