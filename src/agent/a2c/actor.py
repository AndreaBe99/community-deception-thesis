"""Module for the Actor Network"""
from torch import nn
import torch
import os

import sys
sys.path.append('../../')

from src.utils.utils import FilePaths
from src.embedding.graph_encoder import GraphEncoder


class ActorNetwork(nn.Module):
    """Actor Network"""
    
    def __init__(
            self,
            g_in_size,
            g_hidden_size,
            g_embedding_size,
            hidden_size,
            nb_actions,
            chkpt_dir=FilePaths.CHKPT_DIR.value):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_rl')
        self.graph_encoder = GraphEncoder(
            g_in_size, g_hidden_size, g_embedding_size)
        self.linear1 = nn.Linear(g_embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, nb_actions)
        self.nb_actions = nb_actions
        self.tanh = nn.Tanh()
        #TODO try with a Softmax
        self.relu = nn.ReLU()

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: torch.Tensor):
        g = self.graph_encoder(state)
        actions = self.relu(self.linear1(g))
        actions = self.tanh(self.linear2(actions))
        return actions

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
