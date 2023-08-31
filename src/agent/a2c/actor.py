"""Module for the Actor Network"""
from src.utils.utils import FilePaths
from src.embedding.graph_encoder import GraphEncoder
from torch_geometric.data import Data
from torch import nn

import torch
import os

# import sys
# sys.path.append('../../')

class ActorNetwork(nn.Module):
    """Actor Network"""
    
    def __init__(
            self,
            g_in_size,
            g_hidden_size,
            g_embedding_size,
            hidden_size,
            nb_actions,
            chkpt_dir=FilePaths.LOG_DIR.value):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_rl')
        self.graph_encoder = GraphEncoder(
            in_feature=g_in_size, 
            hidden_feature=g_hidden_size, 
            out_feature=g_embedding_size)
        self.linear1 = nn.Linear(g_embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, nb_actions)
        self.nb_actions = nb_actions
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        #TODO try with a Softmax
        self.softmax = nn.Softmax(dim=1)

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: Data):
        g = self.graph_encoder(state)
        actions = self.relu(self.linear1(g))
        # actions = self.tanh(self.linear2(actions))
        actions = self.softmax(self.linear2(actions))
        
        return actions

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
