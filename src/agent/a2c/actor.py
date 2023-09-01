"""Module for the Actor Network"""
from src.utils.utils import FilePaths
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
            g_in_size,
            g_embedding_size,
            hidden_size_1,
            hidden_size_2,
            nb_actions,
            chkpt_dir=FilePaths.LOG_DIR.value):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_rl')
        
        self.graph_encoder = GraphEncoder(
            in_feature=g_in_size, 
            embdedding_size=g_embedding_size,
            )
        
        self.linear1 = nn.Linear(g_embedding_size, hidden_size_1)
        # self.linear1 = nn.Linear(g_embedding_size, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, nb_actions)
        
        self.nb_actions = nb_actions
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: Data):
        embedding = self.graph_encoder(state)
        # embedding = embedding + state.x
        actions = F.relu(self.linear1(embedding))
        actions = F.relu(self.linear2(actions))
        actions = F.softmax(self.linear3(actions))
        return actions

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
