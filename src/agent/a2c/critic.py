from src.utils.utils import FilePaths
from src.agent.a2c.graph_encoder import GraphEncoder
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch import nn

import torch
import os

class CriticNetwork(nn.Module):
    def __init__(
        self,
        g_in_size,
        g_embedding_size,
        hidden_size_1,
        chkpt_dir=FilePaths.LOG_DIR.value):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_rl')
        self.graph_encoder_critic = GraphEncoder(
            g_in_size, 
            g_embedding_size,
            )
        
        self.linear1 = nn.Linear(g_embedding_size, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, 1)
        self.tanh = nn.Tanh()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: Data):
        embedding = self.graph_encoder_critic(state)
        value = self.tanh(self.linear1(embedding))
        value = self.tanh(self.linear2(value))
        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)