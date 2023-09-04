from src.utils.utils import FilePaths
from src.agent.a2c.graph_encoder import GraphEncoder
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import functional as F
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
        self.graph_encoder = GraphEncoder(g_in_size)
        # self.linear1 = nn.Linear(g_embedding_size, hidden_size_1)
        # self.linear2 = nn.Linear(hidden_size_1, 1)
        # self.tanh = nn.Tanh()
        
        # TEST
        self.gcnconv = GCNConv(g_in_size, g_in_size)
        self.linear1 = nn.Linear(g_in_size, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 1)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: Data):
        # embedding = F.relu(self.gcnconv(state.x, state.edge_index))
        # embedding = embedding + state.x
        embedding = self.graph_encoder(state.x, state.edge_index)
        embedding = torch.sum(embedding, dim=0)
        value = F.relu(self.linear1(embedding))
        value = F.relu(self.linear2(value))
        value = self.linear3(value)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)