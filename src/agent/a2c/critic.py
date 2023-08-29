from torch import nn
import torch
import os

import sys
sys.path.append('../../')

from src.utils.utils import FilePaths
from src.embedding.graph_encoder import GraphEncoder

class CriticNetwork(nn.Module):
    def __init__(
        self,
        g_in_size,
        g_hidden_size,
        g_embedding_size,
        chkpt_dir=FilePaths.CHKPT_DIR.value):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_rl')
        self.graph_encoder_critic = GraphEncoder(
            g_in_size, g_hidden_size, g_embedding_size)
        self.linear1 = nn.Linear(g_embedding_size, 1)
        #TODO try with a Softmax
        self.tanh = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()

        # self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        g = self.graph_encoder_critic(state)
        value = self.tanh(self.linear1(g))
        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)