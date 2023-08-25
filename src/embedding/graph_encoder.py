import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GraphEncoder(nn.Module):
    def __init__(self,in_feature, hidden_feature, out_feature):
        super(GraphEncoder, self).__init__()

        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.out_feature = out_feature


        self.conv1 = GCNConv(in_feature, hidden_feature)
        self.linear1 = nn.Linear(hidden_feature,out_feature)
        self.tanh = nn.Tanh()
        self.relu = torch.relu
    
    def forward(self, edge_index: torch.Tensor)-> torch.Tensor:
        n_nodes = edge_index.max().item() + 1
        x = torch.randn([n_nodes, 50])
        batch = torch.zeros(n_nodes).long()
        
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        embedding = global_mean_pool(x, batch)
        embedding = self.linear1(embedding)
        embedding = self.tanh(embedding)

        return embedding
    
    """
    def forward(self, graph: Data)-> torch.Tensor:
        # Features Matrix
        x = graph.x
        
        # The graph is defined using an adjacency matrix. 
        # The first row of the tensor contains the indices of the source nodes, 
        # and the second row contains the indices of the destination nodes. 
        # Each row of the tensor represents an edge in the graph.
        edge_index = graph.edge_index
        
        # The batch tensor is used to indicate which nodes belong to which
        # graphs in a batch. In this case, there is only one graph, so the
        # batch tensor is a tensor of zeros with shape (num_nodes, 1).
        batch = graph.batch
        
        if x is None:
            x = torch.randn([graph.num_nodes, 50])
        # Check if the batch tensor is None
        if batch is None:
            # batch = torch.zeros([graph.num_nodes, 1]).long()
            batch = torch.zeros(graph.num_nodes).long()
        
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        embedding = global_mean_pool(x, batch)
        embedding = self.linear1(embedding)
        embedding = self.tanh(embedding)

        return embedding
    """
    
    """
    def forward(self, edge_index: torch.Tensor):
        # Features Matrix
        # x = graph.x
        
        # The graph is defined using an adjacency matrix. 
        # The first row of the tensor contains the indices of the source nodes, 
        # and the second row contains the indices of the destination nodes. 
        # Each row of the tensor represents an edge in the graph.
        # edge_index = graph.edge_index
        
        # TODO: Remove duplicate edges, without reshaping the tensor
        # Reshape the edge_index tensor in N*2
        edge_index = edge_index.t().contiguous()
        # Remove the duplicate edges
        sorted_edge_index, _ = torch.sort(edge_index, dim=1)
        edge_index = torch.unique(sorted_edge_index, dim=0)
        edge_index = edge_index.t().contiguous()
        
        # The batch tensor is used to indicate which nodes belong to which 
        # graphs in a batch. In this case, there is only one graph, so the 
        # batch tensor is a tensor of zeros with shape (num_nodes, 1).
        # batch = graph.batch
        
        n_nodes = edge_index.max().item() + 1
        # Check if the features matrix is None
        # if x is None:
        x = torch.randn([n_nodes, 50])
        # Check if the batch tensor is None
        # if batch is None:
        # batch = torch.zeros([graph.num_nodes, 1]).long()
        batch = torch.zeros(n_nodes).long()
        
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        embedding = global_mean_pool(x, batch)
        embedding = self.linear1(embedding)
        embedding = self.tanh(embedding)

        return embedding
    """