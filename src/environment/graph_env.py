"""Module for the GraphEnviroment class"""
from src.community_algs.nmi import NormalizedMutualInformation
from src.community_algs.deception_score import DeceptionScore
from src.community_algs.detection_algs import DetectionAlgorithm
from src.utils.utils import DetectionAlgorithms
from src.utils.utils import HyperParams
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from typing import List, Tuple
# from torch_geometric.data import DataLoader

import math
import numpy as np
import networkx as nx
import torch


class GraphEnvironment(object):
    """Enviroment where the agent will act, it will be a graph with a community"""

    def __init__(self, beta: float, debug: float=None) -> None:
        """Constructor for Graph Environment

        Parameters
        ----------
        beta : float
            Percentage of edges to rewire/update, real number between 1 and 100
        debug : float, optional
            Whether to print debug information, by default None
        """
        assert beta >= 0 and beta <= 100, "Beta must be between 0 and 100"
        self.beta = beta
        self.debug = debug
        self.training = None
        self.eps = 1e-8
        
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # List of possible actions, N+M, where N is ADD actions and M is REMOVE actions
        self.possible_actions = None
        # Length of the list of possible actions to add, used to distinguish
        # between ADD and REMOVE actions in the list of possible actions
        # when applying the action
        self.len_add_actions = 0
        
        # Graph State
        self.graph = None
        self.graph_copy = None
        self.data_pyg = None
        self.n_connected_components = None
        
        # Community Algorithms
        self.deception = None
        self.detection = None
        self.nmi = NormalizedMutualInformation()
        # Community to hide
        self.community_target = None
        # Community Structure before the action
        self.community_structure_old = None
        # Community Structure after the action
        self.community_structure_new = None
        
        # Reward
        self.rewards = None
        self.old_rewards = 0

        # Edge budget, i.e. the number of edges to rewire/update
        self.edge_budget = None
        self.used_edge_budget = None
        self.exhausted_budget = None
    
    @staticmethod
    def get_possible_actions(
        graph: nx.Graph, 
        community: List[int])->List[Tuple[int, int]]:
        """Returns the possible actions that can be applied to the graph.
        An action is a tuple of two nodes, where the first node is the source
        node and the second node is the destination node. 
        The action can be:
            - add an edge between the two nodes, iff one belongs to the 
                community and the other does not.
            - remove an edge between the two nodes, iff both belong to the
                community.

        Parameters
        ----------
        graph : nx.Graph
            Graph where the actions will be applied
        community : List[int]
            Community to hide
        
        Returns
        -------
        List[Tuple[int, int]]
            List of possible actions
        """
        possible_actions = {"ADD": [], "REMOVE": []}
        # Helper functions to check if a node is in/out-side the community
        def in_community(node):
            return node in community
        def out_community(node):
            return node not in community

        for u in graph.nodes():
            for v in graph.nodes():
                if u == v:
                    continue
                # We can remove an edge iff both nodes are in the community
                if in_community(u) and in_community(v):
                    if graph.has_edge(u,v):
                        if (v, u) not in possible_actions["REMOVE"]:
                            possible_actions["REMOVE"].append((u,v))
                            
                # We can add an edge iff one node is in the community and the other is not
                elif (in_community(u) and out_community(v)) \
                    or (out_community(u) and in_community(v)):
                    # Check if there is already an edge between the two nodes
                    if not graph.has_edge(u,v):
                        if (v, u) not in possible_actions["ADD"]:
                            possible_actions["ADD"].append((u,v))
        
        return possible_actions
    
    @staticmethod
    def get_edge_budget(graph: nx.Graph, beta:float) -> int:
        """Computes the edge budget for each graph

        Parameters
        ----------
        graph : nx.Graph
            NetworkX Graph objects, i.e. graph to compute the edge 
            budget for 
        beta : float
            Percentage of edges to rewire/update

        Returns
        -------
        int
            Edge budgets of the graph
        """
        return int(math.ceil((graph.number_of_edges() * beta / 100)))

    @staticmethod
    def get_reward( 
        deception_score: float,
        nmi_score: float,
        weight: float=HyperParams.WEIGHT.value) -> float:
        """
        Computes the reward for the agent
        
        Parameters
        ----------
        deception_score : float
            Deception score
        nmi_score : float
            Normalized Mutual Information score

        Returns
        -------
        reward : float
            Reward
        """
        reward = weight * deception_score + (1 - weight) * nmi_score
        return reward
    
    
    def setup(
        self,
        graph: nx.Graph,
        community: List[int],
        community_detection_algorithm: str = DetectionAlgorithms.LOUV.value,
        training: bool = False) -> None:
        """Setup function for the environment

        Parameters
        ----------
        graph : nx.Graph
            NetworkX Graph object
        community : List[int]
            Community to hide
        community_detection_algorithm : str, optional
            Name of the community detection algorithm to use, by default `louv`
        training : bool, optional
            Whether the environment is used for training, by default False
        """
        self.graph = graph
        self.graph_copy = graph.copy()
        self.community_target = community
        self.training = training
        self.rewards = 0.0

        # Get the Number of connected components
        self.n_connected_components = nx.number_connected_components(
            self.graph)

        self.detection = DetectionAlgorithm(community_detection_algorithm)
        self.deception = DeceptionScore(self.community_target)

        # Compute the community structure of the graph, before the action,
        # i.e. before the deception
        self.community_structure_old = self.detection.compute_community(
            self.graph)

        # Compute the edge budget for the graph
        self.edge_budget = self.get_edge_budget(self.graph, self.beta)
        self.used_edge_budget = 0
        self.exhausted_budget = False

        # Compute the set of possible actions
        self.possible_actions = self.get_possible_actions(
            self.graph, self.community_target)
        # Length of the list of possible actions to add
        self.len_add_actions = len(self.possible_actions["ADD"])

    
    def reset(self) -> Data:
        """Reset the environment

        Returns
        -------
        adj_matrix : torch.Tensor
            Adjacency matrix of the graph
        """
        self.used_edge_budget = 0
        self.exhausted_budget = False
        self.graph = self.graph_copy.copy()
        self.possible_actions = self.get_possible_actions(self.graph, self.community_target)
        
        # Return a PyG Data object
        self.data_pyg = from_networkx(self.graph)
        # self.data_pyg = self.remove_duplicated_edges(self.data_pyg)
        # print("Edges:", self.data_pyg.edge_index.shape)
        # self.data_pyg = self.delete_repeat_edges(from_networkx(self.graph))
        
        # Initialize the node features
        self.data_pyg.x = torch.randn([self.data_pyg.num_nodes, HyperParams.G_IN_SIZE.value])
        # Initialize the batch
        self.data_pyg.batch = torch.zeros(self.data_pyg.num_nodes).long()
        return self.data_pyg.to(self.device)
    
    def apply_action(self, actions: np.array)->int:
        """Applies the action to the graph, if there is an edge between the two 
        nodes, it removes it, otherwise it adds it

        Parameters
        ----------
        actions : np.array
            List of possible actions, where each element is a real number
            between 0 and 1
        
        Returns
        -------
        budget_consumed : int
            Amount of budget consumed
        """
        # Get the index of the action to apply
        index = np.argmax(actions)
        #° The number of possible actions is: 
        #°      len(self.possible_actions["ADD"]) + len(self.possible_actions["REMOVE"])
        #° So, if the index is less than the number of possible actions to add,
        #° the action to apply is an action to add, otherwise it is an action to remove
        if index < self.len_add_actions:
            action = self.possible_actions["ADD"][index]
            # If the action is (-1,-1) it means that the action has already been
            # applied, so we do not need to apply it again
            if action == (-1,-1): return 0
            # Apply the action
            self.graph.add_edge(*action, weight=1)
            # Replace the added edge with (-1,-1) in the possible actions, in this way
            # we can keep track of the used actions, and we can avoid to add the same
            # edge multiple times
            self.possible_actions["ADD"][index] = (-1, -1)
            return 1
        else:
            action = self.possible_actions["REMOVE"][index - self.len_add_actions]
            # If the action is (-1,-1) it means that the action has already been
            # applied, so we do not need to apply it again
            if action == (-1, -1): return 0
            # Apply the action
            self.graph.remove_edge(*action)
            # Replace the removed edge with (-1,-1) in the possible actions,
            # in order to keep the same length, and to avoid to remove the same
            # edge multiple times
            self.possible_actions["REMOVE"][index - self.len_add_actions] = (-1, -1)
            return 1
    
    def step(self, actions: np.array) -> Tuple[Data, float]:
        """Step function for the environment
        
        Parameters
        ----------
        actions : np.array
            Actions to take on the graph, which is a list longer as the number
            of possible actions, where each element is a real number between
            0 and 1.
        Returns
        -------
        self.graph, self.rewards: Tuple[torch.Tensor, float]
            Tuple containing the new graph and the reward 
        """
        # Compute the remaining budget
        remaining_budget = self.edge_budget - self.used_edge_budget
        
        if remaining_budget <= 0:
            self.exhausted_budget = True
            return self.graph, self.rewards
        
        # Take action
        budget_consumed = self.apply_action(actions)
        # Decrease the remaining budget
        updated_budget = remaining_budget - budget_consumed
        
        # Compute the new Community Structure
        self.community_structure_new = self.detection.compute_community(
            self.graph)
        
        # Now we have the old and the new community structure, we can compute
        # the NMI score
        nmi = self.nmi.compute_nmi(
            self.community_structure_old, self.community_structure_new)
        # Compute new deception score
        deception_score = self.deception.compute_deception_score(
            self.community_structure_new, self.n_connected_components)
        
        if self.debug:
            print("Community Structure Old:", self.community_structure_new)
            print("Deception Score:", deception_score)
            print("NMI Score:", nmi)

        
        # Compute the reward, using the deception score and the NMI score
        reward = self.get_reward(deception_score, nmi)
        #TEST Subtract the old reward from the new reward 
        # reward -= self.old_rewards
        if abs(reward) < self.eps:
            reward = 0
        self.rewards = reward
        # self.old_rewards = reward
        
        # Update the used edge budget
        self.used_edge_budget += (remaining_budget - updated_budget)

        # Return a PyG Data object
        data = from_networkx(self.graph)
        # Assign the node features and the batch of the old graph to the new graph
        data.x = self.data_pyg.x
        data.batch = self.data_pyg.batch
        self.data_pyg = data
        return self.data_pyg.to(self.device), self.rewards
    
    def plot_graph(self) -> None:
        """Plot the graph using matplotlib"""
        import matplotlib.pyplot as plt
        nx.draw(self.graph, with_labels=True)
        plt.show()
