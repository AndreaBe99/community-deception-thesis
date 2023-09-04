"""Module for the GraphEnviroment class"""
from src.community_algs.nmi import NormalizedMutualInformation
from src.community_algs.deception_score import DeceptionScore
from src.community_algs.detection_algs import DetectionAlgorithm
from src.community_algs.safeness import Safeness
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
    """Enviroment where the agent will act, it will be a graph with a community
    
    Methods
    -------
    get_edge_budget(graph, beta) -> int
        Computes the edge budget for each graph
    get_reward(deception_score, nmi_score) -> float
        Computes the reward for the agent
    reset() -> Data
        Reset the environment
    step(actions) -> Tuple[Data, float]
        Step function for the environment
    get_possible_actions() -> List[Tuple[int, int]]
        Returns the possible actions that can be applied to the graph.
    apply_action(actions) -> int
        Applies the action to the graph, if there is an edge between the two
        nodes, it removes it, otherwise it adds it
    plot_graph() -> None
        Plot the graph using matplotlib
    """

    def __init__(
        self, 
        graph: nx.Graph, 
        community: List[int], 
        nodes_target: List[int],
        beta: float, 
        weight: float = HyperParams.WEIGHT.value,
        debug: float = None, 
        training: bool = False, 
        env_name: str = 'default', 
        community_detection_algorithm: str = DetectionAlgorithms.LOUV.value) -> None:
        """Constructor for Graph Environment
        Parameters
        ----------
        graph : nx.Graph
            Graph to use for the environment
        community : List[int]
            Community of node we want to remove from it
        nodes_target : List[int]
            Nodes we want to remove from the community
        beta : float
            Percentage of edges to rewire/update, real number between 1 and 100
        weight : float, optional
            Weight to balance the reward, by default HyperParams.WEIGHT.value
        debug : float, optional
            Whether to print debug information, by default None
        training : bool, optional
            Whether to train the agent, by default False
        env_name : str, optional
            Name of the environment, by default 'default'
        community_detection_algorithm : str, optional
            Name of the community detection algorithm, by default DetectionAlgorithms.LOUV.value
        """
        self.graph = graph
        self.graph_copy = graph.copy()
        # Get the Number of connected components
        self.n_connected_components = nx.number_connected_components(graph)
        
        # Community to hide
        self.community_target = community
        
        # Node to remove from the community
        assert set(nodes_target).issubset(set(community)), "Nodes must be a subset of the community"
        self.nodes_target = nodes_target
        
        assert beta >= 0 and beta <= 100, "Beta must be between 0 and 100"
        self.beta = beta
        self.weight = weight
        self.eps = 1e-8
        self.debug = debug
        self.training = training
        self.env_name = env_name
        
        # Community Algorithms objects
        self.detection = DetectionAlgorithm(community_detection_algorithm)
        self.deception = DeceptionScore(self.community_target)
        self.safeness = Safeness(self.graph, self.community_target, self.nodes_target)
        self.nmi = NormalizedMutualInformation()
        # Compute the community structure of the graph, before the action,
        # i.e. before the deception
        self.community_structure_old = self.detection.compute_community(graph)
        
        # Compute the edge budget for the graph
        self.edge_budget = self.get_edge_budget()
        self.used_edge_budget = 0
        self.exhausted_budget = False
        self.rewards = 0
        self.old_rewards = 0
        
        # Compute the set of possible actions
        self.possible_actions = self.get_possible_actions()
        # Length of the list of possible actions to add
        self.len_add_actions = len(self.possible_actions["ADD"])
        
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

    def get_edge_budget(self) -> int:
        """Computes the edge budget for each graph

        Returns
        -------
        int
            Edge budgets of the graph
        """
        return int(math.ceil((self.graph.number_of_edges() * self.beta / 100)))

    def get_reward(self, deception_score: float, nmi_score: float) -> float:
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
        reward = self.weight * deception_score + (1 - self.weight) * nmi_score
        return reward

    
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
        self.possible_actions = self.get_possible_actions()
        
        # Return a PyG Data object
        self.data_pyg = from_networkx(self.graph)
        # Initialize the node features
        self.data_pyg.x = torch.randn([self.data_pyg.num_nodes, HyperParams.G_IN_SIZE.value])
        # Initialize the batch
        self.data_pyg.batch = torch.zeros(self.data_pyg.num_nodes).long()
        return self.data_pyg.to(self.device)
    
    
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
        
        # Take action, budget_consumed can be 0 or 1, i.e. if the action has
        # been applied or not
        budget_consumed = self.apply_action(actions)
        # Decrease the remaining budget
        updated_budget = remaining_budget - budget_consumed
        
        # Compute the new Community Structure
        self.community_structure_new = self.detection.compute_community(self.graph)
        
        # Now we have the old and the new community structure, we can compute
        # the NMI score
        nmi = self.nmi.compute_nmi(self.community_structure_old, self.community_structure_new)
        # Compute new deception score
        # deception_score = self.deception.compute_deception_score(self.community_structure_new, self.n_connected_components)
        # Compute the node safeness
        node_safeness = self.safeness.compute_community_safeness(self.nodes_target)
        if self.debug:
            print("Community Structure Old:", self.community_structure_new)
            # print("Deception Score:", deception_score)
            print("NMI Score:", nmi)
        
        # Compute the reward, using the deception score and the NMI score
        # reward = self.get_reward(deception_score, nmi)
        reward = self.get_reward(node_safeness, nmi)
        # TEST Subtract the old reward from the new reward 
        if budget_consumed == 0:
            reward = -1
        reward *= 2*budget_consumed
        reward -= self.old_rewards
        

        if abs(reward) < self.eps:
            reward = 0
        self.rewards = reward
        self.old_rewards = reward
        
        # Update the used edge budget
        self.used_edge_budget += (remaining_budget - updated_budget)
        # If the budget for the graph rewiring is exhausted, stop the episode
        if remaining_budget < 1:
            # print("*", "-" * 19, "Budget exhausted", "-" * 19)
            self.exhausted_budget = True

        # Return a PyG Data object
        data = from_networkx(self.graph)
        # Assign the node features and the batch of the old graph to the new graph
        data.x = self.data_pyg.x
        data.batch = self.data_pyg.batch
        self.data_pyg = data
        return self.data_pyg.to(self.device), self.rewards, self.exhausted_budget
    
    # TEST Goal: remove a node from a community
    def get_possible_actions(self) -> dict:
        """Returns the possible actions that can be applied to the graph.
        An action is a tuple of two nodes, where the first node is the source
        node and the second node is the destination node. 
        The action can be:
            - add an edge between the two nodes, iff one belongs to the 
                community and the other does not.
            - remove an edge between the two nodes, iff both belong to the
                community.
        
        Returns
        -------
        self.possible_actions : dict
            Dictionary containing the possible actions that can be applied to
            the graph. The dictionary has two keys: "ADD" and "REMOVE", each
            key has a list of tuples as value, where each tuple is an action.
        """
        possible_actions = {"ADD": set(), "REMOVE": set()}
        # Helper functions to check if a node is in/out-side the community
        def in_community(node):
            return node in self.community_target

        def out_community(node):
            return node not in self.community_target
        
        def in_community_and_not_v(u, v):
            if u == v:
                return # continue
            # We can remove an edge iff both nodes are in the community
            if in_community(u) and in_community(v):
                if self.graph.has_edge(u, v):
                    if (v, u) not in possible_actions["REMOVE"]:
                        possible_actions["REMOVE"].add((u, v))
            # We can add an edge iff one node is in the community and the other is not
            elif (in_community(u) and out_community(v)) \
                    or (out_community(u) and in_community(v)):
                # Check if there is already an edge between the two nodes
                if not self.graph.has_edge(u, v):
                    if (v, u) not in possible_actions["ADD"]:
                        possible_actions["ADD"].add((u, v))

        # TEST For now consider both goal, i.e. hide an entire community and 
        # TEST remove a node from a community
        for u in self.graph.nodes():
            if self.nodes_target is None:
                for v in self.graph.nodes():
                    in_community_and_not_v(u, v)
            else:
                for node in self.nodes_target:
                    in_community_and_not_v(u, node)
        return possible_actions
    
    def apply_action(self, actions: np.array) -> int:
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
        # TEST: actions is a list longer as the number of nodes in the graph
        # TEST  choose the two nodes with the highest value in the list
        action = np.argsort(actions)[-2:]
        # We need to take into account both the actions (u,v) and (v,u)
        action = (action[0], action[1])
        action_reversed = (action[1], action[0])
        if action in self.possible_actions["ADD"]:
            self.graph.add_edge(*action, weight=1)
            self.possible_actions["ADD"].remove(action)
            return 1
        elif action_reversed in self.possible_actions["ADD"]:
            self.graph.add_edge(*action_reversed, weight=1)
            self.possible_actions["ADD"].remove(action_reversed)
            return 1
        elif action in self.possible_actions["REMOVE"]:
            self.graph.remove_edge(*action)
            self.possible_actions["REMOVE"].remove(action)
            return 1
        elif action_reversed in self.possible_actions["REMOVE"]:
            self.graph.remove_edge(*action_reversed)
            self.possible_actions["REMOVE"].remove(action_reversed)
            return 1
        return 0

    # TEST OLD VERSION
    # def apply_action(self, actions: np.array) -> int:
    #     """Applies the action to the graph, if there is an edge between the two
    #     nodes, it removes it, otherwise it adds it

    #     Parameters
    #     ----------
    #     actions : np.array
    #         List of possible actions, where each element is a real number
    #         between 0 and 1

    #     Returns
    #     -------
    #     budget_consumed : int
    #         Amount of budget consumed
    #     """
    #     # Get the index of the action to apply
    #     index = np.argmax(actions)
    #     #° The number of possible actions is:
    #     #°      len(self.possible_actions["ADD"]) + len(self.possible_actions["REMOVE"])
    #     #° So, if the index is less than the number of possible actions to add,
    #     #° the action to apply is an action to add, otherwise it is an action to remove
    #     if index < self.len_add_actions:
    #         action = self.possible_actions["ADD"][index]
    #         # If the action is (-1,-1) it means that the action has already been
    #         # applied, so we do not need to apply it again
    #         if action == (-1,-1): return 0
    #         # Apply the action
    #         self.graph.add_edge(*action, weight=1)
    #         # Replace the added edge with (-1,-1) in the possible actions, in this way
    #         # we can keep track of the used actions, and we can avoid to add the same
    #         # edge multiple times
    #         self.possible_actions["ADD"][index] = (-1, -1)
    #         return 1
    #     else:
    #         action = self.possible_actions["REMOVE"][index - self.len_add_actions]
    #         # If the action is (-1,-1) it means that the action has already been
    #         # applied, so we do not need to apply it again
    #         if action == (-1, -1): return 0
    #         # Apply the action
    #         self.graph.remove_edge(*action)
    #         # Replace the removed edge with (-1,-1) in the possible actions,
    #         # in order to keep the same length, and to avoid to remove the same
    #         # edge multiple times
    #         self.possible_actions["REMOVE"][index - self.len_add_actions] = (-1, -1)
    #         return 1

    def plot_graph(self) -> None:
        """Plot the graph using matplotlib"""
        import matplotlib.pyplot as plt
        nx.draw(self.graph, with_labels=True)
        plt.show()
