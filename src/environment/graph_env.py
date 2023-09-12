"""Module for the GraphEnviroment class"""
from src.community_algs.metrics.nmi import NormalizedMutualInformation
from src.community_algs.metrics.deception_score import DeceptionScore
from src.community_algs.detection_algs import CommunityDetectionAlgorithm
from src.community_algs.metrics.safeness import Safeness
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
import random
import time



class GraphEnvironment(object):
    """Enviroment where the agent will act, it will be a graph with a community"""

    def __init__(
            self, 
            graph: nx.Graph, 
            env_name: str,
            community_detection_algorithm: str,
            beta: float = HyperParams.BETA.value) -> None:
        """Constructor for Graph Environment
        Parameters
        ----------
        graph : nx.Graph
            Graph to use for the environment
        env_name : str
            Name of the environment, i.e. name of the dataset
        community_detection_algorithm : str
            Name of the community detection algorithm to use
        beta : float, optional
            Percentage of edges to remove, by default HyperParams.BETA.value
        """
        self.graph = graph
        # Save the original graph to restart the rewiring process at each episode
        self.original_graph = graph.copy()
        # Save the graph state before the action, used to compute the metrics
        self.old_graph = None
        # Get the Number of connected components
        self.n_connected_components = nx.number_connected_components(graph)
        
        assert beta >= 0 and beta <= 100, "Beta must be between 0 and 100"
        # Percentage of edges to remove
        self.beta = beta
        # Weights for the reward and the penalty
        self.lambda_metric = None # lambda_metric
        self.alpha_metric = None # alpha_metric
        # Name of the environment and the community detection algorithm
        self.env_name = env_name
        self.detection_alg = community_detection_algorithm
        
        # Community Algorithms objects
        self.detection = CommunityDetectionAlgorithm(community_detection_algorithm)
        
        # Metrics
        # self.deception = DeceptionScore(self.community_target)
        # self.safeness = Safeness(self.graph, self.community_target, self.node_target)
        # self.nmi = NormalizedMutualInformation()
        self.old_metric_value = 0
        
        # Compute the community structure of the graph, before the action,
        # i.e. before the deception
        self.original_community_structure = self.detection.compute_community(graph)
        # ! It is a NodeClustering object
        self.old_community_structure = self.original_community_structure

        # Choose one of the communities found by the algorithm, as initial 
        # community we choose the community with the highest number of nodes
        self.community_target = max(
            self.original_community_structure.communities, key=len)
        # Index of the target community in the list of communities
        self.idx_community_target = self.original_community_structure.communities.index(
            self.community_target)
        # Choose a node randomly from the community, as initial node to remove
        self.node_target = self.community_target[random.randint(0, len(self.community_target)-1)]
        
        # Compute the edge budget for the graph
        self.edge_budget = self.get_edge_budget()
        # Amount of budget used
        self.used_edge_budget = 0
        # Max Rewiring Steps during an episode, set a limit to avoid infinite episodes
        # in case the agent does not find the target node
        self.max_steps = self.graph.number_of_edges()#*2
        # Whether the budget for the graph rewiring is exhausted, or the target
        # node does not belong to the community anymore
        self.stop_episode = False
        self.rewards = 0
        # Reward of the previous step
        self.old_rewards = 0
        
        # Compute the set of possible actions
        self.possible_actions = self.get_possible_actions()
        # Length of the list of possible actions to add
        self.len_add_actions = len(self.possible_actions["ADD"])
        
        # Print the environment information
        print("* Community Detection Algorithm:", self.detection_alg)
        print("* Number of communities found:",
            len(self.original_community_structure.communities))
        print("* Initial Community Target:", self.community_target)
        print("* Initial Index of the Community Target:", self.idx_community_target)
        print("* Initial Nodes Target:", self.node_target)
        print("* Number of possible actions:", 
            len(self.possible_actions["ADD"]) + len(self.possible_actions["REMOVE"]))
        print("* Rewiring Budget:", self.edge_budget, "=",
            self.beta, "*", self.graph.number_of_edges(), "/ 100",)
        print("*", "-"*58, "\n")

    ############################################################################
    #                       GETTERS FUNCTIONS                                  #
    ############################################################################
    def get_edge_budget(self) -> int:
        """
        Computes the edge budget for each graph

        Returns
        -------
        int
            Edge budgets of the graph
        """
        return int(math.ceil((self.graph.number_of_edges() * self.beta / 100)))
    
    def get_metrics(self) -> float:
        """
        Compute the metrics and return the penalty to subtract from the reward

        Returns
        -------
        penalty: float
            Penalty to subtract from the reward
        """
        # Compute the new Community Structure after the action
        self.new_community_structure = self.detection.compute_community(
            self.graph)
        # Search the index of the target community in the new list of communities
        self.idx_community_target = self.get_community_target_idx(
            self.new_community_structure.communities,
            self.community_target)
        
        # ° Distance between community structures
        community_metric = self.new_community_structure.normalized_mutual_information(
            self.old_community_structure).score
        # We want to maximize the NMI, so we subtract it from 1
        community_metric = 1 - community_metric
        
        # ° Distance between graphs
        # graph_metric = nx.graph_edit_distance(self.graph, self.old_graph)
        # Faster approximation of the graph edit distance
        # graph_metric = next(nx.optimize_graph_edit_distance(self.graph, self.old_graph))

        # Normalize the graph edit distance using a null graph:
        #   GED(G1,G2)/[GED(G1,G0) + GED(G2,G0)]
        # with G0 = null graph
        # g_dist_1 = next(nx.optimize_graph_edit_distance(self.graph, nx.null_graph()))
        # g_dist_2 = next(nx.optimize_graph_edit_distance(self.old_graph, nx.null_graph()))
        # graph_metric /= (g_dist_1 + g_dist_2)
        
        # TEST
        def jaccard_similarity(g: nx.Graph, h: nx.Graph) -> float:
            """Compute the Jaccard Similarity between two graphs"""
            i = set(g).intersection(h)
            return round(len(i) / (len(g) + len(h) - len(i)), 3)
        
        graph_metric = jaccard_similarity(self.graph.edges(), self.old_graph.edges())
        # We want to maximize the Jaccard Similarity, so we subtract it from 1
        graph_metric = 1 - graph_metric
        
        # ° Compute metrics with the weight alpha
        assert self.alpha_metric is not None, "Alpha metric is None, must be set in grid search"
        metric = self.alpha_metric * community_metric + (1 - self.alpha_metric) * graph_metric

        # ° Subtract the metric value of the previous step
        metric -= self.old_metric_value
        # Update with the new values
        self.old_metric_value = metric
        self.old_community_structure = self.new_community_structure
        return metric

    def get_reward(self) -> Tuple[float, bool]:
        """
        Computes the reward for the agent, it is a 0-1 value function, if the
        target node still belongs to the community, the reward is 0 minus the
        penalty, otherwise the reward is 1 minus the penalty.

        Returns
        -------
        reward : float
            Reward of the agent
        done : bool
            Whether the episode is finished, if the target node does not belong
            to the community anymore, the episode is finished
        """
        assert self.lambda_metric is not None, "Lambda metric is None, must be set in grid search"
        
        # Compute the metric to subtract from the reward
        metric = self.get_metrics()
        
        communities_list = self.new_community_structure.communities
        if self.node_target in communities_list[self.idx_community_target]:
            # if the intersection between the target community and the new
            # community target contains all the nodes of the original target 
            # community the node still belongs to the community
            if set(communities_list[self.idx_community_target]).issuperset(
                    set(self.community_target)):
                # The episode is not finished            
                reward = -self.lambda_metric * metric
                return reward, False
            # If the new community target does not contain all the nodes of the
            # original target community, the node does not belong to the community
            # anymore, the episode is finished
            pass
        reward = 1 - (self.lambda_metric * metric)
        return reward, True
    
    def get_community_target_idx(
                self,
                community_structure: List[List[int]],
                community_target: List[int]) -> int:
        """
        Returns the index of the target community in the list of communities.
        As the target community after a rewiring action we consider the community
        with the highest number of nodes equal to the initial community.
        
        Parameters
        ----------
        community_structure : List[List[int]]
            List of communities
        community_target : List[int]
            Community of node we want to remove from it
        
        Returns
        -------
        max_list_idx : int
            Index of the target community in the list of communities
        """
        max_count = 0
        max_list_idx = 0
        for i, lst in enumerate(community_structure):
            count = sum(1 for x in lst if x in community_target)
            if count > max_count:
                max_count = count
                max_list_idx = i
        return max_list_idx

    def get_possible_actions(self) -> dict:
        """
        Returns all the possible actions that can be applied to the graph
        given a source node (self.node_target). The possible actions are:
            - Add an edge between the source node and a node outside the community
            - Remove an edge between the source node and a node inside the community
        
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

        u = self.node_target
        for v in self.graph.nodes():
            if u == v:
                continue
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
        return possible_actions
    
    ############################################################################
    #                       EPISODE RESET FUNCTIONS                            #
    ############################################################################
    def reset(self) -> Data:
        """
        Reset the environment

        Returns
        -------
        adj_matrix : torch.Tensor
            Adjacency matrix of the graph
        """
        self.used_edge_budget = 0
        self.stop_episode = False
        self.rewards = 0
        self.old_rewards = 0
        self.graph = self.original_graph.copy()
        self.old_graph = None
        self.old_metric_value = 0
        self.old_community_structure = self.original_community_structure
        self.possible_actions = self.get_possible_actions()
        
        # Change the target community and the target node at each episode
        self.change_target_community()

        return self.graph
        # Return a PyG Data object
        # self.data_pyg = from_networkx(self.graph)
        # # Initialize the node features
        # # self.data_pyg.x = torch.randn([self.data_pyg.num_nodes, HyperParams.STATE_DIM.value])
        # self.data_pyg.x = torch.eye(self.data_pyg.num_nodes, dtype=torch.float)
        # # Initialize the batch
        # self.data_pyg.batch = torch.zeros(self.data_pyg.num_nodes).long()
        # return self.data_pyg
    
    def change_target_node(self, node_target: int=None) -> None:
        """
        Change the target node to remove from the community

        Parameters
        ----------
        node_target : int, optional
            Node to remove from the community, by default None
        """
        if node_target is None:
            # Choose a node randomly from the community
            idx_node = random.randint(0, len(self.community_target)-1)
            self.node_target = self.community_target[idx_node]
        else:
            self.node_target = node_target
    
    def change_target_community(
        self, 
        community: List[int]=None, 
        idx_community: int=None,
        node_target: int=None) -> None:
        """
        Change the target community from which we want to hide the node

        Parameters
        ----------
        community : List[int]
            Community of node we want to remove from it
        idx_community : int
            Index of the community in the list of communities
        """
        if community is None:
            # Choose a community randomly from the list of communities
            self.idx_community_target = random.randint(
                0, len(self.original_community_structure.communities)-1)
            self.community_target = self.original_community_structure.communities[
                self.idx_community_target]
        else:
            self.community_target = community
            self.idx_community_target = idx_community
        self.change_target_node(node_target=node_target)
    
    ############################################################################
    #                      EPISODE STEP FUNCTIONS                              #
    ############################################################################
    def step(self, action: int) -> Tuple[nx.Graph, float, bool]:
        """
        Step function for the environment
        
        Parameters
        ----------
        action : int
            Integer representing a node in the graph, it will be the destination
            node of the rewiring action (out source node is always the target node).
            
        Returns
        -------
        self.graph : nx.Graph
            Graph state after the action
        self.rewards : float
            Reward of the agent
        self.stop_episode : bool
            Whether the episode is finished, if the target node does not belong
            to the community anymore, or the budget for the graph rewiring is
            exhausted, the episode is finished
        """
        # ° ---- ACTION ---- ° #
        # Save the graph state before the action, used to compute the metrics
        self.old_graph = self.graph.copy()
        # Take action, add/remove the edge between target node and the model output
        budget_consumed = self.apply_action(action)
        # Set a negative reward if the action has not been applied
        if budget_consumed == 0:
            self.rewards = -2
            # The state is the same as before
            # return self.data_pyg, self.rewards, self.stop_episode
            return self.graph, self.rewards, self.stop_episode
        
        # ° ---- REWARD ---- ° #
        self.rewards, done = self.get_reward()
        # If the target node does not belong to the community anymore, 
        # the episode is finished
        if done:
            self.stop_episode = True
        
        # ° ---- BUDGET ---- ° #
        # Compute used budget
        self.used_edge_budget += budget_consumed
        # If the budget for the graph rewiring is exhausted, stop the episode
        if self.edge_budget - self.used_edge_budget < 1:
            self.stop_episode = True
            # If the budget is exhausted, and the target node still belongs to
            # the community, the reward is negative
            if not done:
                self.rewards = -1
        
        return self.graph, self.rewards, self.stop_episode
        # # ° ---- PyG Data ---- ° #
        # # OLD CODE
        # # data = from_networkx(self.graph)
        # # data.x = self.data_pyg.x
        # # data.batch = self.data_pyg.batch
        # # self.data_pyg = data
        
        # # Avoid to use from_networkx, in my test is slower than the following code
        # # to compute the edge_index
        # edge_list = nx.to_edgelist(self.graph)
        # # remove weights
        # edge_list = [[e[0], e[1]] for e in edge_list]
        # edge_list += [[e[1], e[0]] for e in edge_list]
        # # order the list, first by first element, then by second element
        # edge_list = sorted(edge_list, key=lambda x: (x[0], x[1]))
        # # Create tensor
        # edge_list = torch.tensor(edge_list)
        # edge_list_t = torch.transpose(edge_list, 0, 1)
        # del edge_list
        # self.data_pyg.edge_index = edge_list_t
        # return self.data_pyg, self.rewards, self.stop_episode
    
    def apply_action(self, action: int) -> int:
        """
        Applies the action to the graph, if there is an edge between the two 
        nodes, it removes it, otherwise it adds it

        Parameters
        ----------
        action : int
            Integer representing a node in the graph, it will be the destination
            node of the rewiring action (out source node is always the target node).
        
        Returns
        -------
        budget_consumed : int
            Amount of budget consumed, 1 if the action has been applied, 0 otherwise
        """
        action = (self.node_target, action)   
        # We need to take into account both the actions (u,v) and (v,u)
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
