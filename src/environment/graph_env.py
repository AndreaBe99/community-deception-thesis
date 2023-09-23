"""Module for the GraphEnviroment class"""
from src.community_algs.detection_algs import CommunityDetectionAlgorithm
from src.utils.utils import HyperParams, SimilarityFunctionsNames, Utils
from src.community_algs.metrics.similarity import CommunitySimilarity, GraphSimilarity
from typing import List, Tuple, Callable

from karateclub import Node2Vec

import math
import networkx as nx
import random
import time
import torch


class GraphEnvironment(object):
    """Enviroment where the agent will act, it will be a graph with a community"""

    def __init__(
        self,
        graph_path: str = HyperParams.GRAPH_NAME.value,
        community_detection_algorithm: str = HyperParams.DETECTION_ALG_NAME.value,
        beta: float = HyperParams.BETA.value,
        tau: float = HyperParams.TAU.value,
        community_similarity_function: str = SimilarityFunctionsNames.SOR.value,
        graph_similarity_function: str = SimilarityFunctionsNames.JAC_1.value,
    ) -> None:
        """Constructor for Graph Environment
        Parameters
        ----------
        graph_path : str, optional
            Path of the graph to load, by default HyperParams.GRAPH_NAME.value
        community_detection_algorithm : str
            Name of the community detection algorithm to use
        beta : float, optional
            Hyperparameter for the edge budget, value between 0 and 100
        tau : float, optional
            Strength of the deception constraint, value between 0 and 1, with 1
            we have a soft constraint, hard constraint otherwise, by default
            HyperParams.T.value
        community_similarity_function : str, optional
            Name of the community similarity function to use, by default
            SimilarityFunctionsNames.SOR.value
        graph_similarity_function : str, optional
            Name of the graph similarity function to use, by default
            SimilarityFunctionsNames.JAC_1.value
        """
        random.seed(time.time())
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # ° ---- GRAPH ---- ° #
        self.env_name = None
        self.graph = None
        self.original_graph = None
        self.old_graph = None
        self.n_connected_components = None
        self.set_graph(graph_path)
        
        # ° ---- NODE FEATURES ---- ° #
        # Set the node features of the graph, using Node2Vec
        self.embedding_model = None
        self.embedding = None
        # self.set_node_features()

        # ° ---- HYPERPARAMETERS ---- ° #
        assert beta >= 0 and beta <= 100, "Beta must be between 0 and 100"
        assert tau >= 0 and tau <= 1, "T value must be between 0 and 1"
        # Percentage of edges to remove
        self.beta = beta
        self.tau = tau
        # Weights for the reward and the penalty
        self.lambda_metric = None  # lambda_metric
        self.alpha_metric = None  # alpha_metric

        # ° ---- SIMILARITY FUNCTIONS ---- ° #
        self.community_similarity = None
        self.graph_similarity = None
        self.set_similarity_funtions(community_similarity_function, graph_similarity_function)

        # ° ---- COMMUNITY DETECTION ---- ° #
        self.detection_alg = None
        self.detection = None
        self.old_penalty_value = None
        self.original_community_structure = None
        self.old_community_structure = None
        self.new_community_structure = None
        self.set_communities(community_detection_algorithm)

        # ° ---- COMMUNITY DECEPTION ---- ° #
        self.community_target = None
        self.node_target = None
        self.set_targets()

        # ° ---- REWIRING STEP ---- ° #
        self.edge_budget = 0
        self.used_edge_budget = 0
        self.max_steps = 0
        self.stop_episode = False
        self.rewards = 0
        self.old_rewards = 0
        self.possible_actions = None
        self.len_add_actions = None
        self.set_rewiring_budget()
        
        # ° ---- PRINT ENVIRONMENT INFO ---- ° #
        # Print the environment information
        self.print_env_info()


    ############################################################################
    #                       EPISODE RESET FUNCTIONS                            #
    ############################################################################
    def reset(self, graph_reset=True) -> nx.Graph:
        """
        Reset the environment
        
        Parameters
        ----------
        graph_reset : bool, optional
            Whether to reset the graph to the original state, by default True

        Returns
        -------
        self.graph : nx.Graph
            Graph state after the reset, i.e. the original graph
        """
        self.used_edge_budget = 0
        self.stop_episode = False
        self.rewards = 0
        self.old_rewards = 0
        if graph_reset:
            self.graph = self.original_graph.copy()
        self.old_graph = None
        self.old_penalty_value = 0
        self.old_community_structure = self.original_community_structure
        self.possible_actions = self.get_possible_actions()
        return self.graph

    def change_target_node(self, node_target: int = None) -> None:
        """
        Change the target node to remove from the community

        Parameters
        ----------
        node_target : int, optional
            Node to remove from the community, by default None
        """
        if node_target is None:
            # Choose a node randomly from the community
            old_node = self.node_target
            while self.node_target == old_node:
                random.seed(time.time())
                self.node_target = random.choice(self.community_target)
        else:
            self.node_target = node_target

    def change_target_community(
            self,
            community: List[int] = None,
            node_target: int = None) -> None:
        """
        Change the target community from which we want to hide the node

        Parameters
        ----------
        community : List[int]
            Community of node we want to remove from it
        node_target : int
            Node to remove from the community
        """
        if community is None:
            # Select randomly a new community target different from the last one
            old_community = self.community_target.copy()
            done = False
            while not done:
                random.seed(time.time())
                self.community_target = random.choice(
                    self.original_community_structure.communities)
                # Check condition on new community
                if (len(self.community_target) > 1 and \
                        self.community_target != old_community) or \
                            len(self.original_community_structure.communities) < 2:
                    done = True
            del old_community
        else:
            self.community_target = community
        # Change the target node to remove from the community
        self.change_target_node(node_target=node_target)

    ############################################################################
    #                      EPISODE STEP FUNCTIONS                              #
    ############################################################################
    def step(self, action: int) -> Tuple[nx.Graph, float, bool, bool]:
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
            If the budget for the graph rewiring is exhausted, or the target
            node does not belong to the community anymore, the episode is finished
        done : bool
            Whether the episode is finished, if the target node does not belong
            to the community anymore, the episode is finished.
        """
        # ° ---- ACTION ---- ° #
        # Save the graph state before the action, used to compute the metrics
        self.old_graph = self.graph.copy()
        # Take action, add/remove the edge between target node and the model output
        budget_consumed = self.apply_action(action)
        # Set a negative reward if the action has not been applied
        if budget_consumed == 0:
            self.rewards = -1
            # The state is the same as before
            # return self.data_pyg, self.rewards, self.stop_episode
            return self.graph, self.rewards, self.stop_episode, False

        # ° ---- COMMUNITY DETECTION ---- ° #
        # Compute the community structure of the graph after the action
        self.new_community_structure = self.detection.compute_community(
            self.graph)

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
            # if not done:
            #    self.rewards = -2

        self.old_community_structure = self.new_community_structure
        return self.graph, self.rewards, self.stop_episode, done

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

    
    ############################################################################
    #                       SETTERS FUNCTIONS                                  #
    ############################################################################
    def set_graph(self, graph_path: str) -> None:
        """Set the graph of the environment"""
        # Load the graph from the dataset folder
        if graph_path is None:
            # Generate a synthetic graph
            graph, graph_path = Utils.generate_lfr_benchmark_graph()
        else:
            graph = Utils.import_mtx_graph(graph_path)

        self.env_name = graph_path.split("/")[-1].split(".")[0]
        self.graph = self.set_node_features(graph)
        
        # Save the original graph to restart the rewiring process at each episode
        self.original_graph = self.graph.copy()
        # Save the graph state before the action, used to compute the metrics
        self.old_graph = None
        # Get the Number of connected components
        self.n_connected_components = nx.number_connected_components(self.graph)
    
    def set_node_features(self, graph)-> None:
        """Set the node features of the graph, using Node2Vec"""
        print("*"*20, "Environment Information", "*"*20)
        print("* Graph Name:", self.env_name)
        print("*", graph)
        print("* * Compute Node Embedding using Node2Vec for nodes features")
        print("* * ...")
        # Build node features using Node2Vec, set the embedding dimension to 128.
        self.embedding_model = Node2Vec(dimensions=HyperParams.EMBEDDING_DIM.value)
        self.embedding_model.fit(graph)
        print("* * End Embedding Computation")
        self.embedding = self.embedding_model.get_embedding()
        # Add the embedding to the graph
        for node in graph.nodes():
            graph.nodes[node]["x"] = torch.tensor(self.embedding[node])
            # delete all the other features
        # Delete edges without "weight" attribute
        for edge in graph.edges():
            if "weight" not in graph.edges[edge]:
                graph.remove_edge(*edge)
        return graph
        
    def set_similarity_funtions(
        self, 
        community_similarity_function: str, 
        graph_similarity_function: str)-> None:
        """
        Set the similarity functions to use to compare the communities and
        the graphs
        """
        # Select the similarity function to use to compare the communities
        self.community_similarity = CommunitySimilarity(
            community_similarity_function).select_similarity_function()
        self.graph_similarity = GraphSimilarity(
            graph_similarity_function).select_similarity_function()
    
    def set_communities(self, community_detection_algorithm)-> None:
        """
        Set the community detection algorithm to use, and compute the community
        structure of the graph before the deception actions.
        """
        self.detection_alg = community_detection_algorithm
        # Community Algorithms objects
        self.detection = CommunityDetectionAlgorithm(community_detection_algorithm)
        # Metrics
        self.old_penalty_value = 0
        # Compute the community structure of the graph, before the action,
        # i.e. before the deception
        self.original_community_structure = self.detection.compute_community(self.graph)
        # ! It is a NodeClustering object
        self.old_community_structure = self.original_community_structure

    def set_targets(self) -> None:
        """
        Set the target community as the community with the highest number 
        of nodes, and the target node as a random node in the community
        """
        # Choose one of the communities found by the algorithm, as initial
        # community we choose the community with the highest number of nodes
        self.community_target = max(
            self.original_community_structure.communities, key=len)
        if len(self.community_target) <= 1:
            raise Exception("Community target must have at least two node.")
        # Choose a node randomly from the community, as initial node to remove
        self.node_target = random.choice(self.community_target)
    
    def set_rewiring_budget(self) -> None:
        """Set the rewiring budget for the graph, and the valid actions"""
        # Compute the action budget for the graph
        self.edge_budget = self.get_edge_budget()
        # Amount of budget used
        self.used_edge_budget = 0
        # Max Rewiring Steps during an episode, set a limit to avoid infinite
        # episodes in case the agent does not find the target node
        self.max_steps = self.edge_budget # * HyperParams.MAX_STEPS_MUL.value
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
        # TEST: Three different ways to compute the edge budget
        
        # 1. Mean degree of the graph times the parameter beta
        return int(self.graph.number_of_edges() / self.graph.number_of_nodes() * self.beta)
        
        # 2. Percentage of edges of the whole graph
        # return int(math.ceil((self.graph.number_of_edges() * self.beta / 100)))
        
        # 3. Percentage of edges of the whole graph divided by the number of nodes in the community
        # return int(math.ceil((self.graph.number_of_edges() * self.beta / 100) / len(self.community_target)))

    def get_penalty(self) -> float:
        """
        Compute the metrics and return the penalty to subtract from the reward

        Returns
        -------
        penalty: float
            Penalty to subtract from the reward
        """
        # ° ---- COMMUNITY DISTANCE ---- ° #
        community_distance = self.new_community_structure.normalized_mutual_information(
            self.old_community_structure).score
        # In NMI 1 means that the two community structures are identical,
        # 0 means that they are completely different
        # We want to maximize the NMI, so we subtract it from 1
        community_distance = 1 - community_distance
        # ° ---- GRAPH DISTANCE ---- ° #
        graph_distance = self.graph_similarity(self.graph, self.old_graph)
        # ° ---- PENALTY ---- ° #
        assert self.alpha_metric is not None, "Alpha metric is None, must be set in grid search"
        penalty = self.alpha_metric * community_distance + \
            (1 - self.alpha_metric) * graph_distance
        # Subtract the metric value of the previous step
        penalty -= self.old_penalty_value
        # Update with the new values
        self.old_penalty_value = penalty
        return penalty

    def get_reward(self) -> Tuple[float, bool]:
        """
        Computes the reward for the agent, it is a 0-1 value function, if the
        target node still belongs to the community, the reward is 0 minus the
        penalty, otherwise the reward is 1 minus the penalty.

        As new community target after the action, we consider the community
        that contains the target node, if this community satisfies the deception
        constraint, the episode is finished, otherwise not.

        Returns
        -------
        reward : float
            Reward of the agent
        done : bool
            Whether the episode is finished, if the target node does not belong
            to the community anymore, the episode is finished
        """
        assert self.lambda_metric is not None, "Lambda metric is None, must be set in grid search"
        # Get the target community in the new community structure that
        # contains the target node
        for community in self.new_community_structure.communities:
            if self.node_target in community:
                new_community_target = community
                break
        assert new_community_target is not None, "New community target is None"
        # ° ---------- PENALTY ---------- ° #
        # Compute the metric to subtract from the reward
        penalty = self.get_penalty()
        # If the target node does not belong to the community anymore,
        # the episode is finished
        if len(new_community_target) == 1:
            reward = 1 - (self.lambda_metric * penalty)
            return reward, True
        # ° ---- COMMUNITY SIMILARITY ---- ° #
        # Remove target node from the communities, but first copy the lists
        # to avoid modifying them
        new_community_target_copy = new_community_target.copy()
        new_community_target_copy.remove(self.node_target)
        community_target_copy = self.community_target.copy()
        community_target_copy.remove(self.node_target)
        # Compute the similarity between the new communities
        community_similarity = self.community_similarity(
            new_community_target_copy,
            community_target_copy,
        )
        # Delete the copies
        del new_community_target_copy, community_target_copy
        # ° ---------- REWARD ---------- ° #
        if community_similarity <= self.tau:
            # We have reached the deception constraint, the episode is finished
            reward = 1 - (self.lambda_metric * penalty)
            return reward, True
        reward = 0 - (self.lambda_metric * penalty)
        return reward, False

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
    #                           ENVIRONMENT INFO                               #
    ############################################################################
    def print_env_info(self) -> None:
        """Print the environment information"""
        print("* Community Detection Algorithm:", self.detection_alg)
        print("* Number of communities found:",
              len(self.original_community_structure.communities))
        # print("* Rewiring Budget:", self.edge_budget, "=", self.beta, "*", self.graph.number_of_edges(), "/ 100",)
        print("* BETA - Rewiring Budget: (n_edges/n_nodes)*BETA =",
              self.graph.number_of_edges(), "/",
              self.graph.number_of_nodes(), "*", self.beta, "=",
              int(self.graph.number_of_edges() / self.graph.number_of_nodes())*self.beta)
        print("* TAU - Weight of the Deception Constraint:", self.tau)
        print("*", "-"*58, "\n")
