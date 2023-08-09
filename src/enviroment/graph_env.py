"""Module for the GraphEnviroment class"""
import sys 
sys.path.append('../../')
from src.community_algs.nmi import NormalizedMutualInformation
from src.community_algs.deception_score import DeceptionScore
from src.community_algs.detection_algs import DetectionAlgorithm
from src.utils import DetectionAlgorithms
from src.utils import HyperParams

from typing import List, Tuple
import math
import numpy as np
import networkx as nx


class GraphEnviroment(object):
    """Enviroment where the agent will act, it will be a graph with a community"""

    def __init__(self, beta: float) -> None:
        """Constructor for GraphEnviroment

        Parameters
        ----------
        beta : float
            Percentage of edges to rewire/update, real number between 1 and 100
        """
        assert beta >= 0 and beta <= 100, "Beta must be between 0 and 100"
        self.beta = beta
        self.eps = 1e-8
        self.rewards_scale_multiplier = 10 # 0, 10, 100
        
        self.nmi = NormalizedMutualInformation()
        # Setup later, with the setup function
        self.graph = None
        self.deception = None
        self.detection = None
        # Community to hide
        self.community_target = None
        # Community Structure before the action
        self.community_structure_old = None
        # Community Structure after the action
        self.community_structure_new = None
        self.n_connected_components = None
        self.training = None
        self.rewards = None
        self.edge_budget = None
        self.used_edge_budget = None
        self.exhausted_budget = None
        
        # Possible actions
        self.possible_actions = None
        
        # objective_function_values[0] --> Old Value, i.e. before the action
        # objective_function_values[1] --> New Value, i.e. after the action
        # The Value is the Deception Score
        self.objective_function_values = np.zeros(2, dtype=np.float)
    
    @staticmethod
    def get_possible_actions(
        graph: nx.Graph, 
        community: List[int])->List[Tuple[int, int]]:
        """Returns the possible actions that can be applied to the graph

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

        for u, v, _ in graph.edges(data=True):
            # We can remove an edge iff both nodes are in the community
            if in_community(u) and in_community(v):
                possible_actions["REMOVE"].append((u,v))
            # We can add an edge iff one node is in the community and the other is not
            elif (in_community(u) and out_community(v)) \
                or (out_community(u) and in_community(v)):
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
        return int(math.ceil((graph.n_edges * beta / 100)))

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
    
    def apply_action(self, action:Tuple[int, int])->int:
        """Applies the action to the graph, if there is an edge between the two 
        nodes, it removes it, otherwise it adds it

        Parameters
        ----------
        graph : nx.Graph
            NetworkX Graph object to apply the action to
        
        Returns
        -------
        budget_consumed : int
            Amount of budget consumed
        """
        # Check if between the two nodes there is an edge:
        #   - If there is an edge, it means that the action is to remove it
        #   - If there is no edge, it means that the action is to add it 
        if action in self.possible_actions["REMOVE"]:
            self.graph.remove_edge(action[0], action[1])
            budget_consumed = 1
        elif action in self.possible_actions["ADD"]:
            self.graph.add_edge(action[0], action[1])
            budget_consumed = 1
        else:
            budget_consumed = 0
        return budget_consumed
    
    def setup(
        self,
        graph: nx.Graph,
        community: List[int],
        community_detection_algorithm: str=DetectionAlgorithms.LOUV.value,
        training: bool=False) -> None:
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
        self.community_target = community
        self.training = training
        self.rewards = 0.0
        
        # Get the Number of connected components
        self.n_connected_components = nx.number_connected_components(self.graph)
        
        self.detection = DetectionAlgorithm(community_detection_algorithm)
        self.deception = DeceptionScore(self.community_target)
        
        # Compute the community structure of the graph, before the action, 
        # i.e. before the deception
        self.community_structure_old = self.detection.compute_community(
            self.graph)
        
        # Compute the edge budget for the graph
        self.edge_budget = self.get_edge_budget(self.graph, self.beta)
        self.used_edge_budget = 0.0
        self.exhausted_budget = False
        
        # Compute the set of possible actions
        self.possible_actions = self.get_possible_actions(
            self.graph, self.community_target)
        
        # NOTE: Old method to compute the reward, i.e. the difference between the new and the old deception score
        # Set first objective function value
        # self.objective_function_values[0] = self.deception.compute_deception_score(self.community_structure_old)
        # if self.training:
        #    self.objective_function_values[0] = self.objective_function_values[0] * self.rewards_scale_multiplier 
    
    def step(self, action: Tuple(int, int)) -> Tuple[float, bool]:
        """Step function for the environment

        Parameters
        ----------
        action : Tuple(int, int)
            Action to take on the graph, which is a tuple containing the
            nodes to rewire

        Returns
        -------
        Tuple[float, bool]
            Tuple containing the reward and whether the episode is done
        """
        
        # Check if the budget of rewiring has been exhausted
        if self.exhausted_budget:
            print("Budget exhausted")
            return self.rewards, True
        
        # Compute the remaining budget
        remaining_budget = self.edge_budget - self.used_edge_budget
        
        # Take action
        budget_consumed = self.apply_action(action)
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
        # Compute the reward, using the deception score and the NMI score
        reward = self.get_reward(deception_score, nmi)
        
        # Update the used edge budget
        self.used_edge_budget += (remaining_budget - updated_budget)
        
        # NOTE: Old method to compute the reward, i.e. the difference between the new and the old deception score
        # Update the objective function value, i.e. compute the new deception score
        # self.objective_function_values[1] = self.deception.compute_deception_score(self.community_structure_new)
        # if self.training:
        #    self.objective_function_values[1] *= self.rewards_scale_multiplier
        
        # Compute the reward as the difference between the Decption Score 
        # before and after the action
        # reward = self.objective_function_values[0] - self.objective_function_values[1]
        
        if abs(reward) < self.eps:
            reward = 0.0
        self.rewards = reward