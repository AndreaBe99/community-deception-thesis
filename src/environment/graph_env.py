"""Module for the GraphEnviroment class"""
import sys 
sys.path.append('../../')
from src.community_algs.nmi import NormalizedMutualInformation
from src.community_algs.deception_score import DeceptionScore
from src.community_algs.detection_algs import DetectionAlgorithm
from src.utils import DetectionAlgorithms
from src.utils import HyperParams

from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from typing import List, Tuple
import math
import numpy as np
import networkx as nx
import torch



class GraphEnvironment(object):
    """Enviroment where the agent will act, it will be a graph with a community"""

    def __init__(self, beta: float) -> None:
        """Constructor for Graph Environment

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
        self.objective_function_values = np.zeros(2, dtype=float)
    
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

    def delete_repeat_edges(self, data: Data) -> torch.Tensor:
        """The Data object contains the edge_index tensor, which is a tensor
        of size 2*N, where N is the number of edges. The first row of the tensor
        contains the indices of the source nodes, and the second row contains
        the indices of the destination nodes. Each row of the tensor represents
        an edge in the graph. This function removes the duplicate edges, i.e.
        the edges that are present in both directions, e.g. (0, 1) and (1, 0)

        Parameters
        ----------
        data : Data
            Graph before the duplicate edges have been removed

        Returns
        -------
        data : Data
            Graph after the duplicate edges have been removed
        """
        # TODO: Remove duplicate edges, without reshaping the tensor
        # Reshape the edge_index tensor in N*2
        edge_index = data.edge_index.t().contiguous()
        # Remove the duplicate edges
        sorted_edge_index, _ = torch.sort(edge_index, dim=1)
        edge_index = torch.unique(sorted_edge_index, dim=0)
        data.edge_index = edge_index.t().contiguous()
        return data
    
    def reset(self) -> Data:
        """Reset the environment

        Returns
        -------
        Data
            Graph after the reset
        """
        self.used_edge_budget = 0
        self.exhausted_budget = False
        self.graph = self.graph_copy.copy()
        # self.rewards = 0.0
        
        data = self.delete_repeat_edges(from_networkx(self.graph))
        return data
    
    def apply_action(self, actions: List[float])->int:
        """Applies the action to the graph, if there is an edge between the two 
        nodes, it removes it, otherwise it adds it

        Parameters
        ----------
        actions : List[float]
            List of possible actions, where each element is a real number
            between 0 and 1
        
        Returns
        -------
        budget_consumed : int
            Amount of budget consumed
        """
        # Check if between the two nodes there is an edge:
        #   - If there is an edge, it means that the action is to remove it
        #   - If there is no edge, it means that the action is to add it 

        # Get the index of the maximum value in the action list
        index = actions.index(max(actions))
        # Get the action to apply from self.possible_actions
        # NOTE: the number of possible actions is: 
        #   len(self.possible_actions["ADD"]) + len(self.possible_actions["REMOVE"])
        # So, if the index is less than the number of possible actions to add,
        # the action to apply is an action to add, otherwise it is an action to remove
        if index < len(self.possible_actions["ADD"]):
            action = self.possible_actions["ADD"][index]
        else:
            action = self.possible_actions["REMOVE"][index - len(self.possible_actions["ADD"])]
        
        # TODO: Join th following code with the code above
        # Check if the action is to add or to remove an edge
        if action == (-1,-1):
            budget_consumed = 0
        elif action in self.possible_actions["REMOVE"]:
            self.graph.remove_edge(*action)
            # Replace the removed edge with (-1,-1) in the possible actions, 
            # in order to keep the same length, and to avoid to remove the same
            # edge multiple times
            idx = self.possible_actions["REMOVE"].index(action)
            self.possible_actions["REMOVE"][idx] = (-1,-1)
            budget_consumed = 1
            # print("Remove", action)
        elif action in self.possible_actions["ADD"]:
            self.graph.add_edge(*action, weight=1)
            # Replace the added edge with (-1,-1) in the possible actions, in this way
            # we can keep track of the used actions, and we can avoid to add the same
            # edge multiple times
            idx = self.possible_actions["ADD"].index(action)
            self.possible_actions["ADD"][idx] = (-1,-1)
            budget_consumed = 1
            # print("Add", action)
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
        self.graph_copy = graph.copy()
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
        self.used_edge_budget = 0
        self.exhausted_budget = False
        
        # Compute the set of possible actions
        self.possible_actions = self.get_possible_actions(
            self.graph, self.community_target)
        
        # NOTE: Old method to compute the reward, i.e. the difference between the new and the old deception score
        # Set first objective function value
        # self.objective_function_values[0] = self.deception.compute_deception_score(self.community_structure_old)
        # if self.training:
        #    self.objective_function_values[0] = self.objective_function_values[0] * self.rewards_scale_multiplier 
    
    def step(self, actions: List[float]) -> Tuple[Data, float]:
        """Step function for the environment

        Parameters
        ----------
        actions : List[float]
            Actions to take on the graph, which is a list longer as the number
            of possible actions, where each element is a real number between
            0 and 1
        Returns
        -------
        self.graph, self.rewards: Tuple[Data, float]
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
        # Compute the reward, using the deception score and the NMI score
        reward = self.get_reward(deception_score, nmi)
        
        # Update the used edge budget
        self.used_edge_budget += (remaining_budget - updated_budget)
        
        # NOTE: Old method to compute the reward, i.e. the difference between the new and the old deception score
        # Update the objective function value, i.e. compute the new deception score
        # self.objective_function_values[1] = self.deception.compute_deception_score(self.community_structure_new)
        # if self.training:
        #    self.objective_function_values[1] *= self.rewards_scale_multiplier
        
        # Compute the reward as the difference between the Deception Score
        # before and after the action
        # reward = self.objective_function_values[0] - self.objective_function_values[1]
        
        if abs(reward) < self.eps:
            reward = 0.0
        self.rewards = reward
        
        data = self.delete_repeat_edges(from_networkx(self.graph))
        return data, self.rewards