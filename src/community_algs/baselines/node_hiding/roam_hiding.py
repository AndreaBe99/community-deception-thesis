import sys
sys.path.append("../../../")
from src.community_algs.detection_algs import CommunityDetectionAlgorithm
from src.community_algs.baselines.community_hiding.test_safeness import Safeness
from src.utils.utils import Utils, FilePaths, DetectionAlgorithmsNames, HyperParams

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random


class RoamHiding():
    """Given a network and a source node v,our objective is to conceal the 
    importance of v by decreasing its centrality without compromising its
    influence over the network.
    
    From the article "Hiding Individuals and Communities in a Social Network".
    """
    def __init__(
        self, 
        graph: nx.Graph, 
        target_node: int, 
        edge_budget: int,
        detection_alg: str) -> None:
        self.graph = graph
        self.target_node = target_node
        self.edge_budget = edge_budget
        self.detection_alg = CommunityDetectionAlgorithm(detection_alg)
    
    def roam_heuristic(self, budget: int) -> tuple:
        """
        The ROAM heuristic given a budget b:
            - Step 1: Remove the link between the source node, v, and its 
            neighbour of choice, v0;
            - Step 2: Connect v0 to b − 1 nodes of choice, who are neighbours 
            of v but not of v0 (if there are fewer than b − 1 such neighbours, 
            connect v0 to all of them).

        Returns
        -------
        graph : nx.Graph
            The graph after the ROAM heuristic.
        """
        graph = self.graph.copy()
        # ° --- Step 1 --- ° #
        target_node_neighbours = list(graph.neighbors(self.target_node))
        if len(target_node_neighbours) == 0:
            print("No neighbours for the target node", self.target_node)
            return graph, self.detection_alg.compute_community(graph)
        
        # Choose v0 as the neighbour of target_node with the most connections
        v0 = target_node_neighbours[0]
        for v in target_node_neighbours:
            if graph.degree[v] > graph.degree[v0]:
                v0 = v
        # v0 = random.choice(target_node_neighbours)    # Random choice
        # Remove the edge between v and v0
        graph.remove_edge(self.target_node, v0)
        
        # ° --- Step 2 --- ° #
        # Get the neighbours of v0
        v0_neighbours = list(graph.neighbors(v0))
        # Get the neighbours of v, who are not neighbours of v0
        v_neighbours_not_v0 = [x for x in target_node_neighbours if x not in v0_neighbours]
        # If there are fewer than b-1 such neighbours, connect v_0 to all of them
        if len(v_neighbours_not_v0) < self.edge_budget-1:
            self.edge_budget = len(v_neighbours_not_v0) + 1
        # Make an ascending order list of the neighbours of v0, based on their degree
        sorted_neighbors = sorted(v_neighbours_not_v0, key=lambda x: graph.degree[x]) 
        # Connect v_0 to b-1 nodes of choice, who are neighbours of v but not of v_0
        for i in range(self.edge_budget-1):
            v0_neighbour = sorted_neighbors[i]
            # v0_neighbour = random.choice(v_neighbours_not_v0)   # Random choice
            graph.add_edge(v0, v0_neighbour)
            v_neighbours_not_v0.remove(v0_neighbour)
        
        new_community_structure = self.detection_alg.compute_community(graph)
        return graph, new_community_structure
