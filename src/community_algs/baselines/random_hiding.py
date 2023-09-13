import sys
sys.path.append("../../../")
from src.utils.utils import DetectionAlgorithms
from src.community_algs.detection_algs import CommunityDetectionAlgorithm

import networkx as nx
from cdlib import algorithms
from typing import List
import random

class RandomHiding():
    
    def __init__(
        self, 
        graph: nx.Graph, 
        steps: int, 
        target_node: int, 
        target_community: List[int],
        detection_alg: str = DetectionAlgorithms.INF.value):
        self.graph = graph
        self.steps = steps
        self.target_node = target_node
        self.target_community = target_community
        self.detection_alg = CommunityDetectionAlgorithm(detection_alg)
        self.original_community = self.detection_alg.compute_community(graph)
        self.possible_edges = self.get_possible_edges()

    def hide_target_node_from_community(self)->tuple:
        """
        Hide the target node from the target community by rewiring its edges, 
        choosing randomly between adding or removing an edge.
        
        Returns
        -------
        G_prime: nx.Graph
        """
        graph = self.graph.copy()
        done = False
        while self.steps > 0 and not done:
            # Random choose a edge from the possible edges
            edge = self.possible_edges.pop()
            if graph.has_edge(*edge):
                # Remove the edge
                graph.remove_edge(*edge)
            else:
                # Add the edge
                graph.add_edge(*edge)
            
            # Compute the new community structure
            communities = self.detection_alg.compute_community(graph)
            idx = self.get_community_target_idx(communities.communities, self.target_community)
            
            if self.target_node in communities.communities[idx]:
                if set(communities.communities[idx]).issuperset(set(self.target_community)):
                    # If the target community is a subset of the new community, the episode is finished
                    done = True
            if self.target_node not in communities.communities[idx]:
                # If the target node is not in the target community, the episode is finished
                done = True
            
            self.steps -= 1
        return graph, communities
        
    def get_possible_edges(self)->set:
        """
        Returns all the possible actions that can be applied to the graph
        given a source node(self.node_target). The possible actions are:
            - Add an edge between the source node and a node outside the community
            - Remove an edge between the source node and a node inside the community

        Returns
        -------
        possible_edges: set
            Set of possible edges that can be added or removed from the graph 
        """
        possible_edges = set()
        
        def in_community(node):
            return node in self.target_community
        def out_community(node):
            return node not in self.target_community
        
        u = self.target_node
        for v in self.graph.nodes():
            if v == u:
                continue
            if in_community(u) and in_community(v):
                if self.graph.has_edge(u, v):
                    possible_edges.add((u, v))
            elif (in_community(u) and out_community(v)) \
                or (out_community(u) and in_community(v)):
                if not self.graph.has_edge(u, v):
                    possible_edges.add((u, v))
        return possible_edges


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


# Example usage:
if __name__ == "__main__":
    # Import karate club graph
    graph = nx.karate_club_graph()

    detection_alg_name = DetectionAlgorithms.INF.value
    detection_alg = CommunityDetectionAlgorithm(detection_alg_name)
    communities = detection_alg.compute_community(graph)

    # Choose randomly a community
    community = communities.communities[random.randint(0, len(communities.communities)-1)]
    # Choose randomly a node from the community
    node = community[random.randint(0, len(community)-1)]
    
    edge_budget = graph.number_of_edges()*0.1
    
    random_hiding = RandomHiding(graph, edge_budget, node, community, detection_alg)
    
    new_graph = random_hiding.hide_target_node_from_community()
    
    # Compute the new community structure
    new_communities = detection_alg.compute_community(new_graph)
    
    print("Original community: ", communities.communities)
    print("New community: ", new_communities.communities)
    