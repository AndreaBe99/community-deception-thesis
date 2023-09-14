# import sys
# sys.path.append("../../../")
from src.environment.graph_env import GraphEnvironment
from src.utils.utils import DetectionAlgorithmsNames, Utils
from src.community_algs.detection_algs import CommunityDetectionAlgorithm

import networkx as nx
from cdlib import algorithms
from typing import List
import random

class RandomHiding():
    
    def __init__(
        self, 
        env: GraphEnvironment, 
        steps: int, 
        target_community: List[int]):
        self.env = env
        self.graph = self.env.original_graph
        self.steps = steps
        self.target_node = self.env.node_target
        self.target_community = target_community
        self.detection_alg = self.env.detection
        self.original_community_structure = self.env.original_community_structure
        self.possible_edges = self.env.possible_actions
        # Put all the edges in a list
        self.possible_edges = list(self.possible_edges["ADD"]) + list(self.possible_edges["REMOVE"])
        

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
            new_community = Utils.get_new_community(
                self.target_node, communities)

            check = Utils.check_goal(
                self.env, self.target_node, self.target_community, new_community)
            if check == 1:
                # If the target community is a subset of the new community, the episode is finished
                done = True
            self.steps -= 1
            
            self.steps -= 1
        return graph, communities
        


# Example usage:
if __name__ == "__main__":
    # Import karate club graph
    graph = nx.karate_club_graph()

    detection_alg_name = DetectionAlgorithmsNames.INF.value
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
    