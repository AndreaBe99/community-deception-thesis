# import sys
# sys.path.append("../../../")
from src.environment.graph_env import GraphEnvironment
from src.utils.utils import DetectionAlgorithmsNames, Utils
from src.community_algs.detection_algs import CommunityDetectionAlgorithm

import networkx as nx
from typing import List
import random


class DegreeHiding():

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
        self.possible_edges = self.get_possible_action()  # self.env.possible_actions
        # Put all the edges in a list
        # self.possible_edges = self.env.possible_actions
        # self.possible_edges = list(self.possible_edges["ADD"]) + list(self.possible_edges["REMOVE"])

    def get_possible_action(self):
        # Put all edge between the target node and its neighbors in a list
        possible_actions_add = []
        for neighbor in self.graph.neighbors(self.target_node):
            possible_actions_add.append((self.target_node, neighbor))

        # Put all the edges that aren't neighbors of the target node in a list
        possible_actions_remove = []
        for node in self.graph.nodes():
            if node != self.target_node and node not in self.graph.neighbors(self.target_node):
                possible_actions_remove.append((self.target_node, node))
        possible_action = possible_actions_add + possible_actions_remove
        return possible_action
    
    def hide_target_node_from_community(self) -> tuple:
        """
        Hide the target node from the target community by rewiring its edges, 
        choosing the node with the highest degree between adding or removing an edge.
        
        Returns
        -------
        G_prime: nx.Graph
        """
        graph = self.graph.copy()
        done = False
        # From the list possible_edges, create a list of tuples 
        # (node1, node2, degree_of_node2)
        possible_edges = []
        for edge in self.possible_edges:
                possible_edges.append(
                    (edge[0], edge[1], graph.degree(edge[1])))
        while self.steps > 0 and not done:
            # Choose the edge with the highest degree
            max_tuple = max(possible_edges, key=lambda x: x[2])
            possible_edges.remove(max_tuple)
            edge = (max_tuple[0], max_tuple[1])
            
            if graph.has_edge(*edge):
                # Remove the edge
                graph.remove_edge(*edge)
            else:
                # Add the edge
                graph.add_edge(*edge)

            # Compute the new community structure
            communities = self.detection_alg.compute_community(graph)
            new_community = Utils.get_new_community(self.target_node, communities)

            check = Utils.check_goal(self.env, self.target_node, self.target_community, new_community)
            if check == 1:
                # If the target community is a subset of the new community, the episode is finished
                done = True
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
    community = communities.communities[random.randint(
        0, len(communities.communities)-1)]
    # Choose randomly a node from the community
    node = community[random.randint(0, len(community)-1)]

    edge_budget = graph.number_of_edges()*0.1

    random_hiding = DegreeHiding(
        graph, edge_budget, node, community, detection_alg)

    new_graph = random_hiding.hide_target_node_from_community()

    # Compute the new community structure
    new_communities = detection_alg.compute_community(new_graph)

    print("Original community: ", communities.communities)
    print("New community: ", new_communities.communities)
