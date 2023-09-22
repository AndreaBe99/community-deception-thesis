# import sys
# sys.path.append("../../../")
from src.environment.graph_env import GraphEnvironment
from src.utils.utils import DetectionAlgorithmsNames, Utils
from src.community_algs.detection_algs import CommunityDetectionAlgorithm

import networkx as nx
from typing import List, Callable, Tuple
import random
import copy

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
        self.original_community_structure = copy.deepcopy(self.env.original_community_structure)
        self.possible_edges = self.get_possible_action()
        
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
    
    def hide_target_node_from_community(self) -> Tuple[nx.Graph, List[int], int]:
        """
        Hide the target node from the target community by rewiring its edges, 
        choosing the node with the highest degree between adding or removing an edge.
        
        Returns
        -------
        Tuple[nx.Graph, List[int], int]
            The new graph, the new community structure and the number of steps
        """
        graph = self.graph.copy()
        communities = self.original_community_structure
        done = False
        steps = self.steps
        # From the list possible_edges, create a list of tuples 
        # (node1, node2, degree_of_node2)
        possible_edges = []
        for edge in self.possible_edges:
                possible_edges.append(
                    (edge[0], edge[1], graph.degree(edge[1])))
        while steps > 0 and not done:
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
            new_community = self.get_new_community(communities)

            check = self.check_goal(new_community)
            if check == 1:
                # If the target community is a subset of the new community, the episode is finished
                done = True
            steps -= 1
        step = self.steps - steps
        return graph, communities, step
    
    def get_new_community(
                self,
                new_community_structure: List[List[int]]) -> List[int]:
        """
        Search the community target in the new community structure after 
        deception. As new community target after the action, we consider the 
        community that contains the target node, if this community satisfies 
        the deception constraint, the episode is finished, otherwise not.

        Parameters
        ----------
        node_target : int
            Target node to be hidden from the community
        new_community_structure : List[List[int]]
            New community structure after deception

        Returns
        -------
        List[int]
            New community target after deception
        """
        if new_community_structure is None:
            # The agent did not perform any rewiring, i.e. are the same communities
            return self.target_community
        for community in new_community_structure.communities:
            if self.target_node in community:
                return community
        raise ValueError("Community not found")

    def check_goal(self, new_community: int) -> int:
        """
        Check if the goal of hiding the target node was achieved

        Parameters
        ----------
        new_community : int
            New community of the target node

        Returns
        -------
        int
            1 if the goal was achieved, 0 otherwise
        """
        if len(new_community) == 1:
            return 1
        # Copy the communities to avoid modifying the original ones
        new_community_copy = new_community.copy()
        new_community_copy.remove(self.target_node)
        old_community_copy = self.target_community.copy()
        old_community_copy.remove(self.target_node)
        # Compute the similarity between the new and the old community
        similarity = self.env.community_similarity(
            new_community_copy,
            old_community_copy
        )
        del new_community_copy, old_community_copy
        if similarity <= self.env.tau:
            return 1
        return 0


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
