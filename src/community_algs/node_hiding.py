import sys
sys.path.append("../../")
from src.community_algs.detection_algs import CommunityDetectionAlgorithm
from src.community_algs.metrics.safeness import Safeness
from src.utils.utils import Utils, FilePaths, DetectionAlgorithms, HyperParams

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random


class DisguisingIndividuals():
    """Given a network and a source node v,our objective is to conceal the 
    importance of v by decreasing its centrality without compromising its
    influence over the network.
    
    From the article "Hiding Individuals and Communities in a Social Network".
    """
    def __init__(self, graph: nx.Graph, target_node: int) -> None:
        self.graph = graph
        self.target_node = target_node
    
    @staticmethod
    def get_edge_budget(graph: nx.Graph, budget: float) -> int:
        """
        Compute the number of edges to add given a budget and a graph.

        Parameters
        ----------
        graph : nx.Graph
            Graph to add edges to.
        budget : int
            Budget of the attack, value between 0 and 100.

        Returns
        -------
        int
            Number of edges to add.
        """
        assert budget > 0 and budget <= 100, "Budget must be between 0 and 100"
        return int(budget * graph.number_of_edges() / 100)
    
    def roam_heuristic(self, budget: int) -> nx.Graph:
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
        edge_budget = self.get_edge_budget(self.graph, budget)
        
        # ° --- Step 1 --- ° #
        target_node_neighbours = list(self.graph.neighbors(self.target_node))
        
        # Choose v0 as the neighbour of target_node with the most connections
        v0 = target_node_neighbours[0]
        for v in target_node_neighbours:
            if self.graph.degree[v] > self.graph.degree[v0]:
                v0 = v
        # v0 = random.choice(target_node_neighbours)    # Random choice
        # Remove the edge between v and v0
        self.graph.remove_edge(self.target_node, v0)
        
        # ° --- Step 2 --- ° #
        # Get the neighbours of v0
        v0_neighbours = list(self.graph.neighbors(v0))
        # Get the neighbours of v, who are not neighbours of v0
        v_neighbours_not_v0 = [x for x in target_node_neighbours if x not in v0_neighbours]
        # If there are fewer than b-1 such neighbours, connect v_0 to all of them
        if len(v_neighbours_not_v0) < edge_budget-1:
            edge_budget = len(v_neighbours_not_v0) + 1
        # Make an ascending order list of the neighbours of v0, based on their degree
        sorted_neighbors = sorted(v_neighbours_not_v0, key=lambda x: self.graph.degree[x]) 
        # Connect v_0 to b-1 nodes of choice, who are neighbours of v but not of v_0
        for i in range(edge_budget-1):
            v0_neighbour = sorted_neighbors[i]
            # v0_neighbour = random.choice(v_neighbours_not_v0)   # Random choice
            self.graph.add_edge(v0, v0_neighbour)
            v_neighbours_not_v0.remove(v0_neighbour)
        
        return self.graph



def get_community_target(community_structure, community_target):
    max_count = 0
    max_list_idx = 0
    for i, lst in enumerate(community_structure):
        count = sum(1 for x in lst if x in community_target)
        if count > max_count:
            max_count = count
            max_list_idx = i
    return max_list_idx
    
if __name__ == "__main__":
    print("*"*20, "Setup Information", "*"*20)

    # ° ------ Graph Setup ------ ° #
    # ! REAL GRAPH Graph path (change the following line to change the graph)
    graph_path = "../../"+FilePaths.WORDS.value
    # Load the graph from the dataset folder
    graph = Utils.import_mtx_graph(graph_path)
    # ! SYNTHETIC GRAPH Graph path (change the following line to change the graph)
    # graph, graph_path = Utils.generate_lfr_benchmark_graph()
    # Set the environment name as the graph name
    env_name = graph_path.split("/")[-1].split(".")[0]
    # Print the number of nodes and edges
    print("* Graph Name:", env_name)
    print("*", graph)
    
    # ° ------ Community Setup ------ ° #
    detection_alg = DetectionAlgorithms.WALK.value
    print("* Community Detection Algorithm:", detection_alg)
    # Apply the community detection algorithm on the graph
    dct = CommunityDetectionAlgorithm(detection_alg)
    community_structure = dct.compute_community(graph)
    print("* Number of communities found:",
        len(community_structure.communities))
    # Choose one of the communities found by the algorithm, for now we choose
    # the community with the highest number of nodes
    community_target = max(community_structure.communities, key=len)
    idx_community = community_structure.communities.index(community_target)
    print("* Initial Community Target:", community_target)
    print("* Initial Index of the Community Target:", idx_community)
    # TEST: Choose a node to remove from the community
    node_target = community_target[random.randint(0, len(community_target)-1)]
    print("* Initial Nodes Target:", node_target)
    
    # Compute the number of edges to remove
    beta = HyperParams.BETA.value
    assert beta > 0 and beta <= 100, "Budget must be between 0 and 100"
    edge_budget = int(beta * graph.number_of_edges() / 100)
    
    # Safe graph for a comparison
    graph_before = graph.copy()
    
    # Hide and Seek Graph
    hs_graph = graph.copy()
    # Safeness Graph
    sf_graph = graph.copy()
    
    # Apply Hide and Seek
    deception = DisguisingIndividuals(hs_graph, node_target)
    hs_graph = deception.roam_heuristic(edge_budget)
    
    # Node Hiding with safeness
    safeness = Safeness(sf_graph, community_target, node_target)
    sf_graph = safeness.node_hiding(edge_budget)
    
    new_cs_hs = dct.compute_community(hs_graph)
    nee_cs_sf = dct.compute_community(sf_graph)
    print("* Communities Before:\n", community_structure.communities)
    print("* Communities After Hide and Seek:\n", new_cs_hs.communities)
    print("* Communities After Safeness:\n", nee_cs_sf.communities)
    
    # Index of the community target after deception
    idx_ct_hs = get_community_target(new_cs_hs.communities, community_target)
    print("* Community Target After Hide and Seek:\n", new_cs_hs.communities[idx_ct_hs])
    idx_ct_sf = get_community_target(nee_cs_sf.communities, community_target)
    print("* Community Target After Safeness:\n", nee_cs_sf.communities[idx_ct_sf])

    
    if node_target in new_cs_hs.communities[idx_ct_hs]:
        print("\n* Node Target Found in the Community Target after Hide and Seek\n")
    if node_target in nee_cs_sf.communities[idx_ct_sf]:
        print("\n* Node Target Found in the Community Target after Safeness\n")
    
    print("* Target Node Centrality Before:\t\t", 
        nx.degree_centrality(graph_before)[node_target])
    print("* Target Node Centrality After Hide and Seek:\t",
        nx.degree_centrality(hs_graph)[node_target])
    print("* Target Node Centrality After Safeness:\t",
        nx.degree_centrality(sf_graph)[node_target])
    
    
    print("* NMI After Hide and Seek:\t", new_cs_hs.normalized_mutual_information(community_structure).score)
    print("* NMI After Safeness:\t\t", nee_cs_sf.normalized_mutual_information(community_structure).score)
    
    # Plot the graph
    # nx.draw(graph_before, with_labels=True)
    # nx.draw(graph, with_labels=True)
    # plt.show()
