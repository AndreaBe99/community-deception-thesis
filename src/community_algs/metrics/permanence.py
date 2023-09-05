from typing import List
import networkx as nx
import numpy as np


class PermanenceCalculator:
    """Class to compute permanence of a node in a graph with respect to a
    community and a community structure.
    
    Returns a value between -1 and 1, where -1 is the worst permanence and 1
    is the best permanence. But for the community deception problem, we want
    to minimize the permanence, so the best permanence is -1.
    """
    def __init__(self, graph: nx.Graph, community_target: List[int], community_structure: List[List[int]]):
        self.graph = graph
        self.community_target = community_target
        self.community_structure = community_structure

    def internal_pull(self, v: int)->int:
        """
        Compute the internal pull of a node v in a graph G with respect to a 
        community C, denoted by the internal connections of a node $v$ within 
        its own community;

        Parameters
        ----------
        v : int
            Target node

        Returns
        -------
        internal_edges : int
            Internal pull of node v
        """
        internal_edges = 0
        for node in nx.neighbors(self.graph, v):
            if node in self.community_target:
                internal_edges += 1
        return internal_edges

    def max_external_pull(self, v: int)->int:
        """
        Compute the maximum connections to a single external community.

        Parameters
        ----------
        v : int
            Target node

        Returns
        -------
        max_external_edges : int
            Maximum external pull of node v
        """
        max_external_edges = 0
        for community in self.community_structure:
            if v in community:
                continue
            external_edges = 0
            for node in nx.neighbors(self.graph, v):
                if node in community:
                    external_edges += 1
            max_external_edges = max(max_external_edges, external_edges)
        return max_external_edges

    def internal_clustering_coefficient(self, v: int)->float:
        """
        Compute the internal clustering coefficient of a node v in a graph G
        denoted by the fraction of actual and possible number of edges among 
        the internal neighbors of v.

        Parameters
        ----------
        v : int
            Target node

        Returns
        -------
        float
            Internal clustering coefficient of node v
        """
        # Get subgraph of the community
        community_subgraph = self.graph.subgraph(self.community_target)
        
        # Delete node v from the subgraph
        subgraph_copy = community_subgraph.copy()
        subgraph_copy.remove_node(v)
        
        # Compute the number of actual edges, excluding the edges of node v
        n_actual_edges = subgraph_copy.number_of_edges()
        
        # Compute the number of possible edges, excluding the edges of node v
        n_nodes = subgraph_copy.number_of_nodes()
        num_possible_edges = (n_nodes * (n_nodes - 1)) / 2
        del subgraph_copy
        del community_subgraph
        return n_actual_edges / num_possible_edges


    def permanence(self, v: int)->float:
        """
        Permanence of a node v in a graph G with respect to a community C and
        a community structure, denoted by the fraction of the
        internal pull of v and the maximum external pull of v, minus the
        internal clustering coefficient of v.

        Parameters
        ----------
        v : iny
            Node to compute permanence

        Returns
        -------
        permanence_v : float
            Permanence of node v
        """
        I_v = self.internal_pull(v)
        E_max_v = self.max_external_pull(v)
        deg_v = len(list(self.graph.neighbors(v)))
        C_in_v = self.internal_clustering_coefficient(v)
        assert E_max_v > 0, "E_max_v must be greater than 0"
        assert deg_v > 0, "deg_v must be greater than 0"
        permanence_v = (I_v / E_max_v) * (1 / deg_v) - (1 - C_in_v)
        return permanence_v
    
    def normalized_permanence(self, v:int)->float:
        """
        Normalized permanence of a node v in a graph G with respect to a 
        community C and a community structure, denoted by the fraction of the
        internal pull of v and the maximum external pull of v, minus the
        internal clustering coefficient of v.

        Parameters
        ----------
        v : int
            Node to compute permanence

        Returns
        -------
        permanence_v : float
            Permanence of node v
        """
        # Get permanence, return a value between -1 and 1
        permanence_v = self.permanence(v)
        # Normalized permanence between 0 and 1, it is a value
        normalized_permanence_v = (permanence_v + 1) / 2
        return normalized_permanence_v



if __name__ == "__main__":
    graph = nx.karate_club_graph()
    # plot graph
    import matplotlib.pyplot as plt
    
    
    import sys
    sys.path.append("../../../")
    from src.community_algs.detection_algs import DetectionAlgorithm
    community_structure = DetectionAlgorithm("walktrap").compute_community(graph)
    community_target = community_structure[2]
    print(f"Community target: {community_target}")
    print(f"Community structure: {community_structure}")
    
    
    permanence_calculator = PermanenceCalculator(graph, community_target, community_structure)
    node_to_compute = community_target[3]
    permanence_value = permanence_calculator.permanence(node_to_compute)
    normalized_permanence_value = permanence_calculator.normalized_permanence(node_to_compute)
    
    print(f"Permanence of node {node_to_compute}: {permanence_value}")
    print(f"Normalized permanence of node {node_to_compute}: {normalized_permanence_value}")
    nx.draw(graph, with_labels=True)
    plt.show()
