import networkx as nx
import numpy as np
from typing import List, Set, Tuple

class Safeness:
    """Computes the safeness of a node in a community and the safeness of a community."""
    def __init__(self, graph: nx.Graph, community_target: List[int]):
        self.graph = graph
        self.community_target = community_target
        
        # self.node_target = node_target
        # Compute the number of nodes in a community C that are in the same connected component of u
        self.V_u_C = self.num_nodes_in_same_component()
        # Get the number of intra-community edges for u.
        self.E_u_C = self.get_intra_comminty_edges()
        # Get the number of inter-community edges for u.
        self.E_u_C_bar = self.get_inter_community_edges()
        
        
        # Dict of add ratio for each node in the community {node: add_ratio}
        self.node_minimum_add_ratio = self.get_node_minimum_add_ratio(self.community_target)
        # Dictionary of external nodes for each node in the community {node: first external_node founded}
        self.external_node_dict = self.find_external_node_dict(self.community_target)
        # Dict of (edge, eps_del) for each edge in the community {edge: eps_del}
        self.best_del_excl_bridges = self.get_best_del_excl_bridges(self.community_target)
        
    def community_hiding(self, community_target, edge_budget: int) -> nx.Graph:
        """
        Hide the target community using the safeness metric.

        Parameters
        ----------
        community_target : _type_
            Community to hide.
        edge_budget : int
            Budget of edges to use.

        Returns
        -------
        nx.Graph
            Graph with the target community hidden.
        """
        initial_budget = edge_budget
        while True:
            # n_p = self.get_node_minimum_add_ratio(community_target)

            n_p = min(self.node_minimum_add_ratio, key=self.node_minimum_add_ratio.get)
            
            # n_t = self.find_external_node(n_p, community_target)
            n_t = max(self.external_node_dict, key=self.external_node_dict.get)
            
            eps_add = self.get_addition_gain((n_p, n_t), community_target)
            
            # n_k, n_l = self.get_best_del_excl_bridges(community_target)
            if len(self.best_del_excl_bridges) < 1:
                n_k, n_l = (None, None)
            else:
                # n_k, n_l = max(self.best_del_excl_bridges,key=lambda x: x[1])[0]
                # Get the edge with the maximum eps_del
                n_k, n_l = max(self.best_del_excl_bridges, key=self.best_del_excl_bridges.get)
            
            
            if n_k == None and n_l == None:
                eps_del = -1
            else:
                eps_del = self.get_deletion_gain((n_k, n_l), community_target)
            
            
            if eps_add >= eps_del:
                self.graph.add_edge(n_p, n_t)
                
                # Update the node_minimum_add_ratio
                min_add_ratio = 0
                self.node_minimum_add_ratio = self.get_node_minimum_add_ratio(self.community_target)
                self.external_node_dict = self.find_external_node_dict(self.community_target)
                
            elif eps_del > 0:
                self.graph.remove_edge(n_k, n_l)
                # Update the best_del_excl_bridges
                self.best_del_excl_bridges = self.get_best_del_excl_bridges(self.community_target)
                
            edge_budget -= 1
            
            if edge_budget <= 0 or (eps_add <= 0 and eps_del <= 0):
                break
        # print("Initial budget: {}, final budget: {}".format(initial_budget, edge_budget))
        steps = initial_budget - edge_budget
        return self.graph, steps
    
    def get_node_minimum_add_ratio(self, community_target: List[int])->int:
        """
        Computes for each node n inside the target community, the fraction of
        n’s edges that point outside C.

        Parameters
        ----------
        community_target : List[int]
            Target community.

        Returns
        -------
        min_add_ratio : int
            Node with the minimum add ratio.
        """
        # List of Tuple of (node, min_add_ratio)
        # node_min_add_ratio = dict()
        # for n in community_target:
        #     min_add_ratio = 0
        #     for neighbor in self.graph.neighbors(n):
        #         if neighbor not in community_target:
        #             min_add_ratio += 1
        #     if self.graph.degree(n) > 0:
        #         min_add_ratio = min_add_ratio / self.graph.degree(n)
        #     else:
        #         min_add_ratio = 0
        #     node_min_add_ratio[n] = min_add_ratio
        # return node_min_add_ratio
        # Get the node with the minimum add ratio
        # min_add_ratio = min(node_min_add_ratio, key=lambda x: x[1])
        # return min_add_ratio[0]

        node_min_add_ratio = {}
        for node in community_target:
            neighbors = set(self.graph.neighbors(node))
            min_add_ratio = sum(
                1 if neighbor not in community_target else 0 for neighbor in neighbors)
            degree = self.graph.degree(node)
            min_add_ratio /= degree if degree > 0 else 1
            node_min_add_ratio[node] = min_add_ratio
        return node_min_add_ratio
        

    def find_external_node_dict(self, community_target: List[int]) -> dict:
        """
        Find a node n_t not in community_target, such that the edge (n_p, n_t)
        does not exist, and that maximize the addition gain.
        
        Parameters
        ----------
        n_p : int
            Node p.
        community_target : List[int]
            Target community.

        Returns
        -------
        external_node_dict : dict
            Dict of (node, external_node) for each node in the community
            with external_node that maximize the addition gain.
        """
        external_node_dict = dict()
        for n_p in community_target:
            # get neighbors of n_p
            n_t_external = self.find_external_node(n_p, community_target)
            # For each node n_t, in the list of external nodes, compute the 
            # addition gain, and save in the dict the node with the maximum
            # addition gain.
            if len(n_t_external) > 0:
                max_addition_gain = -1
                for n_t in n_t_external:
                    addition_gain = self.get_addition_gain(
                        (n_p, n_t), community_target)
                    if addition_gain > max_addition_gain:
                        max_addition_gain = addition_gain
                        external_node_dict[n_p] = n_t
                
        return external_node_dict
    
    def find_external_node(self, n_p: int, community_target: List[int]) -> list:
        """
        Find a node n_t not in community_target, such that the edge (n_p, n_t)
        does not exist.
        
        Parameters
        ----------
        n_p : int
            Node p.
        community_target : List[int]
            Target community.

        Returns
        -------
        n_t_external : list
            List of nodes that are not in community_target and that are not
            neighbors of n_p.
        """
        # get neighbors of n_p
        neighbors_p = self.graph.neighbors(n_p)
        n_t_external = list()
        for n_t in self.graph.nodes():
            if n_t not in community_target and n_t not in neighbors_p:
                n_t_external.append(n_t)
        return n_t_external
    
    def get_best_del_excl_bridges(self, community_target: List[int]) -> dict:
        """
        It works in two phases:
            1. It excludes bridge edges that, if deleted, could disconnect 
                target community.
            2. For each remeining edge, it computes the value specified in 
                theorem 8.

        Parameters
        ----------
        community_target : 
            Community target.

        Returns
        -------
        dict
            Dict of (edge, eps_del) for each edge in the community {edge: eps_del}
        """
        graph = self.graph.copy()
        # Delete all bridge edges that, if deleted, could disconnect target community.
        for node in community_target:
            for neighbor in self.graph.neighbors(node):
                if neighbor not in community_target and self.is_bridge((node, neighbor)):
                    graph.remove_edge(node, neighbor)
        # List of Tuple of (edge, eps_del)
        esp_del_list = dict()
        for node in community_target:
            # Get the neighbors of the node
            for neighbor in self.graph.neighbors(node):
                if neighbor not in community_target:
                    esp_del_list[(node, neighbor)] = self.get_deletion_gain(
                        (node, neighbor), community_target)
        return esp_del_list
    
    @DeprecationWarning
    def get_best_del_excl_bridges_tuple(self, community_target: List[int]) -> Tuple[int, int]:
        """
        It works in two phases:
            1. It excludes bridge edges that, if deleted, could disconnect 
                target community.
            2. For each remeining edge, it computes the value specified in 
                theorem 8.

        Parameters
        ----------
        community_target : 
            Community target.

        Returns
        -------
        Tuple[int, int]
            Edge with the maximum eps_del.
        """
        graph = self.graph.copy()

        # Delete all bridge edges that, if deleted, could disconnect target community.
        for node in community_target:
            for neighbor in self.graph.neighbors(node):
                if neighbor not in community_target and self.is_bridge((node, neighbor)):
                    graph.remove_edge(node, neighbor)

        # List of Tuple of (edge, eps_del)
        esp_del_list = dict()
        for node in community_target:
            # Get the neighbors of the node
            for neighbor in self.graph.neighbors(node):
                if neighbor not in community_target:
                    esp_del_list[(node, neighbor)] = self.get_deletion_gain(
                        (node, neighbor), community_target)

        # Get the edge with the maximum eps_del
        if len(esp_del_list) < 1:
            return None, None
        n_k, n_l = max(self.best_del_excl_bridges, key=self.best_del_excl_bridges.get)
        return n_k, n_l 
    
    def is_bridge(self, edge: Tuple[int, int]) -> bool:
        """
        Check if the edge (node, neighbor) is a bridge, i.e. if we remove it 
        the graph will be disconnected.

        Parameters
        ----------
        edge : Tuple[int, int]
            Edge to check.

        Returns
        -------
        bool
            True if the edge is a bridge, False otherwise.
        """
        graph = self.graph.copy()
        graph.remove_edge(*edge)
        return not nx.is_connected(graph)
    
    def get_addition_gain(self, edge: Tuple[int, int], community_target: List[int])->float:
        """
        Computes the addition gain of adding an edge.

        Parameters
        ----------
        edge : Tuple[int, int]
            Edge to add.
        community_target : List[int]
            Community target.

        Returns
        -------
        float
            Addition gain.
        """
        graph = self.graph.copy()
        # Compute the safeness before and after adding the edge.
        safeness_before = self.compute_community_safeness(
            graph, community_target)
        graph.add_edge(*edge)
        safeness_after = self.compute_community_safeness(
            graph, community_target)
        return safeness_after - safeness_before
    
    def get_deletion_gain(self, edge: Tuple[int, int], community_target: List[int])->float:
        """
        Computes the deletion gain of deleting an edge.

        Parameters
        ----------
        edge : Tuple[int, int]
            Edge to delete.
        community_target : List[int]
            Community target.

        Returns
        -------
        float
            Delete gain.
        """
        graph = self.graph.copy()
        # Compute the safeness before and after adding the edge.
        safeness_before = self.compute_community_safeness(
            graph, community_target)
        graph.remove_edge(*edge)
        safeness_after = self.compute_community_safeness(
            graph, community_target)
        return safeness_after - safeness_before
    
    
    def compute_community_safeness(self, graph, community_target: List[int]) -> float:
        """
        Computes the community safeness of the community.
        
        Parameters
        ----------
        community_taget: List[int]
            The community that we want to compute the safeness.

        Returns
        -------
        float
            The community safeness.
        """
        community_safeness = 0
        for node in community_target:
            community_safeness += self.compute_node_safeness(
                graph, community_target, node)
        return community_safeness / len(community_target)
    
    def compute_node_safeness(
        self, 
        graph: nx.Graph, 
        community_target: List[int], 
        node: int) -> float:
        """
        Computes the node safeness of the node in the community.
        
        Parameters
        ----------
        graph: nx.Graph
            The graph.
        community_target: List[int]
            The community of the node that we want to compute the safeness.
        
        node: int
            The node that we want to compute the safeness.

        Returns
        -------
        sigma_u_C: float
            The node safeness.
        """
        # Get the degree of u.
        deg_u = graph.degree(node)

        # Compute the node safeness.
        # assert len(community_target) > 1, "The community must have at least 2 nodes."
        # assert deg_u > 0, "The node must have at least 1 edge."
        
        if len(community_target) <= 1:
            argument_1 = 0
        else:
            argument_1 = ((self.V_u_C[node] - 1) / (len(community_target) - 1))
            
        if deg_u < 1:
            argument_2 = 0
        else:
            argument_2 = len(self.E_u_C) / deg_u
        
        sigma_u_C = 0.5*argument_1 + 0.5*argument_2
        return sigma_u_C
    
    # TEST, check if it si correct
    def num_nodes_in_same_component(self):
        """
        Computes the number of nodes in a community C that are in the same 
        connected component of u.

        Returns
        -------
        _type_
            _description_
        """
        V_u_C = dict()
        # Create a subgraph induced by the nodes in community C
        subgraph = self.graph.subgraph(self.community_target)

        # Compute the connected components of the subgraph
        components = list(nx.connected_components(subgraph))
        # Find the component that contains node u
        for u in self.community_target:
            V_u_C[u] = 0
            for component in components:
                if u in component:
                    # Return the number of nodes in the component
                    V_u_C[u] = len(component)
        return V_u_C

    def get_intra_comminty_edges(self) -> List[int]:
        """
        Get the intra-community edges of the community.

        Returns
        -------
        intra_community_edges: Set[int]
            The intra-community edges of the community.
        """
        intra_community_edges = list()
        for u in self.community_target:
            for v in self.community_target:
                if u != v and self.graph.has_edge(u, v) and (v, u) not in intra_community_edges:
                    intra_community_edges.append((u, v))
        return intra_community_edges

    def get_inter_community_edges(self) -> List[int]:
        """
        Get the inter-community edges of the community.

        Returns
        -------
        inter_community_edges: Set[int]
            The inter-community edges of the community.
        """
        inter_community_edges = list()
        for u in self.community_target:
            for v in self.graph.neighbors(u):
                if v not in self.community_target:
                    inter_community_edges.append((u, v))
        return inter_community_edges

if __name__ == "__main__":
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

    community = [0, 1, 2]
    node = 0

    node_safeness = Safeness(graph)

    print(node_safeness.compute_node_safeness(community, node))
    
    print(node_safeness.compute_community_safeness(community))
