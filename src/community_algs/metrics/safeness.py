import networkx as nx
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
            n_p = self.get_node_minimum_add_ratio(community_target)
            n_t = self.find_external_node(n_p, community_target)
            eps_add = self.get_addition_gain((n_p, n_t), community_target)
            
            n_k, n_l = self.get_best_del_excl_bridges(community_target)
            if n_k == None and n_l == None:
                eps_del = -1
            else:
                eps_del = self.get_deletion_gain((n_k, n_l), community_target)
            
            if eps_add >= eps_del:
                self.graph.add_edge(n_p, n_t)
            elif eps_del > 0:
                self.graph.remove_edge(n_k, n_l)
            
            edge_budget -= 1
            
            if edge_budget <= 0 or (eps_add <= 0 and eps_del <= 0):
                break
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
        node_min_add_ratio = list()
        for n in community_target:
            min_add_ratio = 0
            for neighbor in self.graph.neighbors(n):
                if neighbor not in community_target:
                    min_add_ratio += 1
            min_add_ratio = min_add_ratio / self.graph.degree(n)
            node_min_add_ratio.append((n, min_add_ratio))
        
        # Get the node with the minimum add ratio
        min_add_ratio = min(node_min_add_ratio, key=lambda x: x[1])
        return min_add_ratio[0]
    
    def find_external_node(self, n_p: int, community_target: List[int]) -> int:
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
        int
            Destination node.
        """
        # get neighbors of n_p
        neighbors_p = self.graph.neighbors(n_p)
        for n_t in self.graph.nodes():
            if n_t not in community_target and n_t not in neighbors_p:
                return n_t
    
    def get_best_del_excl_bridges(self, community_target: List[int]) -> Tuple[int, int]:
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
            # Get the neighbors of the node
            neighbors = self.graph.neighbors(node)
            for neighbor in neighbors:
                if neighbor not in community_target:
                    # Check if the edge is a bridge
                    if self.is_bridge((node, neighbor)):
                        graph.remove_edge(node, neighbor)

        # List of Tuple of (edge, eps_del)
        esp_del_list = list()
        for node in community_target:
            # Get the neighbors of the node
            neighbors = self.graph.neighbors(node)
            for neighbor in neighbors:
                if neighbor not in community_target:
                    eps_del = self.get_deletion_gain((node, neighbor), community_target)
                    esp_del_list.append(((node, neighbor), eps_del))
        
        # Get the edge with the maximum eps_del
        if len(esp_del_list) < 1:
            return (None, None)
        max_eps_del = max(esp_del_list, key=lambda x: x[1])
        return max_eps_del[0]
    
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
        assert len(community_target) > 1, "The community must have at least 2 nodes."
        assert deg_u > 0, "The node must have at least 1 edge."
        sigma_u_C = 0.5*((self.V_u_C[node] - len(self.E_u_C)) /
                         (len(community_target) - 1)) + 0.5*(len(self.E_u_C_bar) / deg_u)
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
