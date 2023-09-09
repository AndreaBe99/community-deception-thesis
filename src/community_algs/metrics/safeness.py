import networkx as nx
from typing import List, Set, Tuple

class Safeness:
    """Computes the safeness of a node in a community and the safeness of a community."""
    def __init__(self, graph: nx.Graph, community_target: List[int], node_target: int):
        self.graph = graph
        self.community_target = community_target
        self.node_target = node_target
        # Compute the number of nodes in a community C that are in the same connected component of u
        self.V_u_C = self.num_nodes_in_same_component()
        # Get the number of intra-community edges for u.
        self.E_u_C = self.get_intra_comminty_edges()
        # Get the number of inter-community edges for u.
        self.E_u_C_bar = self.get_inter_community_edges()
    
    def node_hiding(self, edge_budget: int) -> nx.Graph:
        """
        Hide the target node from the community.
        """
        print("*"*20, "Node Hiding with Safeness", "*"*20)
        print("* Edge Budget:", edge_budget)
        print("* Node Target:", self.node_target)
        print("* Community Target:", self.community_target)
        print("* Community Nodes Reachability:", self.V_u_C)
        print("* Community Intra-Edges:", self.E_u_C)
        print("* Community Inter-Edges:", self.E_u_C_bar)
        
        print("*"*20, "     Start Rewiring     ", "*"*20)
        while True:
            destination_nodes = self.find_external_node()
            # Get the node from destination_nodes that maximizes the addition gain
            # add_gain = max(self.get_addition_gain((self.node_target, node)) for node in destination_nodes)
            add_max_node = max(destination_nodes, key=lambda x: self.get_addition_gain((self.node_target, x)))
            add_edge = (self.node_target, add_max_node)
            add_gain = self.get_addition_gain(add_edge)

            
            del_edge, del_gain = self.get_best_del_excl_bridges()
            # del_gain = self.get_deletion_gain(del_edge)

            if add_gain >= del_gain:
                self.graph.add_edge(*add_edge)
                print(f"* Add edge: ({add_edge})")
            elif del_gain > 0:
                self.graph.remove_edge(*del_edge)
                print(f"* Remove edge: {del_edge}")
            else:
                print("* No more edges to add or remove")
            if edge_budget <= 0 or (add_gain <= 0 and del_gain <= 0):
                break
        print("*"*20, "      End Rewiring      ", "*"*20)
        return self.graph
    
    def get_node_minimum_add_ratio(self) -> float:
        """
        Computes, for a node n inside the target community, the fraction of 
        n’s edges that point outside C
        
        Returns
        -------
        int:
            Node with the minimum add ratio.
        """
        min_add_ratio = dict()
        n_neighbors = self.graph.neighbors(self.node_target)
        for neighbor in n_neighbors:
            min_add_ratio.append[neighbor] = 0
            if neighbor in self.community_target:
                min_add_ratio.append[neighbor] += 1
        for neighbor in min_add_ratio:
            min_add_ratio[neighbor] = min_add_ratio[neighbor] / len(n_neighbors)
        # Return the neighbor with the minimum add ratio
        return min(min_add_ratio, key=lambda x: x[1])

    def find_external_node(self) -> int:
        """
        Find a list of external nodes, such that the node is not connected to
        the target node.
        
        Returns
        -------
        int
            Destination node.
        """
        external_nodes = list()
        for node in self.graph.nodes():
            if node not in self.community_target \
                and (not self.graph.has_edge(self.node_target, node) or \
                    not self.graph.has_edge(node, self.node_target)):
                external_nodes.append(node)
        # List of external nodes, such that the node is not connected to the
        # target node.
        return external_nodes
        
    
    def get_addition_gain(self, edge: Tuple[int]) -> float:
        """
        Computes the addition gain of adding an edge.

        Parameters
        ----------
        edge : Tuple[int]
            Edge to add.

        Returns
        -------
        float
            Addition gain.
        """
        graph = self.graph.copy()
        # Compute the safeness before and after adding the edge.
        safeness_before = self.compute_node_safeness(
            graph, self.community_target, self.node_target)
        graph.add_edge(*edge)
        safeness_after = self.compute_node_safeness(
            graph, self.community_target, self.node_target)
        return safeness_after - safeness_before

    def get_deletion_gain(self, edge: Tuple[int]) -> float:
        """
        Computes the deletion gain of deleting an edge.

        Parameters
        ----------
        edge : Tuple[int]
            Edge to delete.

        Returns
        -------
        float
            Deletion gain.
        """
        graph = self.graph.copy()
        # Compute the safeness before and after removing the edge.
        safeness_before = self.compute_node_safeness(
            graph, self.community_target, self.node_target)
        graph.remove_edge(*edge)
        safeness_after = self.compute_node_safeness(
            graph, self.community_target, self.node_target)
        return safeness_after - safeness_before
    
    def get_best_del_excl_bridges(self) -> Tuple[int]:
        """ 
        It works in two phases; it excludes bridge edges that, if deleted, 
        could disconnect C; then, for each remaining edge, it computes the 
        value specified in Theorem 8.
        
        Returns
        -------
        Tuple[int]
            The edge to delete.
        """
        graph = self.graph.copy()
        max_gain = dict()
        for edge in self.graph.neighbors(self.node_target):
            edge = (self.node_target, edge)
            # Compute the safeness before and after removing the edge.
            safeness_before = self.compute_node_safeness(
                graph, self.community_target, self.node_target)
            
            if graph.has_edge(*edge):
                graph.remove_edge(*edge)
            else:
                graph.remove_edge(*edge[::-1])
            # Compute subgraph of the community
            subgraph = graph.subgraph(self.community_target)
            # Check if the subgraph is connected
            if not nx.is_connected(subgraph):
                graph.add_edge(*edge)
            else:
                safeness_after = self.compute_node_safeness(
                    graph, self.community_target, self.node_target)
                max_gain[edge] = safeness_after - safeness_before
        
        if len(max_gain) > 0:
            # get max value and its key
            max_edge = max(max_gain, key=lambda x: x[1])
            max_gain = max_gain[max_edge]
            return max_edge, max_gain
        return None, -1
        
    def compute_community_safeness(self, community_target: List[int]) -> float:
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
                self.graph, community_target, node)
            # print(f"Node {node} safeness: {node_safeness}")
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
