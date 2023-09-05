import networkx as nx
from typing import List, Set

class Safeness:
    """Computes the safeness of a node in a community and the safeness of a community."""
    def __init__(self, graph: nx.Graph, community_target: List[int], nodes_target: List[int]):
        self.graph = graph
        self.community_target = community_target
        self.nodes_target = nodes_target
        # Get the set of nodes reachable from u passing only via nodes in C.
        print("* Compute Community Nodes Reachability...")
        # self.V_u_C = self.compute_reachability()
        # Compute the number of nodes in a community C that are in the same connected component of u
        self.V_u_C = self.num_nodes_in_same_component()
        print("* V_u_C:", self.V_u_C)
        # Get the number of intra-community edges for u.
        print("* Compute Community Intra-Edges...")
        self.E_u_C = self.get_intra_comminty_edges()
        # Get the number of inter-community edges for u.
        print("* Compute Community Inter-Edges...")
        self.E_u_C_bar = self.get_inter_community_edges()
    
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
        for u in self.nodes_target:
            V_u_C[u] = 0
            for component in components:
                if u in component:
                    # Return the number of nodes in the component
                    V_u_C[u] = len(component)
        return V_u_C
    
    def compute_community_safeness(self, community: List[int]) -> float:
        """
        Computes the community safeness of the community.
        
        Parameters
        ----------
        community: List[int]
            The community that we want to compute the safeness.

        Returns
        -------
        float
            The community safeness.
        """
        community_safeness = 0
        for node in self.nodes_target:
            node_safeness = self.compute_node_safeness(node)
            community_safeness += node_safeness
            # print(f"Node {node} safeness: {node_safeness}")
        return community_safeness / len(community)
        
    def compute_node_safeness(self, node: int)->float:
        """
        Computes the node safeness of the node in the community.
        
        Parameters
        ----------
        community: List[int]
            The community of the node that we want to compute the safeness.
        
        node: int
            The node that we want to compute the safeness.

        Returns
        -------
        sigma_u_C: float
            The node safeness.
        """
        # Get the degree of u.
        deg_u = self.graph.degree(node)

        # Compute the node safeness.
        assert len(self.community_target) > 1, "The community must have at least 2 nodes."
        assert deg_u > 0, "The node must have at least 1 edge."
        sigma_u_C = 0.5*((self.V_u_C[node] - len(self.E_u_C)) /
                         (len(self.community_target) - 1)) + 0.5*(len(self.E_u_C_bar) / deg_u)
        return sigma_u_C

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
