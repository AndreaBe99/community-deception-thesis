import networkx as nx
import numpy as np
import cdlib
from typing import List, Set, Tuple
import copy

class Safeness:
    """Class that implements the Safeness algorithm"""
    def __init__(
            self,
            budget: int,
            graph: nx.Graph,
            community_target: List[int],
            communities_object: cdlib.NodeClustering):
        self.budget = budget
        self.graph = graph.copy()
        self.community_target = community_target
        self.community_obj = communities_object

    def getNodeMinimumAddRatio(self, graph):
        """Computes, for each n∈C, the fraction of n’s edges that point outside C"""
        community_degree = dict()
        for n in self.community_target:
            n_neighbors = graph.neighbors(n)
            n_outside = 0
            for neighbor in n_neighbors:
                if neighbor not in self.community_target:
                    n_outside += 1
            if graph.degree(n) > 0:
                community_degree[n] = n_outside / graph.degree(n)
            else:
                community_degree[n] = 0
        # Return the key which its value is the minimum
        return min(community_degree, key=community_degree.get)

    def findExternalNode(self, graph, np):
        """Finds a node (not in C) such that the edge (np, nt) does not exist"""
        for community in self.community_obj.communities:
            if community != self.community_target:
                for nt in community:
                    if not graph.has_edge(np, nt):
                        return nt

    def getAdditionGain(self, graph, np, nt):
        """Computes the addition gain ξaddC"""
        temp_graph = graph.copy()
        safeness_before = self.compute_community_safeness(graph=temp_graph)
        temp_graph.add_edge(np, nt)
        safeness_after = self.compute_community_safeness(graph=temp_graph)
        return safeness_after - safeness_before

    def getBestDelExclBridges(self, graph):
        """
        Excludes bridge edges that, if deleted, could disconnect C.
        Computes the value specified in Theorem 8 for each remaining edge
        Returns the most convenient (safeness-wise) edge update
        """
        # get subgraph induced by the nodes in community C
        subgraph = graph.subgraph(self.community_target).copy()
        
        temp_subgraph = subgraph.copy()
        for edge in subgraph.edges():
            temp_subgraph.remove_edge(*edge)
            if not nx.is_connected(temp_subgraph):
                subgraph.remove_edge(*edge)
        
        community_deletion = {} # {edge: eps_del}
        for edge in subgraph.edges():
            community_deletion[edge] = self.getDeletionGain(subgraph, edge[0], edge[1])
        
        if len(community_deletion) < 1:
            return None, None
        return max(community_deletion, key=community_deletion.get)

    def getDeletionGain(self, graph, nk, nl):
        """Computes the deletion gain ξdelC"""
        temp_graph = graph.copy()
        safeness_before = self.compute_community_safeness(graph=temp_graph)
        temp_graph.remove_edge(nk, nl)
        safeness_after = self.compute_community_safeness(graph=temp_graph)
        return safeness_after - safeness_before

    def run(self):
        """Run Safeness algorithm"""
        graph = self.graph.copy()
        beta = self.budget
        while beta > 0:
            np = self.getNodeMinimumAddRatio(graph)
            nt = self.findExternalNode(graph, np)
            xi_add_C = self.getAdditionGain(graph, np, nt)
            
            (nk, nl) = self.getBestDelExclBridges(graph)
            if nk is None and nl is None:
                xi_del_C = -1
            else:
                xi_del_C = self.getDeletionGain(graph, nk, nl)
            
            if xi_add_C >= xi_del_C and xi_add_C > 0:
                graph.add_edge(np, nt)
            elif xi_del_C > 0:
                graph.remove_edge(nk, nl)
            beta -= 1
            if xi_add_C <= 0 and xi_del_C <= 0:
                break
            
        return graph, self.budget - beta
    
    def compute_community_safeness(self, graph):
        """Computes the community safeness of the community"""
        safeness_sum = 0
        for node in self.community_target:
            safeness_sum += self.compute_node_safeness(graph, node)
        return safeness_sum / len(self.community_target)
    
    def compute_node_safeness(self, graph, n):
        """Computes the node safeness of the node"""
        deg_u = graph.degree(n)
        
        E_u_C = []
        E_u_C_bar = []
        for v in graph.neighbors(n):
            if v in self.community_target:
                E_u_C.append((n, v))
            else:
                E_u_C_bar.append((n, v))
        
        # Set of nodes reachable from u passing only through nodes in C
        V_u_C = dict()
        # Create a subgraph induced by the nodes in community C
        subgraph = graph.subgraph(self.community_target)
        # Compute the connected components of the subgraph
        components = list(nx.connected_components(subgraph))
        # Find the component that contains node u
        for u in self.community_target:
            V_u_C[u] = 0
            for component in components:
                if u in component:
                    # Return the number of nodes in the component
                    V_u_C[u] = len(component)
        
        if len(self.community_target) < 1:
            first_part = 0
        else:
            first_part = ((V_u_C[n] - len(E_u_C)) /
                          (len(self.community_target) - 1))
        if deg_u < 1:
            second_part = 0
        else:
            second_part = len(E_u_C_bar) / deg_u
        return 0.5*first_part + 0.5*second_part