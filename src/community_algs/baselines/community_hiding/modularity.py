import networkx as nx
import numpy
import cdlib
from typing import List, Set, Tuple, Callable
import copy


class Modularity:
    """Class that implements the Modularity algorithm"""

    def __init__(
            self,
            beta: int,
            graph: nx.Graph,
            community_target: List[int],
            communities_object: cdlib.NodeClustering,
            detection_alg: Callable):
        self.beta = beta
        self.graph = graph.copy()
        self.community_target = community_target
        self.community_obj = communities_object
        self.detection_agl = detection_alg
        
        self.old_modularity = nx.community.modularity(
            graph, communities_object.communities)
        
        subgraph = self.graph.subgraph(self.community_target)
        edge_mod = {}  
        # {edge: 
        #       {"gain":eps_del, 
        #       "communities":communities_del, 
        #       "mod_after":mod_after_del}}
        for edge in subgraph.edges():
            gain, communities_del, mod_after_del = self.getDelLoss(edge[0], edge[1], self.graph)
            edge_mod[edge] = {"gain": gain, "communities": communities_del, "mod_after": mod_after_del}
        self.edge_mod = edge_mod

    def computeandSortComDegrees(self, graph, communities):
        # Computes and sorts the degrees of communities in C_bar
        community_degree = dict()
        for i, community in enumerate(communities):
            community_degree[i] = {"community": community, "degree": 0}
            for n in community:
                community_degree[i]["degree"] += graph.degree(n)
        community_degree = sorted(
            community_degree.items(), key=lambda x: x[1]["degree"], reverse=True)
        return community_degree

    def getAddLoss(self, np, nt, graph):
        graph = graph.copy()
        # communities_before = self.detection_agl.compute_community(graph)
        # mod_before = nx.community.modularity(graph, communities_before.communities)
        # mod_before = self.computeModularity(graph)
        
        graph.add_edge(np, nt)
        communities_after = self.detection_agl.compute_community(graph)
        mod_after = nx.community.modularity(
            graph, communities_after.communities)
        # mod_after = self.computeModularity(graph)
        
        gain = mod_after - self.old_modularity
        return gain, communities_after, mod_after

    def getDelLoss(self, nk, nl, graph):
        graph = graph.copy()
        # Compute community structure
        # communities_before = self.detection_agl.compute_community(graph)
        # mod_before = nx.community.modularity(graph, communities_before.communities)
        # mod_before = self.computeModularity(graph)
        
        graph.remove_edge(nk, nl)
        communities_after = self.detection_agl.compute_community(graph)
        mod_after = nx.community.modularity(graph, communities_after.communities)
        # mod_after = self.computeModularity(graph)
        
        gain = mod_after - self.old_modularity
        return gain, communities_after, mod_after

    def run(self):
        graph = self.graph.copy()
        beta = self.beta
        communities = self.community_obj
        while beta > 0:
            deg_C = self.computeandSortComDegrees(graph, communities.communities)
            Ci = deg_C[0][1]["community"]
            Cj = deg_C[1][1]["community"]
            
            
            for np in Ci:
                for nt in Cj:
                    if np != nt:
                        if not graph.has_edge(np, nt):
                            break
                        
            MLadd, communities_add, mod_after_add = self.getAddLoss(np, nt, graph)
            
            # Get the edge with the highest value in "gain" key
            MLdel = -1
            for edge in self.edge_mod.keys():
                if self.edge_mod[edge]["gain"] > MLdel:
                    MLdel = self.edge_mod[edge]["gain"]
                    nk = edge[0]
                    nl = edge[1]

            if MLdel >= MLadd and MLdel > 0:
                graph.remove_edge(nk, nl)
                communities = self.edge_mod[(nk, nl)]["communities"]
                self.old_modularity = self.edge_mod[(nk, nl)]["mod_after"]
                self.edge_mod.pop((nk, nl))
            elif MLadd > 0:
                graph.add_edge(np, nt)
                communities = communities_add
                self.old_modularity = mod_after_add
            beta -= 1
            # communities = self.detection_agl.compute_community(graph)
            if MLadd <= 0 and MLdel <= 0:
                break
        return graph, self.beta - beta, communities
    
    def compute_modularity(self, graph, communities):
        eta = 0
        delta = 0
        for community in communities:
            subgraph = graph.subgraph(community)
            E_Ci = subgraph.number_of_edges()
            eta += E_Ci
            deg_Ci = pow(sum(graph.degree(n) for n in community), 2)
            delta += deg_Ci
        
        m = graph.number_of_edges()
        modularity = (eta / m) - (delta / (4 * pow(m, 2)))
        return modularity
