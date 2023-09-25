# Please treat the code as confidential
from cdlib import evaluation
import igraph as ig
import numpy as np
import copy
import random
import timeit
import math

import networkx as nx

import cdlib
from typing import List


class Safeness:
    def __init__(
            self,
            budget: int,
            graph: nx.Graph,
            community_target: List[int],
            communities_object: cdlib.NodeClustering):

        self.nx_graph = graph.copy()
        self.graph = ig.Graph.from_networkx(self.nx_graph)
        # Due to a bug in cdlib, we need to rename the nodes
        self.graph.vs["name"] = self.graph.vs["name"] = range(
            0, len(self.graph.vs))  # self.graph.vs["_nx_name"]
        del (self.graph.vs["_nx_name"])
        # Add weight attribute to the edges equal to 1
        self.graph.es["weight"] = 1
        
        self.target_community = community_target
        self.communities_object = copy.deepcopy(communities_object)
        self.communities = communities_object.communities
        self.budget = budget
        
        # Useful variables for the algorithm
        edge_sequence = self.graph.es
        # Convert in tuple objects
        self.edge_sequence = [e.tuple for e in edge_sequence]
        self.num_vertices = self.graph.vcount()
        self.adjacency_list = None
        self.deg = None
        self.out_deg = None
        self.out_ratio = None
        self.new_adjacency_list = None
        self.igraph_edge_list = None
        self.new_edge_list = None
        # Compute all the variables
        self.set_all()

    
    def run(self):
        add_gain = 0
        del_gain = 0
        intra_considered = []
        beta = self.budget
        while (True):
            node_list = self.get_min_NodeRatio_index()

            (add_gain, add_node_ind) = self.min_index_edge_addition(node_list)
            add_node = self.target_community[add_node_ind]
            add_node_2 = self.findExternalNode(add_node)

            li = self.getBestDelExclBridges()

            ((del_node, del_node_2), max_gain) = self.deletion_Gain(li, intra_considered)
            del_gain = max_gain

            if add_gain >= del_gain and add_gain > 0 and add_node_2 != None:
                self.igraph_edge_list.append((add_node, add_node_2))
                for i in self.target_community:
                    deg_ = 0
                    out_deg_ = 0
                    for j in self.igraph_edge_list:
                        if i == j[0] or i == j[1]:
                            deg_ = deg_ + 1
                            if (i == j[0] and j[1] not in self.target_community) or \
                                (i == j[1] and j[0] not in self.target_community):
                                out_deg_ = out_deg_ + 1
                    
                    self.deg[self.target_community.index(i)] = deg_
                    self.out_deg[self.target_community.index(i)] = out_deg_

                for i, _ in enumerate(self.out_ratio):
                    self.out_ratio[i] = self.out_deg[i] / self.deg[i]
                    
            elif del_gain > 0:
                self.igraph_edge_list.remove((del_node, del_node_2))
                intra_considered.append((del_node, del_node_2))
                for i in self.target_community:
                    deg_ = 0
                    out_deg_ = 0
                    for j in self.igraph_edge_list:
                        if i == j[0] or i == j[1]:
                            deg_ = deg_ + 1
                            if (i == j[0] and j[1] not in self.target_community) or \
                                (i == j[1] and j[0] not in self.target_community):
                                out_deg_ = out_deg_ + 1
                    
                    self.deg[self.target_community.index(i)] = deg_
                    self.out_deg[self.target_community.index(i)] = out_deg_

                for i, _ in enumerate(self.out_ratio):
                    self.out_ratio[i] = self.out_deg[i] / self.deg[i]
                
                self.new_edge_list.remove((del_node, del_node_2))
                self.new_adjacency_list[del_node].remove(del_node_2)
                self.new_adjacency_list[del_node_2].remove(del_node)
                
            beta = beta - 1
            if (beta > 0 and (add_gain > 0 or del_gain > 0)):
                continue
            else:
                break
        steps = self.budget - beta
        # print("iGraph Edge List: ", self.igraph_edge_list)
        
        # Convert the edge list in a networkx graph
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(self.nx_graph.nodes)
        
        nx_graph.add_edges_from(self.igraph_edge_list)
        return nx_graph, steps

    
    
    ############################################################################
    #                           SAFENESS UTILITY                               #
    ############################################################################
    def get_min_NodeRatio_index(self):
        min_val = min(self.out_ratio)
        node = []
        for i, _ in enumerate(self.out_ratio):
            if self.out_ratio[i] == min_val:
                node.append(i)
        return node
    
    def min_index_edge_addition(self, node_list):
        node_ind = 0
        max_gain = 0
        for i in node_list:
            gain = 0.5*(
                (self.out_deg[i]+1) / (self.deg[i]+1) - 
                self.out_deg[i]/self.deg[i]
                )
            if gain > max_gain:
                max_gain = gain
                node_ind = i
        return (max_gain, node_ind)
    
    def findExternalNode(self, com_node):
        for i in self.communities:
            if i != self.target_community:
                for j in i:
                    if ((com_node, j) or (j, com_node)) not in self.igraph_edge_list:
                        return j
    
    def getBestDelExclBridges(self):
        best_edges = []
        for i in self.new_edge_list:
            Cpy_Adj_List = copy.deepcopy(self.new_adjacency_list)
            Cpy_Adj_List[i[0]].remove(i[1])
            Cpy_Adj_List[i[1]].remove(i[0])
            try:
                if self.connectedComponents(self.target_community, self.num_vertices, Cpy_Adj_List) == 1:
                    best_edges.append(i)
            except:
                continue
        return best_edges
    
    def deletion_Gain(self, li, intra_considered):
        max_gain = 0
        node_u = 0
        node_v = 0
        for i in li:
            if i not in intra_considered:
                u = i[0]
                v = i[1]
                gain = (
                        self.out_deg[self.target_community.index(u)] /
                        (2*self.deg[self.target_community.index(u)] *
                            (self.deg[self.target_community.index(u)]-1)
                        )
                    ) + \
                    (
                        self.out_deg[self.target_community.index(v)] /
                        (2*self.deg[self.target_community.index(v)] *
                            (self.deg[self.target_community.index(v)]-1))
                ) + (1/(len(self.target_community) - 1))
                if (gain > max_gain):
                    max_gain = gain
                    node_u = u
                    node_v = v
        return ((node_u, node_v), max_gain)

    ############################################################################
    #                               UTILS                                      #
    ############################################################################
    def connectedComponents(self, target_comm, num_vertices, Adjacency_List):
        visited = []
        cc = []
        for i in range(num_vertices):
            visited.append(False)
        for v in range(num_vertices):
            if visited[v] == False:
                temp = []
                cc.append(self.DFSUtil(target_comm, temp, v, visited, Adjacency_List))
        return len(cc)
    
    def DFSUtil(self, target_comm, temp, v, visited, Adjacency_List):
        visited[v] = True
        temp.append(v)
        for i in Adjacency_List[target_comm[v]]:
            if visited[target_comm.index(i)] == False:
                temp = self.DFSUtil(target_comm, temp, target_comm.index(i),
                            visited, Adjacency_List)
        return temp
    
    
    ############################################################################
    #                            SETTER METHODS                                #
    ############################################################################
    def set_all(self):
        # Compute adjacency list
        self.adjacency_list = self.set_adj_list()
        igraph_edge_list, g = self.set_igraph_edge_list()
        self.deg = self.set_node_degree(g)
        self.out_deg = self.set_node_out_degree()
        self.out_ratio = self.set_node_out_ratio()
        self.new_adjacency_list = self.set_new_adjacency_list()
        new_edge_list = self.set_new_edge_list(igraph_edge_list)
        intra_community_edges = []
        self.igraph_edge_list = list(set(igraph_edge_list))
        self.new_edge_list = list(set(new_edge_list))
        
    def set_adj_list(self):
        """Compute adjacency list from edge sequence"""
        adjacency_list = {}
        for i, _ in enumerate(self.edge_sequence):
            s, t = self.edge_sequence[i][0], self.edge_sequence[i][1]
            if (s in adjacency_list.keys()):
                adjacency_list[s].append(t)
            else:
                adjacency_list[s] = []
                adjacency_list[s].append(t)
            if (t in adjacency_list.keys()):
                adjacency_list[t].append(s)
            else:
                adjacency_list[t] = []
                adjacency_list[t].append(s)
        return adjacency_list
    
    def set_igraph_edge_list(self):
        igraph_edge_list = []
        for i in self.edge_sequence:
            igraph_edge_list.append((i[0], i[1]))
        # Create a new graph object
        g = ig.Graph(directed=False)
        g.add_vertices(self.num_vertices)
        g.add_edges(igraph_edge_list)
        return igraph_edge_list, g
    
    def set_node_degree(self, g: ig.Graph)->List[int]:
        """
        Compute a list containing the degree of each node in the target 
        community
        """
        deg = []
        for i in self.target_community:
            deg.append(g.vs[i].degree())
        return deg
    
    def set_node_out_degree(self)->List[int]:
        """
        Compute a list containing the out degree of each node in the target 
        community, i.e. the number of edges that start from a node in the target
        community and end in a node outside the target community"""
        out_deg = []
        for i in self.target_community:
            _out = 0
            for j in self.adjacency_list[i]:
                if (j) not in self.target_community:
                    _out += 1
            out_deg.append(_out)
        return out_deg
    
    def set_node_out_ratio(self)->List[float]:
        """
        Compute a list containing the out degree ratio of each node in the
        target community, i.e. the ratio between the out degree and the degree
        of each node in the target community
        """
        out_ratio = []
        for i, _ in enumerate(self.out_deg):
            out_ratio.append(self.out_deg[i]/self.deg[i])
        return out_ratio

    def set_new_adjacency_list(self)->List[List[int]]:
        """
        Compute the adjacency dict containing as keys only the nodes in the
        target community and as values a list of nodes in the target community
        """
        new_adjacency_list = {}
        for i in self.adjacency_list.keys():
            if i in self.target_community:
                new_adjacency_list[i] = []
                for j in self.adjacency_list[i]:
                    if j in self.target_community:
                        new_adjacency_list[i].append(j)
        return new_adjacency_list
    
    def set_new_edge_list(self, igraph_edge_list) -> List[List[int]]:
        """
        Compute the edge list containing only the edges with both nodes in
        the target community
        """
        new_edge_list = []
        for i in igraph_edge_list:
            if i[0] in self.target_community and i[1] in self.target_community:
                new_edge_list.append(i)
        return new_edge_list
