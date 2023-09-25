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

class Modularity:
    def __init__(
        self, 
        beta: int,
        graph: nx.Graph, 
        community_target: List[int],
        communities_object: cdlib.NodeClustering,):
        
        self.nx_graph = graph.copy()
        self.graph = ig.Graph.from_networkx(self.nx_graph)
        # Due to a bug in cdlib, we need to rename the nodes
        self.graph.vs["name"] = self.graph.vs["name"]=range(0,len(self.graph.vs)) # self.graph.vs["_nx_name"]
        del (self.graph.vs["_nx_name"])
        # Add weight attribute to the edges equal to 1
        self.graph.es["weight"] = 1
        
        self.target_community = community_target
        self.communities_object = copy.deepcopy(communities_object)
        self.communities = communities_object.communities
        self.budget = beta
        # set
        self.initial_community_size = len(self.target_community)
        self.community_size = len(self.target_community)
        self.induced_subgraph = self.getInducedSubgraph()
        self.community_membership_dict = self.nodeCommunityDict()
        self.target_com_node_degrees = self.getTargetNodeDegrees()
        # for each node i in comH, find its degree in its community
        self.degi_Ci = self.getTargetNodeDegreePerCommunity()
        # Number of internal edges for each community
        self.E_Ci = self.communities_object.edges_inside(summary=False)
        self.community_degrees = self.getCommunityDegrees()
        self.node_count = self.graph.vcount()

        # Detection time
        self.time_detection = None
        self.communities_after = None
        self.communities_after_object = None
        self.ratio_community_members = None

        self.getDeceptionScore(after_updates=False)

    ############################################################################
    #                           COMMUNITY DECEPTION                            #
    ############################################################################
    def run(self):
        beta = self.budget
        while (True):
            # this gives the best value the one giving hghest DELTA_del
            delta_del, best_node_to_delete = self.computeDeltaNodeDeletion()
            # this gives a single value as we are adding a new node
            delta_add, deg_i_Cj, deg_i, Cj_index = self.computeDeltaNodeAddition()
            # this gives a value for each community member
            delta_move, move_details = self.computeDeltaNodeMoving()

            deltas = [delta_add, delta_del, delta_move]
            best_index = deltas.index(max(deltas))
            # print(deltas)
            # print(best_index)
            if delta_move > 0:
                self.performNodeMove(move_details)
                # print("Move")
            else:
                if delta_del > delta_add:
                    self.performNodeDeletion(best_node_to_delete)
                    self.performNewNodeAddition(deg_i_Cj, deg_i, Cj_index)
                    print("Add")
                else:
                    self.performNewNodeAddition(deg_i_Cj, deg_i, Cj_index)
                    print("Del")
            beta = beta - 1
            # if the community contains all -1 then stop !
            if (beta == 0) or self.target_community.count(-1) == self.initial_community_size-2:
                break
            # else:
            #    break
        steps = budget - beta

        nx_graph = self.graph.to_networkx()
        nx.set_edge_attributes(nx_graph, values=1, name='weight')
        return nx_graph, steps

    def performNodeDeletion(self, node_to_delete):
        neighs = self.graph.neighborhood(
            vertices=node_to_delete, order=1, mode="all", mindist=1)
        for n in neighs:
            self.graph.delete_edges([(node_to_delete, n)])

        nodex_index_in_com = self.target_community.index(node_to_delete)

        self.target_community[nodex_index_in_com] = -1

        node_com = self.getNodeCommunity(node_to_delete)

        copy_coms = self.communities

        if node_to_delete < max(self.community_membership_dict.keys()):
            if len(copy_coms[node_com]) > 1:  # this avoids to have zero-nodes communities
                copy_coms[node_com].remove(node_to_delete)

        self.communities = copy_coms
        degree_in_com_copy = self.degi_Ci
        node_gree_in_its_com = degree_in_com_copy[nodex_index_in_com, node_com]

        degree_in_com_copy[nodex_index_in_com] = 0
        self.degi_Ci = degree_in_com_copy
        coms_degs_copy = self.community_degrees
        coms_degs_copy[node_com] = coms_degs_copy[node_com] - \
            node_gree_in_its_com
        self.community_degrees = coms_degs_copy
        internal_edges_copy = self.E_Ci
        internal_edges_copy[node_com] = internal_edges_copy[node_com] - \
            node_gree_in_its_com
        self.E_Ci = internal_edges_copy

    def performNewNodeAddition(self, degree_i_target_com, total_degree_i, target_com_index):
        new_node_id = self.graph.vcount()
        self.graph.add_vertices(1)
        target_com = self.communities[target_com_index]
        target_nodes_indexes = random.sample(
            range(0, len(target_com)), degree_i_target_com)

        for i in target_nodes_indexes:
            self.graph.add_edges([(new_node_id, target_com[i])])
        self.target_community.append(new_node_id)
        copy_coms = self.communities
        copy_coms[target_com_index].append(new_node_id)
        self.communities = copy_coms
        nodex_index_in_com = self.target_community.index(new_node_id)

        degree_in_com_copy = self.degi_Ci
        newrow = np.zeros(len(self.communities))
        newrow[target_com_index] = degree_i_target_com
        degree_in_com_copy = np.vstack([degree_in_com_copy, newrow])
        self.degi_Ci = degree_in_com_copy

        coms_degs_copy = self.community_degrees
        coms_degs_copy[target_com_index] = coms_degs_copy[target_com_index] + \
            degree_i_target_com
        self.community_degrees = coms_degs_copy
        internal_edges_copy = self.E_Ci
        internal_edges_copy[target_com_index] = internal_edges_copy[target_com_index] + \
            degree_i_target_com
        self.E_Ci = internal_edges_copy
        return self.graph

    def performNodeMove(self, move_details):
        node_to_move = int(move_details[0])  # node to move from Ci in Cj
        new_community = int(move_details[2])  # Cj
        old_community = self.getNodeCommunity(node_to_move)  # Ci
        new_edges_new_com = move_details[3]  # total edges in Cj
        edges_to_be_deleted = move_details[4]  # edges in in Ci
        # nodes tha alredy has in Cj
        node_deg_already_new_com = move_details[5]
        nodex_index_in_com = self.target_community.index(node_to_move)

        target_com = self.communities[new_community]
        target_nodes_indexes = range(0, len(target_com))

        # EDGE ADDITIONS in Cj
        for i in target_nodes_indexes:
            if not (self.graph.are_connected(node_to_move, target_com[i]) and self.graph.are_connected(target_com[i], node_to_move)):
                self.graph.add_edges([(node_to_move, target_com[i])])

        # EDGE DELETIONS in Ci
        neighs = self.graph.neighborhood(
            vertices=node_to_move, order=1, mode="all", mindist=1)
        for n in neighs:
            if old_community == self.getNodeCommunity(n):
                if (self.graph.are_connected(node_to_move, n)):
                    self.graph.delete_edges([(node_to_move, n)])
                else:
                    if (self.graph.are_connected(n, node_to_move)):
                        self.graph.delete_edges([(n, node_to_move)])
        degree_in_com_copy = self.degi_Ci
        node_gree_in_its_com = degree_in_com_copy[nodex_index_in_com, old_community]

        # its degree becomes 0 in Ci
        degree_in_com_copy[nodex_index_in_com, old_community] = 0

        # degree_in_com_copy[nodex_index_in_com, new_community]
        degree_in_com_copy[
            nodex_index_in_com,
            new_community] = new_edges_new_com-node_deg_already_new_com

        self.degi_Ci = degree_in_com_copy

        coms_degs_copy = self.community_degrees
        coms_degs_copy[old_community] = coms_degs_copy[old_community] - \
            node_gree_in_its_com
        coms_degs_copy[new_community] = coms_degs_copy[new_community] + \
            (new_edges_new_com-node_deg_already_new_com)
        self.community_degrees = coms_degs_copy

        internal_edges_copy = self.E_Ci
        internal_edges_copy[old_community] = internal_edges_copy[old_community] - \
            node_gree_in_its_com
        internal_edges_copy[new_community] = internal_edges_copy[new_community] + \
            (new_edges_new_com-node_deg_already_new_com)
        self.E_Ci = internal_edges_copy
        return self.graph
    
    ############################################################################
    #                          MODULARITY UTILITY                              #
    ############################################################################
    def getModularity(self, graph, communities):
        nx_graph = graph.to_networkx()
        nx.set_edge_attributes(nx_graph, values = 1, name = 'weight')
        return evaluation.newman_girvan_modularity(nx_graph, communities).score

    def getCommunityDegrees(self):
        com_degs = []
        for com_index in range(len(self.communities)):
            community = self.communities[com_index]
            com_degs.append(sum(self.graph.degree(community)))
        return np.array(com_degs)

    def getTargetNodeDegreePerCommunity(self):
        # print(self.communities.subgraph(1))
        internal_degree_per_com_matrix = np.zeros(
            (len(self.target_community), len(self.communities)))
        for com_index in range(0, len(self.communities)):
            community = self.communities[com_index]
            for community_node_index in range(0, len(community)):
                for member_of_target_index in range(0, len(self.target_community)):
                    if self.graph.are_connected(community[community_node_index], self.target_community[
                        member_of_target_index]) or self.graph.are_connected(
                            self.target_community[member_of_target_index], community[community_node_index]):
                        internal_degree_per_com_matrix[member_of_target_index, com_index] = \
                            internal_degree_per_com_matrix[member_of_target_index,
                                                           com_index] + 1
        return internal_degree_per_com_matrix

    def getCommunityBridges(self):
        bridge_edges = []
        bridge_edges_original_ids = []
        indG = self.induced_subgraph
        numCompConn = len(ig.Graph.decompose(indG))
        for e in indG.es:
            copy_induced = copy.deepcopy(indG)
            copy_induced.delete_edges(e)
            new_number = len(copy_induced.decompose())
            if (new_number) > numCompConn:
                bridge_edges.append(
                    (indG.vs[e.source].index, indG.vs[e.target].index))
                bridge_edges_original_ids.append(
                    (indG.vs[e.source]["name"], indG.vs[e.target]["name"]))
        return bridge_edges_original_ids, bridge_edges

    def getOriginalGraphNodeLabel(self, graph, node):
        return graph.vs[node]["name"]

    def getInducedGraphNodeId(self, graph, nodeLabel):
        return graph.vs[nodeLabel]["name"]

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def getTargetCommunityID(self):
        com_len = [len(self.communities[i])
                   for i in range(0, len(self.communities))]
        preferred_size = int(np.ceil(max(com_len)*0.5))/2
        closest = self.find_nearest(np.array(com_len), preferred_size)
        target_community_index = com_len.index(closest)
        return target_community_index

    def getTargetCommunityNOWorstCase(self):
        com_len = [len(self.communities[i])
                   for i in range(0, len(self.communities))]
        preferred_size = int(np.ceil(max(com_len) * 0.5)) / 2
        number_of_nodes_in_target_com = self.find_nearest(
            np.array(com_len), preferred_size)
        target_com = random.sample(
            range(0, self.graph.vcount()), number_of_nodes_in_target_com)
        return target_com

    def getInducedSubgraph(self):
        com_excluding_deleted = self.target_community
        com_excluding_deleted[:] = (
            value for value in com_excluding_deleted if value != -1)
        return self.graph.induced_subgraph(com_excluding_deleted, implementation="auto")

    def getTargetNodeDegrees(self):
        degrees = np.zeros(len(self.target_community))
        for i in range(0, len(self.target_community)):
            if self.target_community[i] == -1:
                degrees[i] = 0
            else:
                degrees[i] = self.graph.degree(
                    (self.target_community[i]), mode="all")
        return degrees

    def getNodeTotalDegreesInducedSubgraph(self):
        res = self.getInducedSubgraph().degree(
            range(0, self.community_size), mode="all")
        res = [0 if math.isnan(i) else i for i in res]
        return res

    def belongToCommunity(self, node, com_id):
        return node in self.communities[com_id]

    def nodeCommunityDict(self):
        communityDictionary = dict()
        index = 0
        for comm in self.communities:
            for co in comm:
                communityDictionary.update({co: index})
            index += 1
        return communityDictionary

    def getNodeCommunity(self, node):
        # if node <= self.node_count:
        return self.community_membership_dict[node]
        # else:
        #    return -1

    def convertEdgeIdInducedToOriginal(self, e):
        converted_e = tuple((self.getOriginalGraphNodeLabel(self.induced_subgraph, e.source),
                             self.getOriginalGraphNodeLabel(self.induced_subgraph, e.target), e["weight"]))
        return converted_e

    def getDeceptionScore(self, after_updates):
        number_communities = len(self.communities)
        if after_updates == False:
            community = self.target_community
        else:
            community = self.target_community
            community[:] = (value for value in community if value != -1)
        member_for_community = []
        for member in community:
            current_community_member = [
                1 if member in community else 0 for community in self.communities]
            member_for_community.append(current_community_member)

        member_for_community = [sum(x) for x in zip(*member_for_community)]
        ratio_community_members = [members_for_c / len(com) for (members_for_c, com) in
                                   zip(member_for_community, self.communities)]
        self.ratio_community_members = ratio_community_members
        spread_members = sum(
            [1 if value_per_com > 0 else 0 for value_per_com in ratio_community_members])

        second_part = 1 / 2 * ((spread_members - 1) / number_communities) + 1 / 2 * (
            1 - sum(ratio_community_members) / spread_members)

        num_components = len(self.getInducedSubgraph().decompose())
        first_part = 1 - ((num_components - 1) / (self.community_size - 1))

        dec_score = first_part * second_part

        return dec_score

    def plot_communities(self, communities):
        node_labels = range(0, self.graph.vcount())
        ig.plot(communities, mark_groups=True,
                vertex_size=20, vertex_label=node_labels)
    ########################################### END UTILITIES ##############################

############# NODEC #####################
    # For each member of the target community compute the modularity change if deleted
    def computeDeltaNodeDeletion(self):
        total_internal_edges = sum(np.array(self.E_Ci))/(2)
        range_nodes = range(0, len(self.target_community))

        node_internal_degree = [self.degi_Ci[index][self.getNodeCommunity(node)] for (
            index, node) in zip(range_nodes, self.target_community)]
        m = self.graph.ecount()
        degrees = np.array(self.getTargetNodeDegrees())
        q1_links = (total_internal_edges - node_internal_degree)/(m - degrees)
        group_degs = self.community_degrees
        node_deg_by_group = self.degi_Ci
        expected_impact = np.power((group_degs-node_deg_by_group), 2).sum(1)
        q1_degrees = expected_impact / (4 * (m - degrees) ** 2)
        modularity_after = q1_links - q1_degrees
        modularity_before = self.getModularity(
            self.graph, self.communities_object)
        delta_deletion = (modularity_before - modularity_after).tolist()

        if -1 in self.target_community:
            mask = np.in1d(self.target_community, [-1])
            marr = np.ma.masked_array(mask=mask, data=delta_deletion)
            delta_deletion = marr

        best_index = np.argmax(delta_deletion)
        best_node_to_delete = self.target_community[best_index]

        return delta_deletion[best_index], best_node_to_delete

    def computeDeltaNodeAddition(self):
        ordered_spread = self.ratio_community_members
        Cj_index = int(ordered_spread.index(
            ordered_spread[np.argmin(ordered_spread)]))
        Cj = self.communities[Cj_index]
        internal_edges_Cj = self.E_Ci[Cj_index]
        degree_Cj = self.community_degrees[Cj_index]

        deg_i_Cj = int(len(Cj)-(math.ceil(len(Cj))*0.1))

        deg_i = int(math.ceil(deg_i_Cj*1.2))
        M = self.graph.ecount()
        M1 = M+deg_i

        A = (internal_edges_Cj + deg_i_Cj)/(2 * M1)
        B = math.pow((degree_Cj + 2 * deg_i), 2)/(4 * M1*M1)
        C = (internal_edges_Cj)/(2*M)
        D = (-(math.pow(degree_Cj, 2))-math.pow(deg_i, 2))/(4 * M*M)

        delta_add = A-B-C-D
        return delta_add, deg_i_Cj, deg_i, Cj_index

    def computeDeltaNodeMoving(self):
        nodes_delta_mov = []
        ordered_spread = self.ratio_community_members
        for node in self.target_community:
            if node != -1:
                node_to_move = node
                node_to_move_current_degree = self.graph.degree(node_to_move)
                current_community = self.getNodeCommunity(node_to_move)
                current_community_internal_edges = self.E_Ci[current_community]
                current_community_degree = self.community_degrees[current_community]

                node_deg_current_com = \
                    self.degi_Ci[self.target_community.index(
                        node)][current_community]

                new_community = ordered_spread.index(
                    ordered_spread[np.argmin(ordered_spread)])

                if new_community != current_community:
                    new_community_degree = self.community_degrees[new_community]
                    new_community_nodes = len(self.communities[new_community])

                    new_community_internal_edges = self.E_Ci[new_community]

                    node_deg_already_new_com = \
                        self.degi_Ci[self.target_community.index(
                            node)][new_community]

                    new_edges_new_com = int(
                        math.ceil(new_community_nodes-node_deg_already_new_com)*0.9)

                    edges_to_be_deleted = node_deg_current_com
                    new_node_degree = node_to_move_current_degree + \
                        new_edges_new_com-edges_to_be_deleted
                    M = self.graph.ecount()
                    M1 = M+new_node_degree
                    removing_i_from_Ci = ((current_community_internal_edges-(node_deg_current_com))/2*M-(
                        (current_community_degree-node_deg_current_com)**2))/(4*M*M)
                    adding_i_to_Cj = ((new_community_internal_edges+(new_edges_new_com))/4*M1*M1-(
                        (new_community_degree+new_edges_new_com)**2))/(4*M1*M1)
                    delta_move = removing_i_from_Ci+adding_i_to_Cj
                    nodes_delta_mov.append(np.array(
                        [node_to_move, delta_move, new_community, new_edges_new_com, edges_to_be_deleted, node_deg_already_new_com]))
                else:
                    nodes_delta_mov.append(
                        np.array([node_to_move, -1, -1, -1, -1, -1]))
            else:
                nodes_delta_mov.append(np.array([-1, -1, -1, -1, -1, -1]))

        nodes_delta_mov = np.array(nodes_delta_mov)
        best_index = np.argmax(nodes_delta_mov[:, 1])
        best_delta = nodes_delta_mov[best_index][1]
        #
        return best_delta, nodes_delta_mov[best_index]
# END
