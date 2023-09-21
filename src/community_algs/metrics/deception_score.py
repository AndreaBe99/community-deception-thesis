"""Module for calculating deception score of a community detection algorithm."""
from typing import List
import numpy as np
import networkx as nx

class DeceptionScore(object):
    """Deception score of a community detection algorithm."""

    def __init__(self, community_target: List[int]) -> None:
        self.community_target = community_target

    @staticmethod
    def recall(g_i: List[int], community_target: List[int]) -> float:
        """Calculate recall score of a community g_i

        Parameters
        ----------
        g_i : List[int]
            Community found by a community detection algorithm.

        Returns
        -------
        float
            Recall score of g_i.
        """
        # Number of members in g_i that are also in our community
        members_in_g_i = len(set(community_target) & set(g_i))
        return members_in_g_i / len(community_target)

    @staticmethod
    def precision(g_i: List[int], community_target: List[int]) -> float:
        """Calculate precision score of a community g_i

        Parameters
        ----------
        g_i : List[int]
            Community found by a community detection algorithm.

        Returns
        -------
        float
            Precision score of g_i.
        """
        # Number of members in G_i that are also in our community
        members_in_g_i = len(set(community_target) & set(g_i))
        return members_in_g_i / len(g_i)

    @DeprecationWarning
    def compute_deception_score(
            self,
            community_structure: List[List[int]],
            connected_components: int) -> float:
        """Calculate deception score of a community detection algorithm.

        Parameters
        ----------
        community_structure : List(List(int))
            Community structure found by a community detection algorithm.
        connected_components : int
            Number of connected components in the graph.
        
        Returns
        -------
        deception_score : float
            Deception score of a community detection algorithm.
        """
        # Number of intersecting nodes between the community structure and community target
        n_intersecting_nodes = [g_i for g_i in community_structure if len(
            set(self.community_target) & set(g_i)) > 0]
        
        recall = max([self.recall(g_i, self.community_target) for g_i in community_structure])
        precision = sum([self.precision(g_i, self.community_target) for g_i in n_intersecting_nodes])
        
        # Ideal situation occurs when each member of the community target is 
        # placed in a different community and the value of the maximum recall 
        # is lower possible.
        assert len(self.community_target) - 1 > 0, "Community target must have at least 2 members."
        community_spread = 1 - (connected_components - 1) / (len(self.community_target) - 1)
        
        # Ideal situation occurs when each member of the community structure 
        # contains little percentage of the community target.
        assert len(n_intersecting_nodes) > 0, "Community structure must have at least 1 member."
        community_hiding = 0.5 * (1 - recall) + 0.5 * (1 - precision / len(n_intersecting_nodes))
        
        # Deception score is the product of community spread and community hiding. 
        deception_score = community_spread * community_hiding
        return deception_score
    
    # TEST
    def get_deception_score(self, graph, community_structure: List[List[int]]):
        """
        New version of the deception score, based on the repository:
            - https://github.com/vfionda/BHC/tree/main

        Parameters
        ----------
        community_structure : List[List[int]]
            _description_

        Returns
        -------
        _type_
            _description_
        """
        number_communities = len(community_structure)
        
        # Number of the target community members in the various communities
        member_for_community = np.zeros(number_communities, dtype=int)
        
        for i in range(number_communities):
            for node in community_structure[i]:
                if node in self.community_target:
                    member_for_community[i] += 1
        
        # ratio of the targetCommunity members in the various communities
        ratio_community_members = [members_for_c/len(com) for (members_for_c, com) in zip(member_for_community, community_structure)]
        
        # In how many commmunities are the members of the target spread?
        spread_members = sum([1 if mc > 0 else 0 for mc in member_for_community])
        
        second_part = 1 / 2 * ((spread_members - 1) / number_communities) + \
            1/2 * (1 - sum(ratio_community_members) / spread_members)
        
        # induced subraph sonly on target community nodes
        num_components = nx.number_connected_components(
            graph.subgraph(self.community_target))
        first_part = 1 - ((num_components - 1) / (len(self.community_target) - 1))
        dec_score = first_part * second_part
        return dec_score


if __name__ == "__main__":
    # Test  
    # community = [1, 2, 3] # Community target to hide
    community = [1]
    deception = DeceptionScore(community)
    
    # Completly Detected
    structure = [[1, 2, 3], [4, 5], [6, 7, 8], [9, 10]]
    print(deception.compute_deception_score(structure, 1))
    
    # Completly Hidden
    structure = [[1, 4, 5], [2, 6, 7, 8], [3, 9, 10]]
    print(deception.compute_deception_score(structure, 1))
