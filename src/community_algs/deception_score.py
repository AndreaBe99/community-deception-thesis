"""Module for calculating deception score of a community detection algorithm."""
from typing import List

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
