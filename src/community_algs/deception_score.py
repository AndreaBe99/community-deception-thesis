from typing import List

class DeceptionScore(object):
    def __init__(self, community: List(int)) -> None:
        self.community = community

    @staticmethod
    def recall(G_i: List(int), community: List(int)) -> float:
        """Calculate recall score of a community G_i

        Parameters
        ----------
        G_i : List(int)
            Community found by a community detection algorithm.

        Returns
        -------
        float
            Recall score of G_i.
        """
        # Number of members in G_i that are also in our community
        members_in_G_i = len(set(community) & set(G_i))
        return members_in_G_i / len(community)

    @staticmethod
    def precision(G_i: List(int), community: List(int)) -> float:
        """Calculate precision score of a community G_i

        Parameters
        ----------
        G_i : List(int)
            Community found by a community detection algorithm.

        Returns
        -------
        float
            Precision score of G_i.
        """
        # Number of members in G_i that are also in our community
        members_in_G_i = len(set(community) & set(G_i))
        return members_in_G_i / len(G_i)

    def compute_deception_score(
            self,
            community_structure: List(List(int))) -> float:
        """Calculate deception score of a community detection algorithm.

        Parameters
        ----------
        community_structure : List(List(int))
            Community structure found by a community detection algorithm.
        
        Returns
        -------
        deception_score : float
            Deception score of a community detection algorithm.
        """
        S_C = [G_i for G_i in community_structure if len(
            set(self.community) & set(G_i)) > 0]
        num_S_C = len(S_C)
        num_C = len(self.community)

        max_recall = max([self.recall(G_i, self.community) for G_i in S_C])
        avg_precision = sum([self.precision(G_i, self.community)
                            for G_i in S_C]) / num_S_C

        deception_score = (1 - (num_S_C - 1) / (num_C - 1)) \
            * (0.5 * (1 - max_recall) + 0.5 * (1 - avg_precision))
        return deception_score
