# Typing Hinting
from typing import List, Tuple
from collections import Counter

import math

class NormalizedMutualInformation(object):
    @staticmethod
    def calculate_confusion_matrix(
            communities_old: List[List[int]],
            communities_new: List[List[int]]) -> Counter:
        """
        Calculate the confusion matrix between two sets of communities.
        Where the element (i, j) of the confusion matrix is the number of shared
        members between an initially detected community C_i and the community 
        C_j after deception.

        Parameters
        ----------
        communities_old : List[List[int]]
            Communities before deception
        communities_new : List[List[int]]
            Communities after deception

        Returns
        -------
        confusion_matrix : Counter
            Confusion matrix
        """
        confusion_matrix = Counter()
        #° Avoid to process the same community twice
        #BUG ZeroDivisionError if we use this optimization
        #BUG processed_new = set()
        for i, old in enumerate(communities_old):
            for j, new in enumerate(communities_new):
                #BUG if j not in processed_new:
                intersection = len(set(old) & set(new))
                confusion_matrix[(i, j)] = intersection
                #BUG    if intersection > 0:
                #BUG        processed_new.add(j)
        return confusion_matrix

    @staticmethod
    def calculate_sums(confusion_matrix: Counter) -> Tuple[Counter, Counter, int]:
        """
        Calculate the row sums, column sums and total sum of a confusion matrix.

        Parameters
        ----------
        confusion_matrix : Counter
            Confusion matrix

        Returns
        -------
        (row_sums, col_sums, total_sum) : Tuple[Counter, Counter, int]
            Tuple containing the row sums, column sums and total sum of the
            confusion matrix.
        """
        row_sums = Counter()
        col_sums = Counter()
        total_sum = 0
        for (i, j), value in confusion_matrix.items():
            row_sums[i] += value
            col_sums[j] += value
            total_sum += value
        return row_sums, col_sums, total_sum

    def compute_nmi(
            self,
            communities_old: List[List[int]],
            communities_new: List[List[int]]) -> float:
        """
        Calculate the normalized mutual information between two sets of
        Communities.

        Parameters
        ----------
        communities_old : List[List[int]]
            List of communities before deception
        communities_new : List[List[int]]
            List of communities after deception

        Returns
        -------
        nmi : float
            Normalized mutual information, value between 0 and 1.
        """
        confusion_matrix = self.calculate_confusion_matrix(
            communities_old, communities_new)
        row_sums, col_sums, total_sum = self.calculate_sums(confusion_matrix)
        
        # Numerator
        nmi_numerator = 0
        for (i, j), n_ij in confusion_matrix.items():
            n_i = row_sums[i]
            n_j = col_sums[j]
            try:
                nmi_numerator += n_ij * math.log((n_ij * total_sum) / (n_i * n_j))
            except ValueError:
                # We could get a math domain error if n_ij is 0
                continue
        
        # Denominator
        nmi_denominator = 0
        for i, n_i in row_sums.items():
            nmi_denominator += n_i * math.log(n_i / total_sum)
        for j, n_j in col_sums.items():
            nmi_denominator += n_j * math.log(n_j / total_sum)
        # Normalized mutual information
        nmi_score = -2 * nmi_numerator / nmi_denominator
        return nmi_score

if __name__ == "__main__":
    # Test
    nmi = NormalizedMutualInformation()
    communities_old = [[1, 2, 3], [4, 5, 6]]
    communities_new = [[1, 2, 3], [4, 5, 6]]
    print(nmi.compute_nmi(communities_old, communities_new))

    communities_old = [[1, 2, 3], [4, 5, 6]]
    communities_new = [[1, 3], [2, 5], [4, 6]]
    print(nmi.compute_nmi(communities_old, communities_new))
