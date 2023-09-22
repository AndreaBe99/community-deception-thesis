# import sys
# sys.path.append("../../")
from typing import List, Callable
from src.utils.utils import SimilarityFunctionsNames

import networkx as nx
import numpy as np


class CommunitySimilarity():
    """Class to compute the similarity between two lists of integers"""
    def __init__(self, function_name: str) -> None:
        self.function_name = function_name
    
    def select_similarity_function(self) -> Callable:
        """
        Select the similarity function to use

        Returns
        -------
        Callable
            Similarity function to use
        """
        if self.function_name == SimilarityFunctionsNames.JAC.value:
            return self.jaccard_similarity
        elif self.function_name == SimilarityFunctionsNames.OVE.value:
            return self.overlap_similarity
        elif self.function_name == SimilarityFunctionsNames.SOR.value:
            return self.sorensen_similarity
        else:
            raise Exception("Similarity function not found")
    
    @staticmethod
    def jaccard_similarity(a: List[int], b: List[int]) -> float:
        """
        Compute the Jaccard similarity between two lists, A and B:
            J(A,B) = |A ∩ B| / |A U B|

        Parameters
        ----------
        a : List[int]
            First List
        b : List[int]
            Second List

        Returns
        -------
        float
            Jaccard similarity between the two lists, between 0 and 1
        """
        assert len(a) > 0 and len(b) > 0, "Lists must be not empty"
        # Convert lists to sets
        a_set = set(a)
        b_set = set(b)
        # Compute the intersection and union
        intersection = a_set.intersection(b_set)
        union = a_set.union(b_set)
        return len(intersection) / len(union)

    @staticmethod
    def overlap_similarity(a: List[int], b: List[int]) -> float:
        """
        Compute the Overlap similarity between two lists, A and B:
            O(A,B) = |A ∩ B| / min(|A|, |B|)

        Parameters
        ----------
        a : List[int]
            First List
        b : List[int]
            Fist List

        Returns
        -------
        float
            Overlap coefficient between the two lists, value between 0 and 1
        """
        assert len(a) > 0 and len(b) > 0, "Lists must be not empty"
        # Convert lists to sets
        a_set = set(a)
        b_set = set(b)
        # Compute the intersection
        intersection = a_set.intersection(b_set)
        return len(intersection) / min(len(a_set), len(b_set))

    @staticmethod
    def sorensen_similarity(a: List[int], b: List[int]) -> float:
        """
        Compute the Sorensen similarity between two lists, A and B:
            S(A,B) = 2 * |A ∩ B| / (|A| + |B|)

        Parameters
        ----------
        a : List[int]
            First List
        b : List[int]
            Second List

        Returns
        -------
        float
            Sorensen similarity between the two lists, between 0 and 1
        """
        assert len(a) > 0 and len(b) > 0, "Lists must be not empty"
        # Convert lists to sets
        a_set = set(a)
        b_set = set(b)
        # Compute the intersection
        intersection = a_set.intersection(b_set)
        return 2 * len(intersection) / (len(a_set) + len(b_set))


class GraphSimilarity():
    """Class to compute the similarity between two graphs"""
    def __init__(self, function_name: str) -> None:
        """
        Initialize the GraphSimilarity class

        Parameters
        ----------
        function_name : str
            Name of the similarity function to use 
        """
        self.function_name = function_name
    
    def select_similarity_function(self) -> Callable:
        """
        Select the similarity function to use

        Returns
        -------
        Callable
            Similarity function to use
        """
        if self.function_name == SimilarityFunctionsNames.GED.value:
            return self.graph_edit_distance
        elif self.function_name == SimilarityFunctionsNames.JAC_1.value:
            return self.jaccard_similarity_1
        elif self.function_name == SimilarityFunctionsNames.JAC_2.value:
            return self.jaccard_similarity_2
        else:
            raise Exception("Similarity function not found")
    
    def graph_edit_distance(self, g: nx.Graph, h: nx.Graph) -> float:
        """
        Compute the graph edit distance between two graphs, then normalize it
        using a null graph:
            GED(G1,G2)/[GED(G1,G0) + GED(G2,G0)]  with G0 = null graph

        Parameters
        ----------
        g : nx.Graph
            First graph
        h : nx.Graph
            Second graph

        Returns
        -------
        graph_distance : float
            Graph edit distance between the two graphs normalized
        """
        # Slow, but precise
        # graph_distance = nx.graph_edit_distance(self.graph, self.old_graph)

        # Faster approximation of the graph edit distance
        graph_distance = next(nx.optimize_graph_edit_distance(g, h))
        # Normalize
        g_dist_1 = next(nx.optimize_graph_edit_distance(g, nx.null_graph()))
        g_dist_2 = next(nx.optimize_graph_edit_distance(h, nx.null_graph()))
        graph_distance /= (g_dist_1 + g_dist_2)
        return graph_distance
    
    def jaccard_similarity_1(self, g: nx.Graph, h: nx.Graph) -> float:
        """
        Compute the Jaccard Similarity between two graphs
        J(G, H) = (∑_{i,j} |A_{ij}^G - A_{i,j}^H|) / (∑_{i,j} max(A_{i,j)^G, A_{i,j}^H))
        
        Parameters
        ----------
        g : nx.Graph
            First graph
        h : nx.Graph
            Second graph
        
        Returns
        -------
        jaccard_sim : float
            Jaccard Similarity between the two graphs, between 0 and 1,
            where 0 means the two graphs are identical and 1 means they are
            completely different
        """
        # Get adjacency matrices
        g_matrix = nx.to_numpy_array(g)
        h_matrix = nx.to_numpy_array(h)
        # Ensure G and H have the same shape
        if g_matrix.shape != h_matrix.shape:
            raise ValueError("Input matrices must have the same shape.")
        # Calculate the numerator (sum of absolute differences)
        numerator = np.sum(np.abs(g_matrix - h_matrix))
        # Calculate the denominator (sum of element-wise maximum values)
        denominator = np.sum(np.maximum(g_matrix, h_matrix))
        # Calculate the Jaccard similarity
        jaccard_sim = numerator / denominator
        return jaccard_sim
    
    def jaccard_similarity_2(self, g: nx.Graph, h: nx.Graph) -> float:
        """
        Compute the Jaccard Similarity between two graphs, second version

        Parameters
        ----------
        g : nx.Graph
            First graph
        h : nx.Graph
            Second graph

        Returns
        -------
        float
            jaccard similarity between the two graphs
        """
        g = g.edges()
        h = h.edges()
        i = set(g).intersection(h)
        j = round(len(i) / (len(g) + len(h) - len(i)), 3)
        # Normalize to have 0 if the graphs are identical and 1 if they are 
        # completely different
        return 1-j

if __name__ == "__main__":
    # Test the the graph jaccard 1 similarity
    g = nx.Graph()
    g.add_edges_from([(1,2), (2,3), (3,4), (4,5), (5,1)])
    h = nx.Graph()
    h.add_edges_from([(1,2), (2,3), (3,4), (4,5), (5,1)])
    
    similarity_f = GraphSimilarity(SimilarityFunctionsNames.JAC_1.value).select_similarity_function()
    sim = similarity_f(g, h)
    print(sim)
    
    # generate two graph completely different
    g = nx.Graph()
    g.add_edges_from([(1,2), (2,3), (3,4), (4,5), (5,1)])
    h = nx.Graph()
    h.add_edges_from([(1,3), (2,4), (1,4), (2,5), (5,3)])
    
    similarity_f = GraphSimilarity(SimilarityFunctionsNames.JAC_1.value).select_similarity_function()
    sim = similarity_f(g, h)
    print(sim)
    