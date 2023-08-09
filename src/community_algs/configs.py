"""Module for the configuration of the community detection algorithms"""
from enum import Enum

class DetectionAlgorithms(Enum):
    """
    Enum class for the detection algorithms
    """
    LOUV = "louvain"
    WALK = "walktrap"
    GRE  = "greedy"
    INF  = "infomap"
    LAB  = "label_propagation"
    EIG  = "eigenvector"
    BTW  = "edge_betweenness"
    SPIN = "spinglass"
    OPT  = "optimal"
    SCD  = "scalable_community_detection"