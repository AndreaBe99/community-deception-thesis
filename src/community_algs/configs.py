from enum import Enum

class DetectionAlgorithms(Enum):
    """
    Enum class for the detection algorithms
    """
    louv = "louvain"
    walk = "walktrap"
    gre  = "greedy"
    inf  = "infomap"
    lab  = "label_propagation"
    eig  = "eigenvector"
    btw  = "edge_betweenness"
    spin = "spinglass"
    opt  = "optimal"
    scd  = "scalable_community_detection"