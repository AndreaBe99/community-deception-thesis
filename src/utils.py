"""Module to store utility functions and constants"""
from enum import Enum
import os

class Utils:
    """Class to store utility functions"""
    def get_device_placement(self):
        """Get device placement, CPU or GPU"""
        return os.getenv("RELNET_DEVICE_PLACEMENT", "CPU")

class FilePaths(Enum):
    """Class to store file paths for data and models"""
    MODELS_DIR = 'models'
    CHKPT_DIR  = 'tmp/rl'

class HyperParams(Enum):
    """
    Hyperparameters for the model.
    """
    WEIGHT = 0.5

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
