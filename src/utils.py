"""Module to store utility functions and constants"""
from enum import Enum
import os
import scipy
import networkx as nx

class Utils:
    """Class to store utility functions"""
    
    @staticmethod
    def get_device_placement():
        """Get device placement, CPU or GPU"""
        return os.getenv("RELNET_DEVICE_PLACEMENT", "CPU")
    
    @staticmethod
    def import_mtx_graph(file_path: str)->nx.Graph:
        """
        Import a graph from a .mtx file

        Parameters
        ----------
        file_path : str
            File path of the .mtx file

        Returns
        -------
        nx.Graph
            Graph imported from the .mtx file
        """
        try:
            graph_matrix = scipy.io.mmread(file_path)
            graph = nx.Graph(graph_matrix)
            return graph
        except Exception as exception:
            print("Error: ", exception)
            return None

class FilePaths(Enum):
    """Class to store file paths for data and models"""
    MODELS_DIR = 'models'
    CHKPT_DIR  = 'tmp/rl'
    LOG_DIR    = 'logs'
    DATASETS_DIR = 'dataset/data'
    KARATE_PATH = DATASETS_DIR + '/kar.mtx'

class HyperParams(Enum):
    """
    Hyperparameters for the model.
    """
    WEIGHT = 0.8
    G_IN_SIZE = 50
    HIDDEN1 = 300
    HIDDEN2 = 300
    G_HIDDEN_SIZE = 50
    G_EMBEDDING_SIZE = 50
    HIDDEN_SIZE = 300
    ACTION_STD = 0.5
    EPS_CLIP = 0.2
    LR = 0.0003
    GAMMA = 0.99
    K_EPOCHS = 10
    LOG_INTERVAL = 20
    MAX_EPISODES = 15000
    MAX_TIMESTEPS = 100
    UPDATE_TIMESTEP = 100
    SOLVED_REWARD = 0.6 * MAX_TIMESTEPS
    
    """Hyperparameters for the Environment"""
    BETA = 30
    DEBUG = False

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
