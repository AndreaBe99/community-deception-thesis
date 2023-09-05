"""Module to store utility functions and constants"""
from enum import Enum
from typing import List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import scipy
import json
import os

"""Module to store utility functions and constants"""


class FilePaths(Enum):
    """Class to store file paths for data and models"""
    # ° Local
    DATASETS_DIR = 'dataset/data'
    LOG_DIR    = 'src/logs/'
    TEST_DIR = 'test/'
    # ° Kaggle
    # DATASETS_DIR = '/kaggle/input/network-community'
    # LOG_DIR    = 'logs/'
    # ° Google Colab
    # DATASETS_DIR = "/content/drive/MyDrive/Sapienza/Tesi/Datasets"
    # LOG_DIR = "/content/drive/MyDrive/Sapienza/Tesi/Logs/"
    
    # Dataset file paths
    KAR = DATASETS_DIR + '/kar.mtx'
    DOL = DATASETS_DIR + '/dol.mtx'
    MAD = DATASETS_DIR + '/mad.mtx'
    LESM = DATASETS_DIR + '/lesm.mtx'
    POLB = DATASETS_DIR + '/polb.mtx'
    WORDS = DATASETS_DIR + '/words.mtx'
    ERDOS = DATASETS_DIR + '/erdos.mtx'
    POW = DATASETS_DIR + '/pow.mtx'
    FB_75 = DATASETS_DIR + '/fb-75.mtx'
    DBLP = DATASETS_DIR + '/dblp.mtx'
    ASTR = DATASETS_DIR + '/astr.mtx'
    AMZ = DATASETS_DIR + '/amz.mtx'
    YOU = DATASETS_DIR + '/you.mtx'
    ORK = DATASETS_DIR + '/ork.mtx'


class HyperParams(Enum):
    """ Hyperparameters for the model."""

    """ Graph Encoder Parameters """""
    G_IN_SIZE = 64
    G_HIDDEN_SIZE_1 = 128
    G_HIDDEN_SIZE_2 = 64
    G_EMBEDDING_SIZE = 32

    """ Agent Parameters"""
    HIDDEN_SIZE_1 = 64
    HIDDEN_SIZE_2 = 128
    ACTION_STD = 0.5
    EPS_CLIP = 0.2
    LR = 1e-3
    GAMMA = 0.99

    """ Training Parameters """
    # Number of episodes to collect experience
    MAX_EPISODES = 1000 # 200 # 15000
    # Maximum number of time steps per episode
    MAX_TIMESTEPS = 10  # ! Unused, I set it to the double of the edge budget
    # Update the policy after N timesteps
    UPDATE_TIMESTEP = 100  # ! Unused, I set it to 10 times the edge budget
    # Update policy for K epochs
    K_EPOCHS = 20
    # Print info about the model after N episodes
    LOG_INTERVAL = 20
    # Exit if the average reward is greater than this value
    SOLVED_REWARD = 0.7
    # Save model after N episodes
    SAVE_MODEL = int(MAX_EPISODES / 10)
    # Use a random seed
    RANDOM_SEED = 42

    """Hyperparameters for the Environment"""
    BETA = 10  # Numeber of possible action with BETA=30, is 30% of the edges
    DEBUG = False
    # Weight to balance the reward between NMI and Deception Score
    WEIGHT = 0.7


class DetectionAlgorithms(Enum):
    """
    Enum class for the detection algorithms
    """
    LOUV = "louvain"
    WALK = "walktrap"
    GRE = "greedy"
    INF = "infomap"
    LAB = "label_propagation"
    EIG = "eigenvector"
    BTW = "edge_betweenness"
    SPIN = "spinglass"
    OPT = "optimal"
    SCD = "scalable_community_detection"


class Utils:
    """Class to store utility functions"""

    @staticmethod
    def get_device_placement():
        """Get device placement, CPU or GPU"""
        return os.getenv("RELNET_DEVICE_PLACEMENT", "CPU")

    @staticmethod
    def import_mtx_graph(file_path: str) -> nx.Graph:
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
            for node in graph.nodes:
                # graph.nodes[node]['name'] = node
                graph.nodes[node]['num_neighbors'] = len(
                    list(graph.neighbors(node)))
            return graph
        except Exception as exception:
            print("Error: ", exception)
            return None
    
    @staticmethod
    def generate_lfr_benchmark_graph(
        n: int=10000,
        tau1: float=3,
        tau2: float=1.5,
        mu: float=0.1,              # TODO: Test also 0.3 and 0.6
        average_degree: float=5, 
        min_community: int=20, 
        seed: int=10)->Tuple[nx.Graph, str]:
        """
        Generate a LFR benchmark graph for community detection algorithms.

        Parameters
        ----------
        n : int, optional
            _description_, by default 250
        tau1 : float, optional
            _description_, by default 3
        tau2 : float, optional
            _description_, by default 1.5
        mu : float, optional
            _description_, by default 0.1
        average_degree : float, optional
            _description_, by default 5
        min_community : int, optional
            _description_, by default 20
        seed : int, optional
            _description_, by default 10

        Returns
        -------
        nx.Graph
            Synthetic graph generated with the LFR benchmark
        file_path : str
            Path to the file where the graph is saved
        """
        graph = nx.generators.community.LFR_benchmark_graph(
            n=n,
            tau1=tau1,
            tau2=tau2,
            mu=mu,
            average_degree=average_degree,
            min_community=min_community,
            seed=seed)
        file_path = FilePaths.DATASETS_DIR.value + f"/lfr_benchmark_mu-{mu}.mtx"
        nx.write_edgelist(graph, file_path, data=False)
        return graph, file_path
        
    @staticmethod
    def check_dir(path: str):
        """
        Check if the directory exists, if not create it.

        Parameters
        ----------
        path : str
            Path to the directory
        """
        if not os.path.exists(path):
            os.makedirs(path)
    
    
    @staticmethod
    def plot_training(
        log: dict, 
        env_name: str, 
        detection_algorithm: str,
        file_path: str):
        """Plot the training results

        Parameters
        ----------
        log : dict
            Dictionary containing the training logs
        env_name : str
            Name of the environment
        detection_algorithm : str
            Name of the detection algorithm
        file_path : str
            Path to save the plot
        """
        # Plot the average reward and the time steps of the episodes in the same
        # plot, using matplotlib, where the average reward is the blue line and
        # the episode length are the orange line.
        _, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Reward', color=color)
        # ° Plot Lines
        ax1.plot(log["train_avg_reward"], color=color)
        # ° Plot Points
        # ax1.scatter(range(len(log["train_avg_reward"])), log["train_avg_reward"], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Time Steps', color=color)
        # ° Plot Lines
        ax2.plot(log["train_steps"], color=color)
        # ° Plot Points
        # ax2.scatter(range(len(log["train_steps"])), log["train_steps"], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f"Training on {env_name} graph with {detection_algorithm} algorithm")
        plt.savefig(
            f"{file_path}/{env_name}_{detection_algorithm}_training_reward.png")
        plt.show()
        
        # Plot the Actor and Critic loss in the same plot, using matplotlib
        # with the Actor loss in green and the Critic loss in red.
        _, ax1 = plt.subplots()
        color = 'tab:green'
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Actor Loss', color=color)
        # ° Plot Lines
        ax1.plot(log["a_loss"], color=color)
        # ° Plot Points
        # ax1.scatter(range(len(log["a_loss"])), log["a_loss"], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Critic Loss', color=color)
        # ° Plot Lines
        ax2.plot(log["v_loss"], color=color)
        # ° Plot Points
        # ax2.scatter(range(len(log["v_loss"])), log["v_loss"], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f"Training on {env_name} graph with {detection_algorithm} algorithm")
        plt.savefig(
            f"{file_path}/{env_name}_{detection_algorithm}_training_loss.png")
        plt.show()
    
    @staticmethod
    def save_training(
            log: dict,
            env_name: str,
            detection_algorithm: str,
            file_path: str):
        """Plot the training results

        Parameters
        ----------
        log : dict
            Dictionary containing the training logs
        env_name : str
            Name of the environment
        detection_algorithm : str
            Name of the detection algorithm
        file_path : str
            Path to save the plot
        """
        file_name = f"{file_path}/{env_name}_{detection_algorithm}_results.json"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=4)
