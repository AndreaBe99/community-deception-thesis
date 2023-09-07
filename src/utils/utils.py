"""Module to store utility functions and constants"""
from enum import Enum
from typing import List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import json
import os


class FilePaths(Enum):
    """Class to store file paths for data and models"""
    # ° Local
    DATASETS_DIR = 'dataset/data'
    LOG_DIR    = 'src/logs/'
    TEST_DIR = 'test/'
    # ° Kaggle
    # DATASETS_DIR = '/kaggle/input/network-community'
    # LOG_DIR = '/kaggle/working/logs/'
    # TEST_DIR = '/kaggle/working/test/'
    # ° Google Colab
    # DATASETS_DIR = "/content/drive/MyDrive/Sapienza/Tesi/Datasets"
    # LOG_DIR = "/content/drive/MyDrive/Sapienza/Tesi/Logs/"
    # TEST_DIR = "/content/drive/MyDrive/Sapienza/Tesi/Test/"
    
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
    """Hyperparameters for the Environment"""
    # Numeber of possible action with BETA=30, is 30% of the edges
    BETA = 10  
    # Weight to balance the reward
    WEIGHT = 0.1  # 0.001, 0.01, 0.1, 1, 10
    
    """ Graph Encoder Parameters """""
    STATE_DIM = 64
    # G_HIDDEN_SIZE_1 = 128
    # G_HIDDEN_SIZE_2 = 64
    # G_EMBEDDING_SIZE = 32

    """ Agent Parameters"""
    HIDDEN_SIZE_1 = 32
    HIDDEN_SIZE_2 = 32
    ACTION_DIM = 1      # We will return a  N*1 vector of actions, where N is the number of nodes
    # ACTION_STD = 0.5
    EPS_CLIP = np.finfo(np.float32).eps.item()  # 0.2
    LR = 0.0001
    GAMMA = 0.1 # 0.97
    BEST_REWARD = 0.7  # -np.inf

    """ Training Parameters """
    # Number of episodes to collect experience
    MAX_EPISODES = 1000  # 200 # 15000
    # Dictonary for logging
    LOG_DICT = {
        'train_reward': [],
        # Number of steps per episode
        'train_steps': [],
        # Average reward per step
        'train_avg_reward': [],
        # Average Actor loss per episode
        'a_loss': [],
        # Average Critic loss per episode
        'v_loss': [],
        # set max number of training episodes
        'train_episodes': MAX_EPISODES,
    }
    
    """Graph Generation Parameters"""
    N_NODE = 10000
    TAU1 = 3
    TAU2 = 1.5
    MU = 0.1             # TODO: Test also 0.3 and 0.6
    AVERAGE_DEGREE = 5
    MIN_COMMUNITY = 20
    SEED= 10

    """Old Training Parameters"""
    # Maximum number of time steps per episode
    # MAX_TIMESTEPS = 10  # ! Unused, I set it to the double of the edge budget
    # Update the policy after N timesteps
    # UPDATE_TIMESTEP = 100  # ! Unused, I set it to 10 times the edge budget
    # Update policy for K epochs
    # K_EPOCHS = 20
    # Print info about the model after N episodes
    # LOG_INTERVAL = 20
    # Exit if the average reward is greater than this value
    # SOLVED_REWARD = 0.7
    # Save model after N episodes
    # SAVE_MODEL = int(MAX_EPISODES / 10)
    # Use a random seed
    # RANDOM_SEED = 42


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
        n: int=HyperParams.N_NODE.value,
        tau1: float=HyperParams.TAU1.value,
        tau2: float=HyperParams.TAU2.value,
        mu: float=HyperParams.MU.value,              
        average_degree: float=HyperParams.AVERAGE_DEGREE.value, 
        min_community: int=HyperParams.MIN_COMMUNITY.value, 
        seed: int=HyperParams.SEED.value)->Tuple[nx.Graph, str]:
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
        # Delete community attribute from the nodes to handle PyG compatibility
        for node in graph.nodes:
            if 'community' in graph.nodes[node]:
                del graph.nodes[node]['community']
        for edge in graph.edges:
            graph.edges[edge]['weight'] = 1
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
        file_path: str,
        window_size: int=100):
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
        window_size : int, optional
            Size of the rolling window, by default 100
        """
        def plot_time_series(
            list_1: List[float],
            list_2: List[float],
            label_1: str,
            label_2: str,
            color_1: str,
            color_2: str,
            file_name: str):
            _, ax1 = plt.subplots()
            color = 'tab:'+color_1
            ax1.set_xlabel("Episode")
            ax1.set_ylabel(label_1, color=color)
            ax1.plot(list_1, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()
            color = 'tab:'+color_2
            ax2.set_ylabel(label_2, color=color)
            ax2.plot(list_2, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title(
                f"Training on {env_name} graph with {detection_algorithm} algorithm")
            plt.savefig(file_name)
            plt.show()
        
        def plot_rolling_window(
            list_1: List[float],
            list_2: List[float],
            label_1: str,
            label_2: str,
            file_name: str,
            window_size: int = 100):
            time_series_1 = np.array(list_1)
            time_series_2 = np.array(list_2)
            # Compute the rolling windows of the time series data using NumPy
            rolling_data_1 = np.convolve(time_series_1, np.ones(
                window_size) / window_size, mode='valid')
            rolling_data_2 = np.convolve(time_series_2, np.ones(
                window_size) / window_size, mode='valid')
            # Plot the rolling windows of the time series data using matplotlib
            plt.plot(rolling_data_1, label=label_1)
            plt.plot(rolling_data_2, label=label_2)
            plt.title("Rolling Window")
            plt.xlabel("Epochs")
            # plt.ylabel("Epochs")
            plt.legend()
            plt.savefig(file_name)
            plt.show()
        
        file_path = file_path+"/"+env_name+"_"+detection_algorithm
        plot_time_series(
            log['train_avg_reward'],
            log['train_steps'],
            'Avg Reward',
            'Steps per Epoch',
            'blue',
            'orange',
            file_path+"_training_reward.png",
        )
        plot_time_series(
            log["a_loss"],
            log["v_loss"],
            'Actor Loss',
            'Critic Loss',
            'green',
            'red',
            file_path+"_training_loss.png",
        )

        # Same plot with rolling window
        plot_rolling_window(
            log['train_reward'], 
            log['train_steps'], 
            'Avg Reward', 
            'Steps per Epoch',
            file_path+"_rolling_training_reward.png"
        )
        plot_rolling_window(
            log["a_loss"],
            log["v_loss"],
            'Actor Loss',
            'Critic Loss',
            file_path+"_rolling_training_loss.png"
        )
    
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
