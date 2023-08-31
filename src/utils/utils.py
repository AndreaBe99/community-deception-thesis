"""Module to store utility functions and constants"""
from enum import Enum
from typing import List
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
    LR = 0.0003
    GAMMA = 0.99

    """ Training Parameters """
    # Number of episodes to collect experience
    MAX_EPISODES = 200          # 15000
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
    BETA = 30  # Numeber of possible action with BETA=30, is 30% of the edges
    DEBUG = False
    WEIGHT = 0.8


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
    def plot_avg_reward(
            log_reward: List[float],
            log_timesteps: List[int],
            log_loss: List[float],
            env_name: str,
            detection_algorithm: str,
            file_path: str = FilePaths.LOG_DIR.value):
        """
        Plot the average reward and the time steps of the episodes in the same
        plot, using matplotlib, where the average reward is the blue line and
        the episode length are the orange line, and the loss in a different
        image.

        Parameters
        ----------
        log_reward : List[float]
            Average reward for each episode
        log_timesteps : List[int]
            Time steps for each episode
        log_loss : List[float]
            Loss for each episode
        env_name : str
            Environment name
        detection_algorithm : str
            Detection algorithm used
        file_path : str, optional
            Path to save the plot, by default "src/logs/"
        """
        _, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Reward', color=color)
        ax1.plot(log_reward, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Time Steps', color=color)
        ax2.plot(log_timesteps, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f"Training {env_name}")
        plt.savefig(
            f"{file_path}{env_name}_{detection_algorithm}_training_reward.png")
        plt.show()

        _, ax1 = plt.subplots()
        multiplier = int(HyperParams.UPDATE_TIMESTEP.value/HyperParams.MAX_TIMESTEPS.value)
        color = 'tab:green'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot([int(i * multiplier) for i in range(len(log_loss))], log_loss, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        plt.title(f"Training {env_name}")
        plt.savefig(
            f"{file_path}{env_name}_{detection_algorithm}_training_loss.png")
        plt.show()

    @staticmethod
    def write_results_to_json(
            log_reward: List[float],
            log_length: List[int],
            log_loss: List[float],
            hyperparameters: dict,
            env_name: str,
            detection_algorithm: str,
            file_path: str = FilePaths.LOG_DIR.value):
        """
        Write the episodes_avg_reward, episode_length, and hyperparameters to a JSON file.

        Parameters
        ----------
        log_reward : List[float]
            List of average rewards for each episode
        log_length : List[int]
            List of episode lengths
        log_loss : List[float]
            List of losses for each episode
        hyperparameters : dict
            Dictionary of hyperparameters used in the training process
        env_name : str
            Environment name
        detection_algorithm : str
            Detection algorithm used
        file_path : str
            Path to the output JSON file
        """
        data = {
            "episodes_avg_reward": log_reward,
            "episode_avg_length": log_length,
            "episode_avg_loss": log_loss,
            "hyperparameters": hyperparameters
        }
        file_name = f"{file_path}{env_name}_{detection_algorithm}_results.json"
        with open(file_name, "w") as f:
            json.dump(data, f, indent=4)
