"""Module to store utility functions and constants"""
from enum import Enum
from typing import List
import matplotlib.pyplot as plt
import networkx as nx
import scipy
import json
import os

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
            for node in graph.nodes:
                # graph.nodes[node]['name'] = node
                graph.nodes[node]['num_neighbors'] = len(
                    list(graph.neighbors(node)))
            return graph
        except Exception as exception:
            print("Error: ", exception)
            return None
    
    @staticmethod
    def plot_avg_reward(
        reward: List[float], 
        episode_length: List[int], 
        env_name: str,
        file_path: str = "src/logs/"):
        """
        Plot the average reward and the time steps of the episodes in the same
        plot, using matplotlib, where the average reward is the blue line and
        the episode length are the orange line.

        Parameters
        ----------
        reward : List[float]
            Average reward for each episode
        episode_length : List[int]
            Time steps for each episode
        env_name : str
            Environment name
        file_path : str, optional
            Path to save the plot, by default "src/logs/"
        """
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Reward', color=color)
        ax1.plot(reward, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Time Steps', color=color)
        ax2.plot(episode_length, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title(f'Average Reward and Time Steps per Episode ({env_name})')
        file_name = f"{file_path}{env_name}_avg_reward.png"
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def write_results_to_json(
        episodes_avg_reward: List[float], 
        episode_length: List[int], 
        hyperparameters: dict,
        env_name: str,
        file_path: str = "src/logs/"):
        """
        Write the episodes_avg_reward, episode_length, and hyperparameters to a JSON file.

        Parameters
        ----------
        episodes_avg_reward : List[float]
            List of average rewards for each episode
        episode_length : List[int]
            List of episode lengths
        hyperparameters : dict
            Dictionary of hyperparameters used in the training process
                file_path : str
            Path to the output JSON file
        """
        data = {
            "episodes_avg_reward": episodes_avg_reward,
            "episode_length": episode_length,
            "hyperparameters": hyperparameters
        }
        file_name = f"{file_path}{env_name}_results.json"
        with open(file_name, "w") as f:
            json.dump(data, f, indent=4)

class FilePaths(Enum):
    """Class to store file paths for data and models"""
    # MODELS_DIR = 'models'
    # CHKPT_DIR  = 'tmp/rl'
    LOG_DIR    = 'src/logs/'
    DATASETS_DIR = 'dataset/data'
    KARATE_PATH = DATASETS_DIR + '/kar.mtx'
    DOLPHIN_PATH = DATASETS_DIR + '/dol.mtx'

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
    K_EPOCHS = 20
    LOG_INTERVAL = 20
    MAX_EPISODES = 200          # 15000
    MAX_TIMESTEPS = 10
    UPDATE_TIMESTEP = 100
    SOLVED_REWARD = 0.7
    
    # Save model after N episodes
    SAVE_MODEL = int(MAX_EPISODES / 10)
    
    # Use a random seed
    RANDOM_SEED = 42
    
    
    
    """Hyperparameters for the Environment"""
    BETA = 30 # Numeber of possible action with BETA=30, is 30% of the edges
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
