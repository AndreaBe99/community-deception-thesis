from enum import Enum
from typing import List, Tuple
from statistics import mean
# from src.environment.graph_env import GraphEnvironment

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
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
    
    # ! Trained model path for testing (change the following line to change the model)
    MODEL_PATH = "lfr_benchmark_node-200/tau-0.99_beta-30/greedy/lr-0.0001/gamma-0.7/lambda-0.1/alpha-0.7/model.pth"
    # MODEL_PATH = "kar/greedy/lr-0.0001/gamma-0.7/lambda-0.1/alpha-0.7/model.pth"
    TRAINED_MODEL = LOG_DIR + MODEL_PATH
    
    # Dataset file paths
    KAR = DATASETS_DIR + '/kar.mtx'
    # KAR = DATASETS_DIR + '/kar.gml'
    DOL = DATASETS_DIR + '/dol.mtx'
    # DOL = DATASETS_DIR + '/dol.gml'
    MAD = DATASETS_DIR + '/mad.mtx'
    # MAD = DATASETS_DIR + '/mad.gml'
    LESM = DATASETS_DIR + '/lesm.mtx'
    # LESM = DATASETS_DIR + '/lesm.gml'
    POLB = DATASETS_DIR + '/polb.mtx'
    # POLB = DATASETS_DIR + '/polb.gml'
    WORDS = DATASETS_DIR + '/words.mtx'
    # WORDS = DATASETS_DIR + '/words.gml'
    # NETS = DATASETS_DIR + '/nets.mtx'
    NETS = DATASETS_DIR + '/nets.gml'
    ERDOS = DATASETS_DIR + '/erdos.mtx'
    # ERDOS = DATASETS_DIR + '/erdos.gml'
    POW = DATASETS_DIR + '/pow.mtx'
    # POW = DATASETS_DIR + '/pow.gml'
    FB_75 = DATASETS_DIR + '/fb-75.mtx'
    # FB_75 = DATASETS_DIR + '/fb-75.gml'
    
    # The following datasets are too big, scipy cannot load them
    # DBLP = DATASETS_DIR + '/dblp.mtx'
    # ASTR = DATASETS_DIR + '/astr.mtx'
    # AMZ = DATASETS_DIR + '/amz.mtx'
    # YOU = DATASETS_DIR + '/you.mtx'
    # ORK = DATASETS_DIR + '/ork.mtx'


class DetectionAlgorithmsNames(Enum):
    """
    Enum class for the detection algorithms
    """
    LOUV = "louvain"            # ! NOT WORKING with: NETS (each community is a single node)
    WALK = "walktrap"           # ! NOT WORKING with: NETS (bipartite graphs)
    GRE = "greedy"              # ! NOT WORKING with: NETS (division by 0)
    INF = "infomap"             # ! NOT WORKING with: NETS (each community is a single node)
    LAB = "label_propagation"
    EIG = "eigenvector"         # ! NOT WORKING with: NETS (bipartite graphs)
    BTW = "edge_betweenness"
    SPIN = "spinglass"
    OPT = "optimal"
    SCD = "scalable_community_detection"


class SimilarityFunctionsNames(Enum):
    """
    Enum class for the similarity functions
    """
    # Community similarity functions
    JAC = "jaccard"
    OVE = "overlap"
    SOR = "sorensen"
    # Graph similarity functions
    GED = "ged"  # Graph edit distance
    JAC_1 = "jaccard_1"
    JAC_2 = "jaccard_2"


class HyperParams(Enum):
    """Hyperparameters for the Environment"""
    # ! REAL GRAPH Graph path (change the following line to change the graph)
    GRAPH_NAME = FilePaths.WORDS.value
    # ! Define the detection algorithm to use (change the following line to change the algorithm)
    DETECTION_ALG_NAME = DetectionAlgorithmsNames.WALK.value
    # Multiplier for the rewiring action number, i.e. (mean_degree * BETA)
    BETA = 3
    # ! Strength of the deception constraint, value between 0 (hard) and 1 (soft) 
    TAU = 0.5
    # ° Hyperparameters  Testing ° #
    # ! Weight to balance the penalty in the reward
    # The higher its value the more importance the penalty will have
    LAMBDA = [0.1] # [0.01, 0.1, 1]
    # ! Weight to balance the two metrics in the definition of the penalty
    # The higher its value the more importance the distance between communities 
    # will have, compared with the distance between graphs
    ALPHA = [0.7] # [0.3, 0.5, 0.7]
    # Multiplier for the number of maximum steps allowed
    MAX_STEPS_MUL = 2
    
    """ Graph Encoder Parameters """""
    EMBEDDING_DIM = 128 # 256

    """ Agent Parameters"""
    # Chaneg target community and target node with a probability of EPSILON
    EPSILON = 25    # Between 0 and 100
    # Networl Architecture
    HIDDEN_SIZE_1 = 64
    HIDDEN_SIZE_2 = 64
    # Rehularization parameters
    DROPOUT = 0.2
    WEIGHT_DECAY = 1e-3
    # Hyperparameters for the ActorCritic
    EPS_CLIP = np.finfo(np.float32).eps.item()  # 0.2
    BEST_REWARD = -np.inf
    # ° Hyperparameters  Testing ° #
    # ! Learning rate, it controls how fast the network learns
    LR = [1e-4] # [1e-7, 1e-4, 1e-1]
    # ! Discount factor:
    # - 0: only the reward on the next step is important
    # - 1: a reward in the future is as important as a reward on the next step
    GAMMA = [0.7] # [0.9, 0.95]
    
    """ Training Parameters """
    # Number of episodes to collect experience
    MAX_EPISODES = 10000
    # Dictonary for logging
    LOG_DICT = {
        # List of rewards per episode
        'train_reward_list': [],
        # Avg reward per episode, with the last value multiplied per 10 if the 
        # goal is reached
        'train_reward_mul': [],
        # Total reward per episode
        'train_reward': [],
        # Number of steps per episode
        'train_steps': [],
        # Average reward per episode
        'train_avg_reward': [],
        # Average Actor loss per episode
        'a_loss': [],
        # Average Critic loss per episode
        'v_loss': [],
        # set max number of training episodes
        'train_episodes': MAX_EPISODES,
    }
    
    """Evaluation Parameters"""
    # ! Change the following parameters according to the hyperparameters to test
    STEPS_EVAL = 100
    LR_EVAL = 0.0001    # LR[0]
    GAMMA_EVAL = 0.7    # GAMMA[0]
    LAMBDA_EVAL = 0.1   # LAMBDA[0]
    ALPHA_EVAL = 0.7    # ALPHA[0]
    
    """Graph Generation Parameters"""
    # ! Change the following parameters to modify the graph
    # Number of nodes
    N_NODE = 500
    # Power law exponent for the degree distribution of the created graph.
    TAU1 = 2
    # Power law exponent for the community size distribution in the created graph.
    TAU2 = 1.1
    # Fraction of inter-community edges incident to each node.
    MU = 0.1

    # Desired average degree of nodes in the created graph.
    AVERAGE_DEGREE = int(0.05 * N_NODE)  # 20
    # Minimum degree of nodes in the created graph
    MIN_DEGREE = None  # 30
    # Maximum degree of nodes in the created graph
    MAX_DEGREE = int(0.19 * N_NODE)

    # Minimum size of communities in the graph.
    MIN_COMMUNITY = int(0.05 * N_NODE)
    # Maximum size of communities in the graph.
    MAX_COMMUNITY = int(0.2 * N_NODE)

    # Maximum number of iterations to try to create the community sizes, degree distribution, and community affiliations.
    MAX_ITERS = 5000
    # Seed for the random number generator.
    SEED = 10


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
        # try:
        # Check if the graph file is in the .mtx format or .gml
        if file_path.endswith(".mtx"):
            graph_matrix = scipy.io.mmread(file_path)
            graph = nx.Graph(graph_matrix)
        elif file_path.endswith(".gml"):
            graph = nx.read_gml(file_path, label='id')
        else:
            raise ValueError("File format not supported")

        for node in graph.nodes:
            # graph.nodes[node]['name'] = node
            graph.nodes[node]['num_neighbors'] = len(
                list(graph.neighbors(node)))
        return graph
        # except Exception as exception:
        #     print("Error: ", exception)
        #     return None
    
    @staticmethod
    def generate_lfr_benchmark_graph(
        n: int=HyperParams.N_NODE.value,
        tau1: float=HyperParams.TAU1.value,
        tau2: float=HyperParams.TAU2.value,
        mu: float=HyperParams.MU.value,   
        average_degree: int = HyperParams.AVERAGE_DEGREE.value,
        min_degree: int=HyperParams.MIN_DEGREE.value,
        max_degree: int=HyperParams.MAX_DEGREE.value,
        min_community: int=HyperParams.MIN_COMMUNITY.value,
        max_community: int=HyperParams.MAX_COMMUNITY.value,
        max_iters: int=HyperParams.MAX_ITERS.value,
        seed: int=HyperParams.SEED.value)->Tuple[nx.Graph, str]:
        """
        Generate a LFR benchmark graph for community detection algorithms.

        Parameters
        ----------
        n : int, optional
            Number of nodes, by default 500
        tau1 : float, optional
            _description_, by default 3
        tau2 : float, optional
            _description_
        mu : float, optional
            Mixing parameter, by default 0.1
        average_degree : int, optional
            Average degree of the nodes, by default 20
        min_degree : int, optional
            Minimum degree of the nodes, by default 20
        max_degree : int, optional
            Maximum degree of the nodes, by default 50
        min_community : int, optional
            Minimum number of communities, by default 10
        max_community : int, optional
            Maximum number of communities, by default 50
        max_iters : int, optional
            Maximum number of iterations, by default 5000
        seed : int, optional
            Seed for the random number generator, by default 10

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
            min_degree=min_degree,
            max_degree=max_degree,
            min_community=min_community,
            max_community=max_community,
            max_iters=max_iters,
            seed=seed)
        # Save the graph in a .mtx file
        file_path = FilePaths.DATASETS_DIR.value + f"/lfr_benchmark_node-{n}"
        # ! FOR KAGGLE NOTEBOOK
        # file_path = f"/kaggle/working/lfr_benchmark_node-{n}.mtx"
        # Write .gml file
        nx.write_gml(graph, f"{file_path}.gml")
        # Write .mtx file
        nx.write_edgelist(graph, f"{file_path}.mtx", data=False)
        
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
        window_size: int=int(HyperParams.MAX_EPISODES.value/100)):
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
        def plot_seaborn(
                df: pd.DataFrame,
                path: str,
                env_name: str,
                detection_algorithm: str,
                labels: Tuple[str, str],
                colors: Tuple[str, str]) -> None:
            sns.set_style("darkgrid")
            sns.lineplot(data=df, x="Episode", y=labels[0], color=colors[0])
            sns.lineplot(data=df, x="Episode", y=labels[1], color=colors[1],
                        estimator="mean", errorbar=None)
            plt.title(
                f"Training on {env_name} graph with {detection_algorithm} algorithm")
            plt.xlabel("Episode")
            plt.ylabel(labels[0])
            plt.savefig(path)
            plt.clf()
        
        if window_size < 1:
            window_size = 1
        df = pd.DataFrame({
            "Episode": range(len(log["train_avg_reward"])),
            "Avg Reward": log["train_avg_reward"],
            "Steps per Epoch": log["train_steps"],
            "Goal Reward": log["train_reward_mul"],
            "Goal Reached": [1/log["train_steps"][i] if log["train_reward_list"][i][-1]
                > 1 else 0 for i in range(len(log["train_steps"]))],
        })
        df["Rolling_Avg_Reward"] = df["Avg Reward"].rolling(window_size).mean()
        df["Rolling_Steps"] = df["Steps per Epoch"].rolling(window_size).mean()
        df["Rolling_Goal_Reward"] = df["Goal Reward"].rolling(window_size).mean()
        df["Rolling_Goal_Reached"] = df["Goal Reached"].rolling(window_size).mean()
        plot_seaborn(
            df,
            file_path+"/training_reward.png",
            env_name,
            detection_algorithm,
            ("Avg Reward", "Rolling_Avg_Reward"),
            ("lightsteelblue", "darkblue"),
        )
        plot_seaborn(
            df,
            file_path+"/training_steps.png",
            env_name,
            detection_algorithm,
            ("Steps per Epoch", "Rolling_Steps"),
            ("thistle", "purple"),
        )
        plot_seaborn(
            df,
            file_path+"/training_goal_reward.png",
            env_name,
            detection_algorithm,
            ("Goal Reward", "Rolling_Goal_Reward"),
            ("darkgray", "black"),
        )
        plot_seaborn(
            df,
            file_path+"/training_goal_reached.png",
            env_name,
            detection_algorithm,
            ("Goal Reached", "Rolling_Goal_Reached"),
            ("darkgray", "black"),
        )

        df = pd.DataFrame({
            "Episode": range(len(log["a_loss"])),
            "Actor Loss": log["a_loss"],
            "Critic Loss": log["v_loss"],
        })
        df["Rolling_Actor_Loss"] = df["Actor Loss"].rolling(window_size).mean()
        df["Rolling_Critic_Loss"] = df["Critic Loss"].rolling(window_size).mean()
        plot_seaborn(
            df,
            file_path+"/training_a_loss.png",
            env_name,
            detection_algorithm,
            ("Actor Loss", "Rolling_Actor_Loss"),
            ("palegreen", "darkgreen"),
        )
        plot_seaborn(
            df,
            file_path+"/training_v_loss.png",
            env_name,
            detection_algorithm,
            ("Critic Loss", "Rolling_Critic_Loss"),
            ("lightcoral", "darkred"),
        )

        
    ############################################################################
    #                               EVALUATION                                 #
    ############################################################################   
    @staticmethod
    def save_test(
        log: dict, 
        files_path: str, 
        log_name: str, 
        algs: List[str],
        metrics: List[str]):
        """Save and Plot the testing results

        Parameters
        ----------
        log : dict
            Dictionary containing the training logs
        files_path : str
            Path to save the plot
        log_name : str
            Name of the log file
        algs : List[str]
            List of algorithms names to evaluate
        metrics : List[str]
            List of metrics to evaluate
        """
        file_name = f"{files_path}/{log_name}.json"
        # Save json file
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=4)
            
        for metric in metrics:
            # Create a DataFrame with the mean values of each algorithm for the metric
            df = pd.DataFrame({
                "Algorithm": algs,
                metric.capitalize(): [mean(log[alg][metric]) for alg in algs]
            })
            
            # Convert the goal column to percentage
            if metric == "goal":
                df[metric.capitalize()] = df[metric.capitalize()] * 100

            sns.barplot(data=df,
                        x="Algorithm",
                        y=metric.capitalize(),
                        palette=sns.color_palette("Set1"))
            plt.title(f"Evaluation on {log['env']['dataset']} graph with {log['env']['detection_alg']} algorithm")
            plt.xlabel("Algorithm")
            if metric == "goal":
                plt.ylabel(f"{metric.capitalize()} reached %")
            elif metric == "time":
                plt.ylabel(f"{metric.capitalize()} (s)")
            else:
                plt.ylabel(metric.capitalize())
            plt.savefig(f"{files_path}/{log_name}_{metric}.png")
            plt.clf()

