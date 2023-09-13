from src.utils.utils import HyperParams, Utils, FilePaths, DetectionAlgorithms
from src.environment.graph_env import GraphEnvironment
from src.agent.agent import Agent

from src.utils.test import test
import argparse


def get_args():
    """
    Function for handling command line arguments

    Returns
    -------
    args : argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='PyTorch A2C')
    # Mode: train or test
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test')
    # Argument parsing
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print("*"*20, "Setup Information", "*"*20)

    # ° ------ Graph Setup ------ ° #
    # ! REAL GRAPH Graph path (change the following line to change the graph)
    graph_path = FilePaths.DOL.value
    # Load the graph from the dataset folder
    graph = Utils.import_mtx_graph(graph_path)
    # ! SYNTHETIC GRAPH Graph path (change the following line to change the graph)
    # graph, graph_path = Utils.generate_lfr_benchmark_graph()
    # Set the environment name as the graph name
    env_name = graph_path.split("/")[-1].split(".")[0]
    # Print the number of nodes and edges
    print("* Graph Name:", env_name)
    print("*", graph)

    # ° --- Environment Setup --- ° #
    # ! Define the detection algorithm to use (change the following line to change the algorithm)
    detection_alg = DetectionAlgorithms.INF.value
    # Define the environment
    env = GraphEnvironment(
        graph=graph,
        env_name=env_name,
        community_detection_algorithm=detection_alg)

    # Define the agent
    agent = Agent(env=env)
    # ° ------ TRAIN ------ ° #
    if args.mode == "train":
        # Training
        agent.grid_search()
    # ° ------ TEST ------ ° #
    elif args.mode == 'test':
        test(agent=agent)
    else:
        raise ValueError("Invalid mode. Please choose between 'train' and 'test'")
