from src.utils.utils import HyperParams, Utils, FilePaths, DetectionAlgorithmsNames
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
    # NOTE To modify the hyperparameters, dataset, detection algorithm, etc. 
    # NOTE  please refer to the file src/utils/utils.py in the class HyperParams
    # ° --- Environment Setup --- ° #
    env = GraphEnvironment()
    # ° ------ Agent Setup ----- ° #
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
