from src.utils.utils import HyperParams, Utils, FilePaths, DetectionAlgorithmsNames
from src.environment.graph_env import GraphEnvironment
from src.agent.agent import Agent

from src.utils.test import test
from src.utils.community_deception import community_deception
import argparse
import math


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
        # To change the detection algorithm, or the dataset, on which the model
        # will be tested, please refer to the class HyperParams in the file
        # src/utils/utils.py, changing the values of the variables:
        # - GRAPH_NAME, for the dataset
        # - DETECTION_ALG, for the detection algorithm
        
        # To change the model path, please refer to the class FilePaths in the
        # file src/utils/utils.py
        model_path = FilePaths.TRAINED_MODEL.value
        
        # Change the beta and tau parameters on which the model will be tested
        betas = [HyperParams.BETA.value]  # [1,3,5]
        taus = [0.3, 0.5, 0.8]
        
        # Get communty target
        community_target = agent.env.community_target
        for beta in betas:
            for tau in taus:
                print("* * Testing with beta = {} and tau = {}".format(beta, tau))
                test(agent=agent, model_path=model_path, beta=beta, tau=tau)
                # print("* * Community Hiding with beta = {} and tau = {}".format(beta, tau))
                # community_deception(agent=agent, community_target=community_target, beta=beta, tau=tau, model_path=model_path)
                print("*"*50)
    else:
        raise ValueError("Invalid mode. Please choose between 'train' and 'test'")
