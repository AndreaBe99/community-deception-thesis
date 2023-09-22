from src.utils.utils import HyperParams, Utils, FilePaths, DetectionAlgorithmsNames
from src.environment.graph_env import GraphEnvironment
from src.agent.agent import Agent

from src.utils.hiding_node import NodeHiding
from src.utils.hiding_community import CommunityHiding

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
    parser.add_argument('--mode', type=str, default='both',
                        help='train | test | both')
    # Argument parsing
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    # ° --- Environment Setup --- ° #
    env = GraphEnvironment()
    
    # ° ------  Agent Setup ----- ° #
    agent = Agent(env=env)
    
    # ° ------    TRAIN    ------ ° #
    if args.mode == "train" or args.mode == "both":
        # Training
        agent.grid_search()
    
    # ° ------    TEST    ------ ° #
    elif args.mode == 'test' or args.mode == "both":
        # To change the detection algorithm, or the dataset, on which the model
        # will be tested, please refer to the class HyperParams in the file
        # src/utils/utils.py, changing the values of the variables:
        # - GRAPH_NAME, for the dataset
        # - DETECTION_ALG, for the detection algorithm
        
        # To change the model path, please refer to the class FilePaths in the
        # file src/utils/utils.py
        model_path = FilePaths.TRAINED_MODEL.value
        
        # Tau defines the strength of the constraint on the goal achievement
        taus = [0.3, 0.5, 0.8]
        # BETAs defines the number of actions to perform
        # Beta for the community hiding task defines the percentage of rewiring 
        # action, add or remove edges
        community_betas = [1, 3, 5]
        # Beta for the node hiding task is a multiplier of mean degree of the
        # the graph
        node_betas = [1, 2, 3]  # [1,3,5]
        
        # Initialize the test class
        node_hiding = NodeHiding(agent=agent, model_path=model_path)
        community_hiding = CommunityHiding(agent=agent, model_path=model_path)
        
        print("* NOTE:")
        print("*    - Beta for Node Hiding is a multiplier of the mean degree of the graph")
        print("*    - Beta for Community Hiding is the percentage of rewiring action, add or remove edges")
        for tau in taus:
            
            print("* Node Hiding with tau = {}".format(tau))
            for beta in node_betas:
                print("* * Beta Node = {}".format(beta))
                node_hiding.set_parameters(beta=beta, tau=tau)
                node_hiding.run_experiment()
                
            print("* Community Hiding with tau = {}".format(tau))
            for beta in community_betas:
                print("* * Beta Community = {}".format(beta))
                community_hiding.set_parameters(beta=beta, tau=tau)
                community_hiding.run_experiment()
            print("* "*50)
    else:
        raise ValueError("Invalid mode. Please choose between 'train' and 'test'")
