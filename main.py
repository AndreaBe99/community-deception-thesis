from src.utils.utils import HyperParams, Utils, FilePaths, DetectionAlgorithms
from src.community_algs.detection_algs import DetectionAlgorithm, CommunityDetectionAlgorithm
from src.environment.graph_env import GraphEnvironment
from src.agent.a2c.memory import Memory
from src.agent.agent import Agent
import random


if __name__ == "__main__":
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

    # ° ------ Agent Setup ------ ° #
    # Hyperparameters
    lr_list = [1e-3] # HyperParams.LR.value          # [1e-4, 1e-3, 1e-2, 1e-1]
    gamma_list = [0.3] # HyperParams.GAMMA.value    # [0.3, 0.5, 0.7, 0.9]
    lambda_list = [0.1] # HyperParams.LAMBDA.value  # [0.001, 0.01, 0.1, 1, 10]
    alpha_list = [0.1] # HyperParams.ALPHA.value    # [0.001, 0.01, 0.1, 1, 10]
    # Define the agent
    agent = Agent(
        env=env,
        lr=lr_list,
        gamma=gamma_list,
        lambda_metrics=lambda_list,
        alpha_metrics=alpha_list)
    # Training
    agent.grid_search()
