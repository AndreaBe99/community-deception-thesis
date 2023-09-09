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
    graph_path = FilePaths.KAR.value
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
    # Apply the community detection algorithm on the graph
    dct = CommunityDetectionAlgorithm(detection_alg)
    community_structure = dct.compute_community(graph)
    # Choose one of the communities found by the algorithm, for now we choose 
    # the community with the highest number of nodes
    community_target = max(community_structure.communities, key=len)
    idx_community = community_structure.communities.index(community_target)
    node_target = community_target[random.randint(0, len(community_target)-1)]
    # Define the environment
    env = GraphEnvironment(
        graph=graph,
        community=community_target,
        idx_community=idx_community,
        node_target=node_target,
        env_name=env_name,
        community_detection_algorithm=detection_alg)
    # Get list of possible actions which can be performed on the graph by the agent
    n_actions = len(env.possible_actions["ADD"]) + len(env.possible_actions["REMOVE"])
    # Print the environment information
    print("* Community Detection Algorithm:", detection_alg)
    print("* Number of communities found:",len(community_structure.communities))
    print("* Initial Community Target:", community_target)
    print("* Initial Index of the Community Target:", idx_community)
    print("* Initial Nodes Target:", node_target)
    print("* Number of possible actions:", n_actions)
    print("* Rewiring Budget:", env.edge_budget, "=", 
        HyperParams.BETA.value, "*", env.graph.number_of_edges(), "/ 100",)
    print("*", "-"*58, "\n")
    
    # ° ------ Agent Setup ------ ° #
    # Hyperparameters
    lr_list = [1e-3, 1e-2] # [1e-4, 1e-3, 1e-2]
    gamma_list = [0.3] # [0.2, 0.5, 0.8]
    reward_weight_list = [0.1] # [0.001, 0.01, 0.1, 1, 10]
    # Define the agent
    agent = Agent(
        env=env, 
        lr=lr_list, 
        gamma=gamma_list,
        reward_weight=reward_weight_list)
    # Training
    agent.grid_search()
