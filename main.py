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
    detection_alg = DetectionAlgorithms.WALK.value
    print("* Community Detection Algorithm:", detection_alg)
    # Apply the community detection algorithm on the graph
    dct = CommunityDetectionAlgorithm(detection_alg)
    community_structure = dct.compute_community(graph)
    print("* Number of communities found:", len(community_structure.communities))

    # Choose one of the communities found by the algorithm, for now we choose 
    # the community with the highest number of nodes
    community_target = max(community_structure.communities, key=len)
    idx_community = community_structure.communities.index(community_target)
    print("* Community Target:\t", community_target)
    print("* Index Community:\t", idx_community)
    # TEST: Choose a node to remove from the community
    node_target = community_target[random.randint(0, len(community_target)-1)]
    print("* Nodes Target:\t\t", node_target)
    
    # Define the environment
    env = GraphEnvironment(
        graph=graph,
        community=community_target,
        idx_community=idx_community,
        node_target=node_target,
        env_name=env_name,
        community_detection_algorithm=detection_alg)
    # Get list of possible actions which can be performed on the graph by the agent
    n_actions = len(env.possible_actions["ADD"]) + \
        len(env.possible_actions["REMOVE"])
    print("* Number of possible actions:", n_actions)
    print("* Rewiring Budget:", env.edge_budget)

    # ° ------ Agent Setup ------ ° #
    # Define the agent
    agent = Agent()
    # Print Hyperparameters of the Agent (inner method)
    print("*", "-"*53)
    print("*"*20, "End Information", "*"*20, "\n")
    
    log = agent.training(env, env_name, detection_alg)
    file_path = FilePaths.TEST_DIR.value + env_name + '/' + detection_alg
    Utils.check_dir(file_path)
    Utils.save_training(log, env_name, detection_alg, file_path=file_path)
    Utils.plot_training(log, env_name, detection_alg, file_path=file_path)
