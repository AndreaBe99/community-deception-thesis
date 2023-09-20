from src.utils.utils import HyperParams, Utils, FilePaths
from src.environment.graph_env import GraphEnvironment
from src.agent.agent import Agent
from src.community_algs.metrics.deception_score import DeceptionScore
from src.community_algs.metrics.safeness import Safeness

import networkx as nx
from typing import List
from tqdm import trange
import time
import math

def community_deception(
    agent: Agent,
    community_target: List[int],
    beta: float,
    tau: float,
    model_path: str,
    eval_steps: int = HyperParams.STEPS_EVAL.value,
    lr: float = HyperParams.LR_EVAL.value,
    gamma: float = HyperParams.GAMMA_EVAL.value,
    lambda_metric: float = HyperParams.LAMBDA_EVAL.value,
    alpha_metric: float = HyperParams.ALPHA_EVAL.value)->None:
    
    # Compute the edge budget for the community hiding
    community_beta = 0.3
    # Number of edges % community BETA
    community_edge_budget = int(
        math.ceil((agent.env.graph.number_of_edges()*community_beta)))
    # Budget for each node
    node_edge_budget = int(math.ceil(community_edge_budget/len(community_target)))
        
    # Set parameters in the environment
    agent.env.beta = community_beta
    
    # agent.env.edge_budget = agent.env.get_edge_budget() * agent.env.beta
    agent.env.edge_budget = node_edge_budget
    
    agent.env.max_steps = agent.env.edge_budget * HyperParams.MAX_STEPS_MUL.value
    agent.env.tau = tau
    agent.env.community_target = community_target
    
    # Start evaluation
    steps = trange(eval_steps, desc="Testing Episode")

    log_dict = {"env": {}, "Agent": {}, "Safeness": {}}
    
    log_dict["env"]["dataset"] = agent.env.env_name
    log_dict["env"]["detection_alg"] = agent.env.detection_alg
    log_dict["env"]["beta"] = community_beta
    log_dict["env"]["tau"] = tau
    log_dict["env"]["edge_budget"] = agent.env.edge_budget
    log_dict["env"]["max_steps"] = agent.env.max_steps
    log_dict["env"]["n_nodes"] = agent.env.graph.number_of_nodes()
    log_dict["env"]["n_edges"] = agent.env.graph.number_of_edges()
    log_dict["env"]["n_nodes_community_target"] = len(community_target)
    
    log_dict["Agent"]["deception_score"] = []
    log_dict["Agent"]["nmi"] = []
    log_dict["Agent"]["goal"] = []
    log_dict["Agent"]["time"] = []
    log_dict["Agent"]["steps"] = []
    
    log_dict["Safeness"]["deception_score"] = []
    log_dict["Safeness"]["nmi"] = []
    log_dict["Safeness"]["goal"] = []
    log_dict["Safeness"]["time"] = []
    log_dict["Safeness"]["steps"] = []
    
    for step in steps:
        
        # Reset the environment
        agent.env.reset()
        
        orginal_communities = agent.env.original_community_structure
        deception_obj = DeceptionScore(community_target)
        
        # ° ------ Agent ------ ° #
        agent_goal_reached = False
        start = time.time()
        agent_steps = 0
        for node in community_target:
            agent.env.node_target = node
            # The agent possible action are changed in the test function, which
            # calls the reset function of the environment
            new_graph = agent.test(
                lr=lr,
                gamma=gamma,
                lambda_metric=lambda_metric,
                alpha_metric=alpha_metric,
                model_path=model_path,
                graph_reset=False,
            )
            agent_steps += agent.step
            if community_target not in agent.env.new_community_structure.communities:
                agent_goal_reached = True
                break
        end = time.time() - start
        log_dict["Agent"]["time"].append(end)
        log_dict["Agent"]["steps"].append(agent_steps)
        # Check if the goal of hiding the target community was achieved
        if agent_goal_reached:
            log_dict["Agent"]["goal"].append(1)
        else:
            log_dict["Agent"]["goal"].append(0)
        # Compute the deception score
        deception_score = deception_obj.compute_deception_score(
            agent.env.new_community_structure.communities,
            nx.number_connected_components(new_graph),
        )
        log_dict["Agent"]["deception_score"].append(deception_score)
        # Compute NMI between the new community structure and the original one
        nmi = orginal_communities.normalized_mutual_information(
            agent.env.new_community_structure)
        log_dict["Agent"]["nmi"].append(nmi.score)

        
        # ° ------ Safeness ------ ° #
        # Reset the environment
        agent.env.reset()
        start = time.time()
        safeness_obj = Safeness(
            agent.env.graph,
            community_target,
        )
        safeness_graph, steps = safeness_obj.community_hiding(
            community_target=community_target,
            edge_budget=agent.env.edge_budget*len(community_target)
        )
        end = time.time() - start
        log_dict["Safeness"]["time"].append(end)
        log_dict["Safeness"]["steps"].append(steps)
        # Compute the new community structure, after deception
        new_communities = agent.env.detection.compute_community(safeness_graph)
        safeness_deception_score = deception_obj.compute_deception_score(
            new_communities.communities,
            nx.number_connected_components(safeness_graph),
        )
        log_dict["Safeness"]["deception_score"].append(safeness_deception_score)
        safeness_nmi = orginal_communities.normalized_mutual_information(
            new_communities
        )
        log_dict["Safeness"]["nmi"].append(safeness_nmi.score)
        if community_target in new_communities.communities:
            log_dict["Safeness"]["goal"].append(0)
        else:
            log_dict["Safeness"]["goal"].append(1)
    
    # Save log_dict
    path = FilePaths.TEST_DIR.value + \
        f"{log_dict['env']['dataset']}/{log_dict['env']['detection_alg']}/" + \
        f"tau-{tau}/beta-{beta}/" + \
        f"lr-{lr}/gamma-{gamma}/lambda-{lambda_metric}/alpha-{alpha_metric}/"
    Utils.check_dir(path)
    Utils.save_test(
        log_dict, 
        path, 
        "evaluation_community_hiding", 
        algs=["Agent", "Safeness"],
        metrics=["deception_score", "nmi", "goal", "time", "steps"])
        
