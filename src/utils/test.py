from src.utils.utils import HyperParams, Utils, FilePaths
from src.environment.graph_env import GraphEnvironment
from src.agent.agent import Agent

from src.community_algs.baselines.random_hiding import RandomHiding
from src.community_algs.baselines.degree_hiding import DegreeHiding
from src.community_algs.baselines.roam_hiding import RoamHiding

from typing import List
from tqdm import trange
import time


def test(
    agent: Agent,
    beta: float,
    tau: float,
    model_path: str,
    eval_steps: int = HyperParams.STEPS_EVAL.value,
    lr: float = HyperParams.LR_EVAL.value,
    gamma: float = HyperParams.GAMMA_EVAL.value,
    lambda_metric: float = HyperParams.LAMBDA_EVAL.value,
    alpha_metric: float = HyperParams.ALPHA_EVAL.value)->None:
    """
    Function to evaluate the performance of the agent and compare it with 
    the baseline algorithms.
    
    The baseline algorithms are:
        - Random Hiding
        - Degree Hiding
        - Roam Heuristic

    Parameters
    ----------
    agent : Agent
        Agent to evaluate
    beta : float
        Beta parameter for the number of rewiring steps
    tau : float
        Tau parameter as constraint for community target similarity
    model_path : str
        Path to the model to load
    eval_steps : int, optional
        Number of episodes to test, by default 1000
    lr : float, optional
        Learning rate, by default 1e-3
    gamma : float, optional
        Discount factor, by default 0.3
    lambda_metric : float, optional
        Weight to balance the penalty and reward, by default 0.1
    alpha_metric : float, optional
        Weight to balance the penalties, by default 0.1
    """
    # Initialize the log dictionary
    
    log_dict = Utils.initialize_dict(HyperParams.ALGS_EVAL.value)
    
    # Set parameters in the environment
    agent.env.beta = beta
    agent.env.edge_budget = agent.env.get_edge_budget() * agent.env.beta
    agent.env.max_steps = agent.env.edge_budget * HyperParams.MAX_STEPS_MUL.value
    agent.env.tau = tau
    
    # Add environment parameters to the log dictionary
    log_dict["env"] = dict()
    log_dict["env"]["dataset"] = agent.env.env_name
    log_dict["env"]["detection_alg"] = agent.env.detection_alg
    log_dict["env"]["beta"] = beta
    log_dict["env"]["tau"] = tau
    log_dict["env"]["edge_budget"] = agent.env.edge_budget
    log_dict["env"]["max_steps"] = agent.env.max_steps
    
    # Add Agent Hyperparameters to the log dictionary
    log_dict["Agent"]["lr"] = lr
    log_dict["Agent"]["gamma"] = gamma
    log_dict["Agent"]["lambda_metric"] = lambda_metric
    log_dict["Agent"]["alpha_metric"] = alpha_metric
    
    # Start evaluation
    steps = trange(eval_steps, desc="Testing Episode")
    for step in steps:
        
        # Change the target community and node at each episode
        agent.env.change_target_community()
        
        # ° ------ Agent ------ ° #
        steps.set_description(f"* Testing Episode {step+1} | Agent Rewiring")
        start = time.time()
        new_graph = agent.test(
            lr=lr,
            gamma=gamma,
            lambda_metric=lambda_metric,
            alpha_metric= alpha_metric,
            model_path=model_path,
        )
        # "src/logs/lfr_benchmark_n-300/infomap/lr-0.0001/gamma-0.9/lambda-0.1/alpha-0.7"
        end = time.time() - start
        
        # ° Target node and community for this episode ° #
        # We set it after the test to change automatically at each episode
        community_structure = agent.env.original_community_structure
        community_target = agent.env.community_target
        node_target = agent.env.node_target
        # ° ------------------------------------------ ° #
        # Get new target community after deception
        agent_community = Utils.get_new_community(node_target, agent.env.new_community_structure)
        # Compute NMI between the new community structure and the original one
        agent_nmi = community_structure.normalized_mutual_information(
            agent.env.new_community_structure).score
        # Check if the goal of hiding the target node was achieved
        agent_goal = Utils.check_goal(agent.env, node_target, community_target, agent_community)
        # Save the metrics
        log_dict = save_metrics(
            log_dict, "Agent", agent_goal, agent_nmi, end, agent.step)

        
        # Perform Deception with the baseline algorithms
        # ° ------ Random Hiding ------ ° #
        steps.set_description(f"* Testing Episode {step+1} | Random Rewiring")
        random_hiding = RandomHiding(
            env=agent.env,
            steps=agent.env.edge_budget,
            target_community=community_target)

        start = time.time()
        rh_graph, rh_communities = random_hiding.hide_target_node_from_community()
        end = time.time() - start
        
        # Get new target community after deception
        rh_community = Utils.get_new_community(node_target, rh_communities)
        # Compute NMI between the new community structure and the original one
        rh_nmi = community_structure.normalized_mutual_information(
            rh_communities).score
        # Check if the goal of hiding the target node was achieved
        rh_goal = Utils.check_goal(
            agent.env, node_target, community_target, rh_community)
        # Save the metrics
        log_dict = save_metrics(
            log_dict, "Random", rh_goal, rh_nmi, end, agent.env.edge_budget-random_hiding.steps)
        

        # ° ------ Degree Hiding ------ ° #
        steps.set_description(f"* Testing Episode {step+1} | Degree Rewiring")
        degree_hiding = DegreeHiding(
            env=agent.env,
            steps=agent.env.edge_budget,
            target_community=community_target)

        start = time.time()
        dh_graph, dh_communities = degree_hiding.hide_target_node_from_community()
        end = time.time() - start
        
        # Get new target community after deception
        dh_community = Utils.get_new_community(node_target, dh_communities)
        # Compute NMI between the new community structure and the original one
        dh_nmi = community_structure.normalized_mutual_information(
            dh_communities).score
        # Check if the goal of hiding the target node was achieved
        dh_goal = Utils.check_goal(
            agent.env, node_target, community_target, dh_community)
        # Save the metrics
        log_dict = save_metrics(
            log_dict, "Degree", dh_goal, dh_nmi, end, agent.env.edge_budget-degree_hiding.steps)

        # ° ------ Roam Heuristic ------ ° #
        steps.set_description(f"* Testing Episode {step+1} | Roam Rewiring")
        # Apply Hide and Seek
        deception = RoamHiding(
            agent.env.original_graph.copy(), node_target, agent.env.detection_alg)
        start = time.time()
        di_graph, di_communities = deception.roam_heuristic(
            agent.env.edge_budget)
        end = time.time() - start
        
        # Get new target community after deception
        di_community = Utils.get_new_community(node_target, di_communities)
        # Compute NMI between the new community structure and the original one
        di_nmi = community_structure.normalized_mutual_information(
            di_communities).score
        # Check if the goal of hiding the target node was achieved
        di_goal = Utils.check_goal(
            agent.env, node_target, community_target, di_community)
        # Save the metrics
        log_dict = save_metrics(
            log_dict, "Roam", di_goal, di_nmi, end, agent.env.edge_budget)

        steps.set_description(f"* Testing Episode {step+1}")
    # Save the log
    path = FilePaths.TEST_DIR.value + \
        f"{log_dict['env']['dataset']}/{log_dict['env']['detection_alg']}/" + \
        f"tau-{tau}/beta-{beta}/" + \
        f"lr-{lr}/gamma-{gamma}/lambda-{lambda_metric}/alpha-{alpha_metric}/"
    Utils.check_dir(path)
    Utils.save_test(
        log_dict, 
        path, 
        "evaluation_node_hiding", 
        algs=["Agent", "Random", "Degree", "Roam"],
        metrics=["nmi", "goal", "time", "steps"])


################################################################################
#                               Utility Functions                              #
################################################################################
def save_metrics(
        log_dict: dict, alg: str, goal: int,
        nmi: float, time: float, steps: int) -> dict:
    """Save the metrics of the algorithm in the log dictionary"""
    log_dict[alg]["goal"].append(goal)
    log_dict[alg]["nmi"].append(nmi)
    log_dict[alg]["time"].append(time)
    log_dict[alg]["steps"].append(steps)
    return log_dict
