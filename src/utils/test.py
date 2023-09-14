from src.utils.utils import HyperParams, Utils, FilePaths
from src.environment.graph_env import GraphEnvironment
from src.agent.agent import Agent


from src.community_algs.baselines.random_hiding import RandomHiding
from src.community_algs.baselines.degree_hiding import DegreeHiding
from src.community_algs.baselines.roam_hiding import RoamHiding

from tqdm import trange
import time


def check_goal(
    env: GraphEnvironment,
    node_target: int, 
    old_community: int, 
    new_community: int) -> int:
    """
    Check if the goal of hiding the target node was achieved

    Parameters
    ----------
    env : GraphEnvironment
        Environment of the agent
    node_target : int
        Target node
    old_community : int
        Original community of the target node
    new_community : int
        New community of the target node
    similarity_function : Callable
        Similarity function to use
        
    Returns
    -------
    int
        1 if the goal was achieved, 0 otherwise
    """
    # Copy the communities to avoid modifying the original ones
    new_community_copy = new_community.copy()
    new_community_copy.remove(node_target)
    old_community_copy = old_community.copy()
    old_community_copy.remove(node_target)
    # Compute the similarity between the new and the old community
    similarity = env.community_similarity(
        new_community_copy,
        old_community_copy
    )
    del new_community_copy, old_community_copy
    if similarity <= env.tau:
        return 1
    return 0


def test(
    agent: Agent,
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
    log_dict = HyperParams.EVAL_DICT.value
    log_dict['env_name'] = agent.env.env_name
    log_dict['detection_alg'] = agent.env.detection_alg
    log_dict["agent"]["lr"] = lr
    log_dict["agent"]["gamma"] = gamma
    log_dict["agent"]["lambda_metric"] = lambda_metric
    log_dict["agent"]["alpha_metric"] = alpha_metric
    
    # Start evaluation
    steps = trange(eval_steps, desc="Testing Episode")
    for step in steps:
        
        # ° ------ Agent ------ ° #
        start = time.time()
        new_graph = agent.test(
            lr=lr,
            gamma=gamma,
            lambda_metric=lambda_metric,
            alpha_metric= alpha_metric)
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
        agent_goal = check_goal(agent.env, node_target, community_target, agent_community)
        # Save the metrics
        log_dict = Utils.save_metrics(
            log_dict, "agent", agent_goal, agent_nmi, end, agent.step)

        
        # Perform Deception with the baseline algorithms
        # ° ------ Random Hiding ------ ° #
        random_hiding = RandomHiding(
            graph=agent.env.original_graph,
            steps=agent.env.edge_budget,
            target_node=node_target,
            target_community=community_target,
            detection_alg=agent.env.detection_alg)
        
        start = time.time()
        rh_graph, rh_communities = random_hiding.hide_target_node_from_community()
        end = time.time() - start
        
        # Get new target community after deception
        rh_community = Utils.get_new_community(node_target, rh_communities)
        # Compute NMI between the new community structure and the original one
        rh_nmi = community_structure.normalized_mutual_information(
            rh_communities).score
        # Check if the goal of hiding the target node was achieved
        rh_goal = check_goal(agent.env, node_target, community_target, rh_community)
        # Save the metrics
        log_dict = Utils.save_metrics(
            log_dict, "rh", rh_goal, rh_nmi, end, agent.env.edge_budget-random_hiding.steps)
        

        # ° ------ Degree Hiding ------ ° #
        degree_hiding = DegreeHiding(
            graph=agent.env.original_graph,
            steps=agent.env.edge_budget,
            target_node=node_target,
            target_community=community_target,
            detetion_alg=agent.env.detection_alg)
        start = time.time()
        dh_graph, dh_communities = degree_hiding.hide_target_node_from_community()
        end = time.time() - start
        
        # Get new target community after deception
        dh_community = Utils.get_new_community(node_target, dh_communities)
        # Compute NMI between the new community structure and the original one
        dh_nmi = community_structure.normalized_mutual_information(
            dh_communities).score
        # Check if the goal of hiding the target node was achieved
        dh_goal = check_goal(agent.env, node_target, community_target, dh_community)
        # Save the metrics
        log_dict = Utils.save_metrics(
            log_dict, "dh", dh_goal, dh_nmi, end, agent.env.edge_budget-degree_hiding.steps)

        # ° ------ Roam Heuristic ------ ° #
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
        di_goal = check_goal(agent.env, node_target, community_target, di_community)
        # Save the metrics
        log_dict = Utils.save_metrics(
            log_dict, "di", di_goal, di_nmi, end, agent.env.edge_budget)

        steps.set_description(f"* Testing Episode {step+1}")
    # Save the log
    path = FilePaths.TEST_DIR.value + f"{agent.env.env_name}/{agent.env.detection_alg}"
    Utils.check_dir(path)
    Utils.save_test(log_dict, path)

