from typing import List, Callable, Tuple
from src.utils.utils import HyperParams, Utils, FilePaths
from src.environment.graph_env import GraphEnvironment
from src.community_algs.metrics.deception_score import DeceptionScore
from src.community_algs.baselines.safeness import Safeness
from src.agent.agent import Agent

from tqdm import trange
import networkx as nx
import cdlib
import time
import copy
import math 

class CommunityHiding():
    """
    Class to evaluate the performance of the agent in the community hiding task,
    where the agent has to hide a community from a detection algorithm.
    Futhermore, it is compared with other baselines algorithms:
        - Safeness Community Deception
    """

    def __init__(
            self,
            agent: Agent,
            model_path: str,
            lr: float = HyperParams.LR_EVAL.value,
            gamma: float = HyperParams.GAMMA_EVAL.value,
            lambda_metric: float = HyperParams.LAMBDA_EVAL.value,
            alpha_metric: float = HyperParams.ALPHA_EVAL.value,
            eval_steps: int = HyperParams.STEPS_EVAL.value,) -> None:
        self.agent = agent
        self.original_graph = agent.env.original_graph.copy()
        self.model_path = model_path
        self.env_name = agent.env.env_name
        self.detection_alg = agent.env.detection_alg
        self.community_target = agent.env.community_target

        # Copy the community structure to avoid modifying the original one
        self.community_structure = copy.deepcopy(
            agent.env.original_community_structure)
        # self.node_target = agent.env.node_target

        self.lr = lr
        self.gamma = gamma
        self.lambda_metric = lambda_metric
        self.alpha_metric = alpha_metric
        self.eval_steps = eval_steps

        self.beta = None
        self.tau = None
        self.edge_budget = None
        self.max_steps = None
        
        self.evaluation_algs = ["Agent", "Safeness"]

    def set_parameters(self, beta: int, tau: float) -> None:
        """Set the environment with the new parameters, for new experiments

        Parameters
        ----------
        beta : int
            In this case beta is the percentage of edges to remove or add
        tau : float
            Constraint on the goal achievement
        """
        self.beta = beta
        self.tau = tau

        self.agent.env.tau = tau
        # ! NOTE: It isn't the same beta as the one used in the Node Hiding task
        # self.agent.env.beta = beta
        # self.agent.env.set_rewiring_budget()

        # Budget for the whole community
        self.community_edge_budget = int(math.ceil(self.original_graph.number_of_edges() * \
            (self.beta/100)))
        # Budget for each single node
        self.node_edge_budget = int(math.ceil(self.community_edge_budget / len(
            self.community_target)))
        
        # We can't call the set_rewiring_budget function because we don't have
        # the beta value multiplier, and also we need to adapt to the Community
        # Hiding task, where the budget for the agent is set as the BETA percentage
        # of all the edges in the graph divided by the number of nodes in the
        # target community. So we set manually all the values of set_rewiring_budget
        # function.
        self.agent.env.edge_budget = self.node_edge_budget
        self.agent.env.max_steps = self.node_edge_budget
        self.agent.env.used_edge_budget = 0
        self.agent.env.stop_episode = False
        self.agent.env.reward = 0
        self.agent.env.old_rewards = 0
        self.agent.env.possible_actions = self.agent.env.get_possible_actions()
        self.agent.env.len_add_actions = len(self.agent.env.possible_actions["ADD"])

        # Initialize the log dictionary
        self.set_log_dict()

        self.path_to_save = FilePaths.TEST_DIR.value + \
            f"{self.env_name}/{self.detection_alg}/" + \
            f"tau_{self.tau}/" + \
            f"community_hiding/" + \
            f"beta_{self.beta}/" + \
            f"lr_{self.lr}/gamma_{self.gamma}/" + \
            f"lambda_{self.lambda_metric}/alpha_{self.alpha_metric}/"

    def reset_experiment(self) -> None:
        """
        Reset the environment and the agent at the beginning of each episode,
        and change the target community and node
        """
        self.agent.env.change_target_community()

        # Copy the community target to avoid modifying the original one
        self.community_target = copy.deepcopy(self.agent.env.community_target)
        # self.node_target = self.agent.env.node_target
        
        # Initialize the Deception Score algorithm
        self.deception_score_obj = DeceptionScore(
            copy.deepcopy(self.community_target))
        # Initialize the Safeness algorithm
        self.safeness_obj = Safeness(
            self.original_graph.copy(),
            copy.deepcopy(self.community_target),
        )
    
    def run_experiment(self)->None:
        # Start evaluation
        steps = trange(self.eval_steps, desc="Testing Episode")
        for step in steps:

            # Change the target community and node at each episode
            self.reset_experiment()
            # print("* Node Target:", self.node_target)
            # print("* Community Target:", self.community_target)
            
            # ° ------ Agent Rewiring ------ ° #
            steps.set_description(
                f"* * * Testing Episode {step+1} | Agent Rewiring")
            self.run_alg(self.run_agent)
            
            # ° --------- Baselines --------- ° #
            # Safeness
            steps.set_description(
                f"* * * Testing Episode {step+1} | Safeness")
            self.run_alg(self.run_safeness)
        Utils.check_dir(self.path_to_save)
        Utils.save_test(
            log=self.log_dict,
            files_path=self.path_to_save,
            log_name="evaluation_community_hiding",
            algs=self.evaluation_algs,
            metrics=["nmi", "goal", "deception_score", "time", "steps"])

    
    def run_alg(self, function: Callable) -> None:
        """
        Wrapper function to run the evaluation of a generic algorithm

        Parameters
        ----------
        function : Callable
            Algorithm to evaluate
        """
        start = time.time()
        alg_name, goal, nmi, deception_score, step = function()
        end = time.time() - start
        # Save results in the log dictionary
        self.save_metrics(alg_name, goal, nmi, deception_score, end, step)
    
    ############################################################################
    #                               AGENT                                      #
    ############################################################################
    def run_agent(self) -> Tuple[str, int, float, float, int]:
        """
        Evaluate the agent on the Node Hiding task

        Returns
        -------
        Tuple[str, nx.Graph, int, float, int]
            Algorithm name, goal, nmi, deception score, steps
        """
        tot_steps = 0
        agent_goal_reached = False
        for node in self.community_target:
            self.agent.env.node_target = node
            # The agent possible action are changed in the test function, which
            # calls the reset function of the environment
            new_graph = self.agent.test(
                lr=self.lr,
                gamma=self.gamma,
                lambda_metric=self.lambda_metric,
                alpha_metric=self.alpha_metric,
                model_path=self.model_path,
            )
            # print("Node {} - Steps: {}".format(node, agent.step))
            tot_steps += self.agent.step
            if tot_steps >= self.community_edge_budget:
                if self.community_target not in self.agent.env.new_community_structure.communities:
                    agent_goal_reached = True
                break
        # Compute Deception Score between the new community structure and the
        # original one
        deception_score = self.deception_score_obj.get_deception_score(
            self.agent.env.graph.copy(),
            copy.deepcopy(self.agent.env.new_community_structure.communities),
        )
        # Compute NMI between the new community structure and the original one
        nmi = self.get_nmi(self.community_structure, self.agent.env.new_community_structure)
        goal = 1 if agent_goal_reached else 0
        return self.evaluation_algs[0], goal, nmi, deception_score, tot_steps

    ############################################################################
    #                               BASELINES                                  #
    ############################################################################
    def run_safeness(self) -> Tuple[str, int, float, float, int]:
        """
        Evaluate the Safeness algorithm on the Node Hiding task

        Returns
        -------
        Tuple[str, nx.Graph, int, float, int]
            Algorithm name, goal, nmi, deception score, steps
        """
        new_graph, steps = self.safeness_obj.community_hiding(
            community_target=self.community_target,
            edge_budget=self.community_edge_budget,
        )
        # Compute the new community structure
        new_communities = self.agent.env.detection.compute_community(new_graph)
        
        # Compute Deception Score between the new community structure and the
        # original one
        deception_score = self.deception_score_obj.get_deception_score(
            self.original_graph.copy(),
            copy.deepcopy(new_communities.communities),
        )
        # Compute NMI between the new community structure and the original one
        nmi = self.get_nmi(self.community_structure, new_communities)
        goal = 1 if self.community_target not in new_communities.communities else 0
        return self.evaluation_algs[1], goal, nmi, deception_score, steps

    ############################################################################
    #                               UTILS                                      #
    ############################################################################
    def get_nmi(
            self,
            old_communities: cdlib.NodeClustering,
            new_communities: cdlib.NodeClustering) -> float:
        """
        Compute the Normalized Mutual Information between the old and the new
        community structure

        Parameters
        ----------
        old_communities : cdlib.NodeClustering
            Community structure before deception
        new_communities : cdlib.NodeClustering
            Community structure after deception

        Returns
        -------
        float
            Normalized Mutual Information between the old and the new community
        """
        if new_communities is None:
            # The agent did not perform any rewiring, i.e. are the same communities
            return 1
        return old_communities.normalized_mutual_information(new_communities).score

    ############################################################################
    #                               LOG                                        #
    ############################################################################
    def set_log_dict(self) -> None:
        self.log_dict = dict()

        for alg in self.evaluation_algs:
            self.log_dict[alg] = {
                "goal": [],
                "nmi": [],
                "time": [],
                "deception_score": [],
                "steps": [],
            }

        # Add environment parameters to the log dictionaryù
        self.log_dict["env"] = dict()
        self.log_dict["env"]["dataset"] = self.env_name
        self.log_dict["env"]["detection_alg"] = self.detection_alg
        self.log_dict["env"]["beta"] = self.beta
        self.log_dict["env"]["tau"] = self.tau
        self.log_dict["env"]["edge_budget"] = self.edge_budget
        self.log_dict["env"]["max_steps"] = self.max_steps

        # Add Agent Hyperparameters to the log dictionary
        self.log_dict["Agent"]["lr"] = self.lr
        self.log_dict["Agent"]["gamma"] = self.gamma
        self.log_dict["Agent"]["lambda_metric"] = self.lambda_metric
        self.log_dict["Agent"]["alpha_metric"] = self.alpha_metric

    def save_metrics(
            self,
            alg: str,
            goal: int,
            nmi: float,
            deception_score: float,
            time: float,
            steps: int) -> dict:
        """Save the metrics of the algorithm in the log dictionary"""
        self.log_dict[alg]["goal"].append(goal)
        self.log_dict[alg]["nmi"].append(nmi)
        self.log_dict[alg]["deception_score"].append(deception_score)
        self.log_dict[alg]["time"].append(time)
        self.log_dict[alg]["steps"].append(steps)
