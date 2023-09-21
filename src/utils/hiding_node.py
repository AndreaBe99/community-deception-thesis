from src.utils.utils import HyperParams, Utils, FilePaths
from src.environment.graph_env import GraphEnvironment
from src.agent.agent import Agent

from src.community_algs.baselines.random_hiding import RandomHiding
from src.community_algs.baselines.degree_hiding import DegreeHiding
from src.community_algs.baselines.roam_hiding import RoamHiding

from typing import List, Callable, Tuple
from tqdm import trange
import networkx as nx
import cdlib
import time
import copy

class NodeHiding():
    """
    Class to evaluate the performance of the agent on the Node Hiding task, and 
    compare it with the baseline algorithms:
        - Random Hiding: choose randomly the edges to remove/add
        - Degree Hiding: choose the edges to remove/add based on the degree 
        - Roam Heuristic: use roam heuristic
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
        self.community_structure = copy.deepcopy(agent.env.original_community_structure)
        self.node_target = agent.env.node_target

        self.lr = lr
        self.gamma = gamma
        self.lambda_metric = lambda_metric
        self.alpha_metric = alpha_metric
        self.eval_steps = eval_steps
        
        self.beta = None
        self.tau = None
        self.edge_budget = None
        self.max_steps = None
        
        # HyperParams.ALGS_EVAL.value
        self.evaluation_algs = ["Agent", "Random", "Degree", "Roam"]

    def set_parameters(self, beta: int, tau: float) -> None:
        """Set the environment with the new parameters, for new experiments

        Parameters
        ----------
        beta : int
            Multiplicative factor for the number of edges to remove/add
        tau : float
            Constraint on the goal achievement
        """
        self.beta = beta
        self.tau = tau
        
        self.agent.env.beta = beta
        self.agent.env.tau = tau
        self.agent.env.set_rewiring_budget()
        
        self.edge_budget = self.agent.env.edge_budget
        self.max_steps = self.agent.env.max_steps
        
        # Initialize the log dictionary
        self.set_log_dict()
        
        self.path_to_save = FilePaths.TEST_DIR.value + \
            f"{self.env_name}/{self.detection_alg}/" + \
            f"tau_{self.tau}/" + \
            f"node_hiding/" + \
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
        self.node_target = self.agent.env.node_target
    
    ############################################################################
    #                               EVALUATION                                 #
    ############################################################################
    def run_experiment(self):
        """
        Function to run the evaluation of the agent on the Node Hiding task,
        and compare it with the baseline algorithms
        """
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

            # ° ------   Baselines   ------ ° #
            # Random Rewiring
            steps.set_description(
                f"* * * Testing Episode {step+1} | Random Rewiring")
            self.run_alg(self.run_random)

            # Degree Rewiring
            steps.set_description(
                f"* * * Testing Episode {step+1} | Degree Rewiring")
            self.run_alg(self.run_degree)

            # Roam Rewiring
            steps.set_description(
                f"* * * Testing Episode {step+1} | Roam Rewiring")
            self.run_alg(self.run_roam)

        Utils.check_dir(self.path_to_save)
        Utils.save_test(
            log=self.log_dict,
            files_path=self.path_to_save,
            log_name="evaluation_node_hiding",
            algs=self.evaluation_algs,
            metrics=["nmi", "goal", "time", "steps"])
    
    def run_alg(self, function: Callable) -> None:
        """
        Wrapper function to run the evaluation of a generic algorithm

        Parameters
        ----------
        function : Callable
            Algorithm to evaluate
        """
        start = time.time()
        alg_name, new_graph, goal, nmi, step = function()
        end = time.time() - start
        # Save results in the log dictionary
        self.save_metrics(alg_name, goal, nmi, end, step)
        
    ############################################################################
    #                               AGENT                                      #
    ############################################################################
    def run_agent(self) -> Tuple[str, nx.Graph, int, float, int]:
        """
        Evaluate the agent on the Node Hiding task

        Returns
        -------
        Tuple[str, nx.Graph, int, float, int]
            Algorithm name, new graph, goal, nmi, steps
        """
        new_graph = self.agent.test(
            lr=self.lr,
            gamma=self.gamma,
            lambda_metric=self.lambda_metric,
            alpha_metric=self.alpha_metric,
            model_path=self.model_path,
        )
        # Compute NMI between the new community structure and the original one
        nmi = self.get_nmi(self.community_structure, self.agent.env.new_community_structure)
        # Check if the goal of hiding the target node was achieved
        community_target = self.get_new_community(self.agent.env.new_community_structure)
        goal = self.check_goal(community_target)
        return self.evaluation_algs[0], new_graph, goal, nmi, self.agent.step
    
    ############################################################################
    #                               BASELINES                                  #
    ############################################################################
    def run_random(self) -> Tuple[str, nx.Graph, int, float, int]:
        """
        Evaluate the Random Hiding algorithm on the Node Hiding task

        Returns
        -------
        Tuple[str, nx.Graph, int, float, int]
            Algorithm name, new graph, goal, nmi, steps
        """
        random_hiding = RandomHiding(
            env=self.agent.env,
            steps=self.edge_budget,
            target_community=self.community_target)
        rh_graph, rh_communities = random_hiding.hide_target_node_from_community()
        # Compute NMI between the new community structure and the original one
        nmi = self.get_nmi(self.community_structure, rh_communities)
        # Check if the goal of hiding the target node was achieved
        rh_community_target = self.get_new_community(rh_communities)
        goal = self.check_goal(rh_community_target)
        steps = self.edge_budget - random_hiding.steps
        return self.evaluation_algs[1], rh_graph, goal, nmi, steps
    
    def run_degree(self) -> Tuple[str, nx.Graph, int, float, int]:
        """
        Evaluate the Degree Hiding algorithm on the Node Hiding task

        Returns
        -------
        Tuple[str, nx.Graph, int, float, int]
            Algorithm name, new graph, goal, nmi, steps
        """
        degree_hiding = DegreeHiding(
            env=self.agent.env,
            steps=self.edge_budget,
            target_community=self.community_target)
        dh_graph, dh_communities = degree_hiding.hide_target_node_from_community()
        # Compute NMI between the new community structure and the original one
        nmi = self.get_nmi(self.community_structure, dh_communities)
        # Check if the goal of hiding the target node was achieved
        dh_community_target = self.get_new_community(dh_communities)
        goal = self.check_goal(dh_community_target)
        steps = self.edge_budget - degree_hiding.steps
        return self.evaluation_algs[2], dh_graph, goal, nmi, steps
    
    def run_roam(self) -> Tuple[str, nx.Graph, int, float, int]:
        """
        Evaluate the Roam Hiding algorithm on the Node Hiding task

        Returns
        -------
        Tuple[str, nx.Graph, int, float, int]
            Algorithm name, new graph, goal, nmi, steps
        """
        roam_hiding = RoamHiding(
            self.original_graph.copy(),
            self.node_target,
            self.edge_budget,
            self.detection_alg)
        ro_graph, ro_communities = roam_hiding.roam_heuristic(self.edge_budget)
        # Compute NMI between the new community structure and the original one
        nmi = self.get_nmi(self.community_structure, ro_communities)
        # Check if the goal of hiding the target node was achieved
        ro_community_target = self.get_new_community(ro_communities)
        goal = self.check_goal(ro_community_target)
        steps = self.edge_budget
        return self.evaluation_algs[3], ro_graph, goal, nmi, steps

    
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
    
    def get_new_community(
        self,
        new_community_structure: List[List[int]]) -> List[int]:
        """
        Search the community target in the new community structure after 
        deception. As new community target after the action, we consider the 
        community that contains the target node, if this community satisfies 
        the deception constraint, the episode is finished, otherwise not.

        Parameters
        ----------
        node_target : int
            Target node to be hidden from the community
        new_community_structure : List[List[int]]
            New community structure after deception

        Returns
        -------
        List[int]
            New community target after deception
        """
        if new_community_structure is None:
            # The agent did not perform any rewiring, i.e. are the same communities
            return self.community_target
        for community in new_community_structure.communities:
            if self.node_target in community:
                return community
        raise ValueError("Community not found")

    def check_goal(self, new_community: int) -> int:
        """
        Check if the goal of hiding the target node was achieved

        Parameters
        ----------
        new_community : int
            New community of the target node

        Returns
        -------
        int
            1 if the goal was achieved, 0 otherwise
        """
        if len(new_community) == 1:
            return 1
        # Copy the communities to avoid modifying the original ones
        new_community_copy = new_community.copy()
        new_community_copy.remove(self.node_target)
        old_community_copy = self.community_target.copy()
        old_community_copy.remove(self.node_target)
        # Compute the similarity between the new and the old community
        similarity = self.agent.env.community_similarity(
            new_community_copy,
            old_community_copy
        )
        del new_community_copy, old_community_copy
        if similarity <= self.tau:
            return 1
        return 0
    
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
            time: float,
            steps: int) -> dict:
        """Save the metrics of the algorithm in the log dictionary"""
        self.log_dict[alg]["goal"].append(goal)
        self.log_dict[alg]["nmi"].append(nmi)
        self.log_dict[alg]["time"].append(time)
        self.log_dict[alg]["steps"].append(steps)
