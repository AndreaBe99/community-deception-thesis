from typing import List, Callable, Tuple
from src.utils.utils import HyperParams, Utils, FilePaths
from src.environment.graph_env import GraphEnvironment
from src.community_algs.metrics.deception_score import DeceptionScore
# from src.community_algs.baselines.community_hiding.test_safeness import Safeness
from src.community_algs.baselines.community_hiding.safeness import Safeness
from src.community_algs.baselines.community_hiding.modularity import Modularity

# TEST 
from src.community_algs.baselines.community_hiding.safeness_tets import Safeness as SafenessTest
from src.community_algs.baselines.community_hiding.modularity_test import Modularity as ModularityTest

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
            epsilon_prob: float = HyperParams.EPSILON_EVAL.value,
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
        self.epsilon_prob = epsilon_prob
        self.eval_steps = eval_steps

        self.beta = None
        self.tau = None
        self.edge_budget = None
        self.max_steps = None
        
        self.evaluation_algs = ["Agent", "Safeness", "Modularity"]
        
        # Use a list to store the beta values already computed, beacuse the
        # Community Deception algorithms are not influenced by the value of
        # tau, so we can compute the beta values only once
        self.beta_values_computed = []
        # Use a dict to store the results of the Community Deception algorithms
        # for each beta value
        self.beta_values_results = dict()

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
        
        # Set community beta value as key of the dictionary
        if self.beta not in self.beta_values_results:
            self.beta_values_results[self.beta] = dict()

        self.agent.env.tau = tau
        # ! NOTE: It isn't the same beta as the one used in the Node Hiding task
        # self.agent.env.beta = beta
        # self.agent.env.set_rewiring_budget()

        # Budget for the whole community
        self.community_edge_budget = int(math.ceil(self.original_graph.number_of_edges() * \
            (self.beta/100)))
        # Set the node budge as the community budget 
        self.node_edge_budget = self.community_edge_budget
        
        # We can't call the set_rewiring_budget function because we don't have
        # the beta value multiplier, and also we need to adapt to the Community
        # Hiding task, where the budget for the agent is set as the BETA percentage
        # of all the edges in the graph divided by the number of nodes in the
        # target community. So we set manually all the values of set_rewiring_budget
        # function.
        self.agent.env.edge_budget = self.node_edge_budget
        # ! self.agent.env.max_steps = self.agent.env.original_graph.number_of_edges()
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
            f"community_hiding/" + \
            f"tau_{self.tau}/" + \
            f"beta_{self.beta}/" + \
            f"eps_{self.epsilon_prob}/" + \
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
        
        self.safeness_obj = Safeness(
            self.community_edge_budget,
            self.original_graph,
            self.community_target,
            self.community_structure,
        )
        
        self.modularity_obj = Modularity(
            self.community_edge_budget,
            self.original_graph,
            self.community_target,
            self.community_structure,
            self.agent.env.detection,
        )
        
        # ! UNCOMMENT
        # Compute a Dictionary where the keys are the nodes of the community
        # target and the values are the centrality of the nodes
        node_centralities = nx.centrality.degree_centrality(
            self.original_graph)
        # Get the subset of the dictionary with only the nodes of the community
        node_com_centralities = {
            k: node_centralities[k] for k in self.community_target}
        # Order in descending order the dictionary
        self.node_com_centralities = dict(
            sorted(
                node_com_centralities.items(),
                key=lambda item: item[1],
                reverse=True)
        )
        
        # ! Compute the budget for each node in the target community, for the
        # function run_agent_distributed_budget()
        # self.compute_budget_proportionally(self.original_graph, self.community_target)
        
    def compute_budget_proportionally(
        self, 
        graph: nx.Graph, 
        community_target: List[int]) -> None:
        """
        Compute the budget for each node in the target community, proportionally
        to the degree of each node.

        Parameters
        ----------
        graph : nx.Graph
            Graph on which the agent is acting
        community_target : List[int]
            Target community
        """
        # Calculate the total degree of all nodes in the graph
        total_degree = sum(dict(graph.degree()).values())
        remaining_budget = self.community_edge_budget
        self.budget_per_node = {}

        if total_degree == 0:
            # Divide the budget equally between all nodes
            budget_per_node = self.community_edge_budget // len(community_target)
            for node in community_target:
                self.budget_per_node[node] = budget_per_node
            return

        # Order the nodes in descending order based on their degree
        sorted_nodes = sorted(community_target, key=lambda n: graph.degree(n), reverse=True)

        for node in sorted_nodes:
            degree = graph.degree(node)
            proportion = degree / total_degree
            new_budget = math.ceil(self.community_edge_budget * proportion)
            if remaining_budget - new_budget < 0:
                new_budget = 0
            self.budget_per_node[node] = new_budget
            remaining_budget -= new_budget

    
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
            # ! UNCOMMENT
            self.run_alg(self.run_agent)
            # self.run_alg(self.run_agent_distributed_budget)
            
            # ° --------- Baselines --------- ° #
            # Check if the beta value is already computed, if yes, skip
            if self.beta in self.beta_values_computed:
                continue
            
            # Safeness
            steps.set_description(
                f"* * * Testing Episode {step+1} | Safeness Rewiring")
            self.run_alg(self.run_safeness)
            
            # Modularity
            steps.set_description(
                f"* * * Testing Episode {step+1} | Modularity Rewiring")
            self.run_alg(self.run_modularity)
        
        # If the beta value is already computed, copy the results in the log
        # dictionary, otherwise save the results in the backup dictionary
        # for future iterations
        if self.beta not in self.beta_values_computed:
            self.beta_values_computed.append(self.beta)
            self.beta_values_results[self.beta][
                self.evaluation_algs[1]] = self.log_dict[self.evaluation_algs[1]]
            self.beta_values_results[self.beta][
                self.evaluation_algs[2]] = self.log_dict[self.evaluation_algs[2]]
        else:    
            # Cycle for each algorithm (except the agent) and for each metric
            # and save the results in the log dictionary
            for alg in self.evaluation_algs:
                if alg != self.evaluation_algs[0]:
                    for metric in self.beta_values_results[self.beta][alg]:
                        self.log_dict[alg][metric] += self.beta_values_results[
                            self.beta][alg][metric]

                
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
        Evaluate the agent on the Node Hiding task. In this case the agent starts
        to hide the node with the highest centrality in the target community, and
        with the budget equal to the Community Deception baselines, and it is 
        scaled down at each step, based on the number of steps performed.

        Returns
        -------
        Tuple[str, nx.Graph, int, float, int]
            Algorithm name, goal, nmi, deception score, steps
        """
        tot_steps = 0
        agent_goal_reached = False
        # Initialize the new community structure as the original one, because
        # the agent could not perform any rewiring
        communities = self.community_structure
        # As first node to hide, we choose the node with the highest centrality
        # in the target community
        node = self.node_com_centralities.popitem()[0]
        while True:
            self.agent.env.node_target = node
            # The agent possible action are changed in the test function, which
            # calls the reset function of the environment
            new_graph = self.agent.test(
                lr=self.lr,
                gamma=self.gamma,
                lambda_metric=self.lambda_metric,
                alpha_metric=self.alpha_metric,
                epsilon_prob=self.epsilon_prob,
                model_path=self.model_path,
            )
            # Get the new community structure
            new_communities = self.agent.env.new_community_structure
            # Check if the agent performed any rewiring
            if new_communities is None:
                new_communities = communities
            # Get the community in the new community structure, which contains
            # the highest number of nodes of the target community
            new_community = max(new_communities.communities, key=lambda c: sum(
                1 for n in self.community_target if n in c))
            # Recompute the node centralities after the rewiring
            node_centralities = nx.centrality.degree_centrality(new_graph)
            # Choose the next node to hide, as the node with the highest 
            # centrality in the new community
            node = max(
                (n for n in new_community if n in self.community_target), 
                key=lambda n: node_centralities[n])
            # ! Increment the total steps
            # tot_steps += self.agent.step
            tot_steps += self.agent.env.used_edge_budget
            # Reduce the edge budget
            self.agent.env.edge_budget = self.node_edge_budget - tot_steps
            # print("Edge Budget Used:", self.agent.env.used_edge_budget)
            # print("Agent Steps:", self.agent.step)
            # Check if the agent reached the goal
            if tot_steps >= self.community_edge_budget or node is None:
                if self.agent.env.new_community_structure is None:
                    # The agent did not perform any rewiring, i.e. are the same communities
                    agent_goal_reached = False
                    break
                if self.community_target not in self.agent.env.new_community_structure.communities:
                    agent_goal_reached = True
                communities = self.agent.env.new_community_structure
                break
        
        # Compute Deception Score between the new community structure and the
        # original one
        deception_score = self.deception_score_obj.get_deception_score(
            new_graph.copy(),
            copy.deepcopy(communities.communities),
        )
        # Compute NMI between the new community structure and the original one
        nmi = self.get_nmi(self.community_structure, self.agent.env.new_community_structure)
        goal = 1 if agent_goal_reached else 0
        return self.evaluation_algs[0], goal, nmi, deception_score, tot_steps

    # ! Function not used by default, change the function run_experiment() to use it
    def run_agent_distributed_budget(self) -> Tuple[str, int, float, float, int]:
        """
        Evaluate the agent on the Node Hiding task. In this case the budget is
        distributed proportionally to the degree of each node in the target
        community, and the agent starts to hide the node with the highest budget.

        Returns
        -------
        Tuple[str, nx.Graph, int, float, int]
            Algorithm name, goal, nmi, deception score, steps
        """
        tot_steps = 0
        agent_goal_reached = False
        # Initialize the new community structure as the original one, because
        # the agent could not perform any rewiring
        communities = self.community_structure
        # Choose the node from the target community with the highest budget
        node = None
        max_budget = 0
        for n in self.community_target:
            if self.budget_per_node[n] > max_budget:
                node = n
                max_budget = self.budget_per_node[n]
        if node is None:
            print(self.budget_per_node)
            print(self.community_edge_budget)
            for node in self.community_target:
                print(node, self.original_graph.degree(node))
            raise Exception("Node is None")
        while True:
            self.agent.env.node_target = node
            # Set the agent edge budget as the budget of the node
            self.agent.env.edge_budget = self.budget_per_node[node]
            # The agent possible action are changed in the test function, which
            # calls the reset function of the environment
            new_graph = self.agent.test(
                lr=self.lr,
                gamma=self.gamma,
                lambda_metric=self.lambda_metric,
                alpha_metric=self.alpha_metric,
                epsilon_prob=self.epsilon_prob,
                model_path=self.model_path,
            )
            # Get the new community structure
            new_communities = self.agent.env.new_community_structure
            # Check if the agent performed any rewiring
            if new_communities is None:
                new_communities = communities
            # Get the community in the new community structure, which contains
            # the highest number of nodes of the target community
            new_community = max(new_communities.communities, key=lambda c: sum(
                1 for n in self.community_target if n in c))

            # Recompute the self.budget_per_node after the rewiring
            self.compute_budget_proportionally(new_graph, new_community)
            # Choose the next node to hide, as the node with the highest
            # centrality in the new community, that is also in the target community
            node = None
            max_budget = 0
            for n in self.community_target:
                if n in new_community and self.budget_per_node[n] > max_budget:
                    node = n
                    max_budget = self.budget_per_node[n]
            # Increment the total steps
            tot_steps += self.agent.env.used_edge_budget
            # Check if the agent exceeded the budget.
            # Some times it can happen that there is still budget, but no more
            # nodes to hide, it is caused by the fact that the bugdet is
            # distributed proportionally and it is rounded up, so it can happen
            # that the sum of the budget of the nodes is less than the total
            # budget, (see the function compute_budget_proportionally)
            if tot_steps >= self.community_edge_budget or node is None:
                if self.agent.env.new_community_structure is None:
                    # The agent did not perform any rewiring, i.e. are the same communities
                    agent_goal_reached = False
                    break
                if self.community_target not in self.agent.env.new_community_structure.communities:
                    agent_goal_reached = True
                communities = self.agent.env.new_community_structure
                break

        # Compute Deception Score between the new community structure and the
        # original one
        deception_score = self.deception_score_obj.get_deception_score(
            new_graph.copy(),
            copy.deepcopy(communities.communities),
        )
        # Compute NMI between the new community structure and the original one
        nmi = self.get_nmi(self.community_structure,
                           self.agent.env.new_community_structure)
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
        # new_graph, steps = self.safeness_obj.community_hiding(
        #    community_target=self.community_target,
        #    edge_budget=self.community_edge_budget,
        # )
        new_graph, steps = self.safeness_obj.run()
        
        # Compute the new community structure
        new_communities = self.agent.env.detection.compute_community(new_graph)
        
        # Compute Deception Score between the new community structure and the
        # original one
        deception_score = self.deception_score_obj.get_deception_score(
            new_graph.copy(),
            copy.deepcopy(new_communities.communities),
        )
        # Compute NMI between the new community structure and the original one
        nmi = self.get_nmi(self.community_structure, new_communities)
        goal = 1 if self.community_target not in new_communities.communities else 0
        return self.evaluation_algs[1], goal, nmi, deception_score, steps
    
    def run_modularity(self) -> Tuple[str, int, float, float, int]:
        """
        Evaluate the Safeness algorithm on the Node Hiding task

        Returns
        -------
        Tuple[str, nx.Graph, int, float, int]
            Algorithm name, goal, nmi, deception score, steps
        """
        new_graph, steps, new_communities = self.modularity_obj.run()
        # TEST
        # new_graph, steps = self.modularity_obj.run()
        # Compute the new community structure
        # new_communities = self.agent.env.detection.compute_community(new_graph)

        # Compute Deception Score between the new community structure and the
        # original one
        deception_score = self.deception_score_obj.get_deception_score(
            new_graph.copy(),
            copy.deepcopy(new_communities.communities),
        )
        # print("Deception Score:", deception_score)
        # Compute NMI between the new community structure and the original one
        nmi = self.get_nmi(self.community_structure, new_communities)
        goal = 1 if self.community_target not in new_communities.communities else 0
        return self.evaluation_algs[2], goal, nmi, deception_score, steps

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
        self.log_dict["Agent"]["epsilon_prob"] = self.epsilon_prob

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
