"""Module for the agent class"""
from src.agent.a2c.a2c import ActorCritic
from src.environment.graph_env import GraphEnvironment
from src.utils.utils import HyperParams, FilePaths, Utils

from tqdm import trange
from collections import namedtuple
from typing import List, Tuple
from torch_geometric.data import Data
from torch.nn import functional as F

import networkx as nx
import torch
import json
import gc


class Agent:
    def __init__(
            self,
            env: GraphEnvironment,
            state_dim: int = HyperParams.EMBEDDING_DIM.value,
            hidden_size_1: int = HyperParams.HIDDEN_SIZE_1.value,
            hidden_size_2: int = HyperParams.HIDDEN_SIZE_2.value,
            lr: List[float] = HyperParams.LR.value,
            gamma: List[float] = HyperParams.GAMMA.value,
            lambda_metrics: List[float] = HyperParams.LAMBDA.value,
            alpha_metrics: List[float] = HyperParams.ALPHA.value,
            eps: float = HyperParams.EPS_CLIP.value,
            best_reward: float = HyperParams.BEST_REWARD.value):
        """
        Initialize the agent.

        Parameters
        ----------
        env : GraphEnvironment
            Environment to train the agent on
        state_dim : int
            Dimensions of the state, i.e. length of the feature vector
        hidden_size_1 : int
            First A2C hidden layer size
        hidden_size_2 : int
            Second A2C hidden layer size
        action_dim : int
            Dimensions of the action (it is set to 1, to return a tensor N*1)
        lr : List[float]
            List of Learning rate, each element of the list is a learning rate
        gamma : List[float]
            List of gamma parameter, each element of the list is a gamma
        lambda_metrics : List[float]
            List of lambda parameter, each element of the list is a lambda used
            to balance the reward and the penalty
        alpha_metrics : List[float]
            List of alpha parameter, each element of the list is a alpha used
            to balance the two penalties
        eps : List[float]
            Value for clipping the loss function, each element of the list is a
            clipping value
        best_reward : float, optional
            Best reward, by default 0.8
        """
        # ° ----- Environment ----- ° #
        self.env = env
        
        # ° ----- A2C ----- ° #
        self.state_dim = state_dim
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.action_dim = self.env.graph.number_of_nodes()
        self.policy = ActorCritic(
            state_dim=state_dim,
            hidden_size_1=hidden_size_1,
            hidden_size_2=hidden_size_2,
            action_dim=self.action_dim,
            graph=self.env.graph
        )
        # Set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Move model to device
        self.policy.to(self.device)
        
        # ° ----- Hyperparameters ----- ° #
        # A2C hyperparameters
        self.lr_list = lr
        self.gamma_list = gamma
        self.eps = eps
        self.best_reward = best_reward
        # Environment hyperparameters
        self.lambda_metrics = lambda_metrics
        self.alpha_metrics = alpha_metrics
        # Hyperparameters to be set during grid search
        self.lr = None
        self.gamma = None
        self.alpha_metric = None
        self. optimizers = dict()

        # ° ----- Training ----- ° #
        # State, nx.Graph
        self.obs = None
        # Cumulative reward of the episode
        self.episode_reward = 0
        # Boolean variable to check if the episode is ended
        self.done = False
        # Number of steps in the episode
        self.step = 0
        # Tuple to store the values for each action
        self.SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        self.saved_actions = []
        self.rewards = []
        # Initialize lists for logging, it contains: avg_reward, avg_steps per episode
        self.log_dict = HyperParams.LOG_DICT.value
        # Print agent info
        self.print_agent_info()
        
        # ° ----- Evaluation ----- ° #
        # List of actions performed during the evaluation
        self.action_list = {"ADD": [], "REMOVE": []}

    ############################################################################
    #                       PRE-TRAINING/TESTING                               #
    ############################################################################
    def reset_hyperparams(
        self, 
        lr: float, 
        gamma: float, 
        lambda_metric: float, 
        alpha_metric: float,
        test: bool = False) -> None:
        """
        Reset hyperparameters
        
        Parameters
        ----------
        lr : float
            Learning rate
        gamma : float
            Discount factor
        lambda_metric : float
            Lambda parameter used to balance the reward and the penalty
        alpha_metric : float
            Alpha parameter used to balance the two penalties
        test : bool, optional
            Print hyperparameters during training, by default False
        """
        # Set A2C hyperparameters
        self.lr = lr
        self.gamma = gamma
        # Set environment hyperparameters
        self.env.lambda_metric = lambda_metric
        self.env.alpha_metric = alpha_metric
        # Print hyperparameters if we are not testing
        if not test: self.print_hyperparams()
        # Clear logs, except for the training episodes
        for key in self.log_dict.keys():
            if key != 'train_episodes':
                self.log_dict[key] = list()
        # Clear action list
        self.saved_actions = []
        self.rewards = []
        # Clear state
        self.obs = None
        self.episode_reward = 0
        self.done = False
        self.step = 0
        self.optimizers = dict()

    def configure_optimizers(self) -> None:
        """
        Configure optimizers
        
        Returns
        -------
        optimizers : dict
            Dictionary of optimizers
        """
        actor_params = list(self.policy.actor.parameters())
        critic_params = list(self.policy.critic.parameters())
        self.optimizers['a_optimizer'] = torch.optim.Adam(
            actor_params, lr=self.lr)
        self.optimizers['c_optimizer'] = torch.optim.Adam(
            critic_params, lr=self.lr)
    
    ############################################################################
    #                            GRID SEARCH                                   #
    ############################################################################
    def grid_search(self) -> None:
        """Perform grid search on the hyperparameters"""
        for lr in self.lr_list:
            for gamma in self.gamma_list:
                for lambda_metric in self.lambda_metrics:
                    for alpha_metric in self.alpha_metrics:
                        # Change Hyperparameters
                        self.reset_hyperparams(lr, gamma, lambda_metric, alpha_metric)
                        # Configure optimizers with the current learning rate
                        self.configure_optimizers()
                        # Training
                        log = self.training()
                        # Save results in correct folder
                        self.save_plots(log, self.get_path())
                        # Free memory
                        gc.collect()

    ############################################################################
    #                               TRAINING                                   #
    ############################################################################
    def training(self) -> dict:
        """
        Train the agent on the environment, change the target node every 10
        episodes and the target community every 100 episodes. The episode ends
        when the target node is isolated from the target community, or when the
        maximum number of steps is reached.
            
        Returns
        -------
        log_dict : dict
            Dictionary containing the training logs
        """
        epochs = trange(self.log_dict['train_episodes'])  # epoch iterator
        self.policy.train()  # set model in train mode
        for i_episode in epochs:
            self.obs = self.env.reset()
            self.episode_reward = 0
            self.done = False
            self.step = 0
            # Rewiring the graph until the target node is isolated from the
            # target community
            while not self.done and self.step < self.env.max_steps:
                self.rewiring()
            # perform on-policy backpropagation
            self.a_loss, self.v_loss = self.training_step()
            # Send current statistics to screen
            epochs.set_description(
                f"* Episode {i_episode+1} " +
                f"| Avg Reward: {self.episode_reward/self.step:.2f} " +
                f"| Avg Steps: {self.step} " +
                f"| Actor Loss: {self.a_loss:.2f} " +
                f"| Critic Loss: {self.v_loss:.2f}")
            # Checkpoint best performing model
            if self.episode_reward >= self.best_reward:
                self.save_checkpoint()
                self.best_reward = self.episode_reward
            # Log
            self.log_dict['train_reward'].append(self.episode_reward)
            self.log_dict['train_steps'].append(self.step)
            self.log_dict['train_avg_reward'].append(
                self.episode_reward/self.step)
            self.log_dict['a_loss'].append(self.a_loss)
            self.log_dict['v_loss'].append(self.v_loss)
        return self.log_dict

    def rewiring(self, test=False) -> None:
        """
        Rewiring step, select action and take step in environment.
        
        Parameters
        ----------
        test : bool, optional
            If True, print rewiring action, by default False
        """
        # Select action: return a list of the probabilities of each action
        action_rl = self.select_action(self.obs)
        torch.cuda.empty_cache()
        # Save rewiring action if we are testing
        if test:
            edge = (self.env.node_target, action_rl)
            if edge in self.env.possible_actions["ADD"]:
                if not self.env.graph.has_edge(*edge):
                    self.action_list["ADD"].append(edge)
            elif edge in self.env.possible_actions["REMOVE"]:
                if self.env.graph.has_edge(*edge):
                    self.action_list["REMOVE"].append(edge)
        # Take action in environment
        self.obs, reward, self.done = self.env.step(action_rl)
        # Update reward
        self.episode_reward += reward
        # Store the transition in memory
        self.rewards.append(reward)
        self.step += 1

    def select_action(self, state: Data) -> int:
        """
        Select action, given a state, using the policy network.
        
        Parameters
        ----------
        state : Data
            Graph state
        
        Returns
        -------
        action: int
            Integer representing a node in the graph, it will be the destination
            node of the rewiring action
        """
        concentration, value = self.policy(state)
        dist = torch.distributions.Categorical(concentration)
        action = dist.sample()
        self.saved_actions.append(
            self.SavedAction(dist.log_prob(action), value))
        return int(action.item())

    def training_step(self) -> Tuple[float, float]:
        """
        Perform a single training step of the A2C algorithm, which involves
        computing the actor and critic losses, taking gradient steps, and 
        resetting the rewards and action buffer.
        
        Returns
        -------
        mean_a_loss : float
            Mean actor loss
        mean_v_loss : float
            Mean critic loss
        """
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values
        # Compute the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            # insert to the beginning of the list
            returns.insert(0, R)
        # Normalize returns by subtracting mean and dividing by standard deviation
        # NOTE: May cause NaN problem
        if len(returns) > 1:
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + self.eps)
        else:
            returns = torch.tensor(returns)
        # Computing losses
        for (log_prob, value), R in zip(saved_actions, returns):
            # Difference between true value and estimated value from critic
            advantage = R - value.item()
            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)
            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(
                value, torch.tensor([R]).to(self.device)))
        # take gradient steps
        self.optimizers['a_optimizer'].zero_grad()
        a_loss = torch.stack(policy_losses).sum()
        a_loss.backward()
        self.optimizers['a_optimizer'].step()
        self.optimizers['c_optimizer'].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        v_loss.backward()
        self.optimizers['c_optimizer'].step()
        # Compute mean losses
        mean_a_loss = torch.stack(policy_losses).mean().item()
        mean_v_loss = torch.stack(value_losses).mean().item()
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
        return mean_a_loss, mean_v_loss

    ############################################################################
    #                               TEST                                       #
    ############################################################################
    def test(
        self, 
        lr: float, 
        gamma: float, 
        lambda_metric: float, 
        alpha_metric: float) -> nx.Graph:
        """Hide a given node from a given community"""
        # Set hyperparameters to select the correct folder
        self.reset_hyperparams(lr, gamma, lambda_metric, alpha_metric, True)
        # Load best performing model
        self.load_checkpoint()
        # Set model in evaluation mode
        self.policy.eval()
        self.obs = self.env.reset()
        # Rewiring the graph until the target node is isolated from the
        # target community
        while not self.done and self.step < self.env.max_steps:
            self.rewiring(test=True)
        if self.step >= self.env.max_steps:
            print("* !!!Maximum number of steps reached!!!")
        return self.obs

    ############################################################################
    #                            CHECKPOINTING                                 #
    ############################################################################
    def get_path(self) -> str:
        """
        Return the path of the folder where to save the plots and the logs
        
        Returns
        -------
        file_path : str
            Path to the correct folder
        """
        file_path = FilePaths.LOG_DIR.value + \
            f"{self.env.env_name}/{self.env.detection_alg}/" +\
            f"lr-{self.lr}/gamma-{self.gamma}/" +\
            f"lambda-{self.env.lambda_metric}/alpha-{self.env.alpha_metric}"
        return file_path
    
    def save_plots(self, log: dict, file_path: str) -> None:
        """
        Save training plots and logs

        Parameters
        ----------
        log : dict
            Dict containing the training logs
        file_path : str
            Path to the directory where to save the plots and the logs
        """
        Utils.check_dir(file_path)
        self.log(log)
        # Utils.save_training(
        #     log,
        #     self.env.env_name,
        #     self.env.detection_alg,
        #     file_path)
        Utils.plot_training(
            log,
            self.env.env_name,
            self.env.detection_alg,
            file_path)

    def save_checkpoint(self):
        """Save checkpoint"""
        log_dir = self.get_path()
        # Check if the directory exists, otherwise create it
        Utils.check_dir(log_dir)
        checkpoint = dict()
        checkpoint['model'] = self.policy.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        path = f'{log_dir}/model.pth'
        torch.save(checkpoint, path)

    def load_checkpoint(self):
        """Load checkpoint"""
        log_dir = self.get_path()
        path = f'{log_dir}/model.pth'
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['model'])
        for key, _ in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict: dict):
        """Log data
        
        Parameters
        ----------
        log_dict : dict
            Dictionary containing the data to be logged
        """
        log_dir = self.get_path()
        Utils.check_dir(log_dir)
        file_name = f'{log_dir}/results.json'
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(log_dict, f, indent=4)
    
    ############################################################################
    #                   AGENT INFO AND PRINTING                                #
    ############################################################################
    def print_agent_info(self):
        # Print model architecture
        print("*", "-"*18, " Model Architecture ", "-"*18)
        print("* Embedding dimension: ", self.state_dim)
        print("* A2C Hidden layer 1 size: ", self.hidden_size_1)
        print("* A2C Hidden layer 2 size: ", self.hidden_size_2)
        print("* Actor Action dimension: ", self.action_dim)
        # Print Hyperparameters List
        print("*", "-"*18, "Hyperparameters List", "-"*18)
        print("* Learning rate list: ", self.lr_list)
        print("* Gamma parameter list: ", self.gamma_list)
        print("* Lambda Metric list: ", self.lambda_metrics)
        print("* Alpha Metric list: ", self.alpha_metrics)
    
    def print_hyperparams(self):
        print("*", "-"*18, "Model Hyperparameters", "-"*18)
        print("* Learning rate: ", self.lr)
        print("* Gamma parameter: ", self.gamma)
        print("* Lambda Metric: ", self.env.lambda_metric)
        print("* Alpha Metric: ", self.env.alpha_metric)
        print("* Value for clipping the loss function: ", self.eps)
