"""Module for the agent class"""
from src.agent.a2c.a2c import ActorCritic
from src.agent.a2c.memory import Memory
from src.environment.graph_env import GraphEnvironment
from src.utils.utils import HyperParams, FilePaths, Utils

from collections import namedtuple
from typing import List, Tuple

from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch.nn import functional as F
from torch import nn
import torch

from tqdm import trange
import numpy as np
import random
import os
import gc


class Agent:
    def __init__(
            self,
            env: GraphEnvironment,
            state_dim: int = HyperParams.STATE_DIM.value,
            hidden_size_1: int = HyperParams.HIDDEN_SIZE_1.value,
            hidden_size_2: int = HyperParams.HIDDEN_SIZE_2.value,
            action_dim: int = HyperParams.ACTION_DIM.value,
            lr: List[float] = [HyperParams.LR.value],
            gamma: List[float] = [HyperParams.GAMMA.value],
            reward_weight: List[float] = [HyperParams.WEIGHT.value],
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
        reward_weight : List[float]
            List of reward weight, each element of the list is a reward weight
        eps : List[float]
            Value for clipping the loss function, each element of the list is a
            clipping value
        best_reward : float, optional
            Best reward, by default 0.8
        """
        self.env = env
        self.state_dim = state_dim
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.action_dim = action_dim
        self.policy = ActorCritic(
            state_dim=self.env.graph.number_of_nodes(),  # state_dim,
            hidden_size_1=hidden_size_1,
            hidden_size_2=hidden_size_2,
            action_dim=action_dim)

        # Hyperparameters
        self.lr_list = lr
        self.gamma_list = gamma
        self.weight_list = reward_weight
        self.eps = eps
        self.best_reward = best_reward

        # Parameters set in the grid search
        self.lr = None
        self.gamma = None
        self.reward_weight = None
        self. optimizers = dict()

        # Training variables
        self.obs = None
        self.episode_reward = 0
        self.done = False
        self.step = 0
        self.SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        self.saved_actions = []
        self.rewards = []

        # Set device
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)

        # Initialize lists for logging, it contains: avg_reward, avg_steps per episode
        self.log_dict = HyperParams.LOG_DICT.value
        # PAth for save the model and the log
        self.file_path = FilePaths.TEST_DIR.value +\
            self.env.env_name + '/' +\
            self.env.detection_alg + '/'

        # Print model architecture
        print("*", "-"*18, " Model Architecture ", "-"*18)
        print("* State dimension: ", self.state_dim)
        print("* Hidden layer 1 size: ", self.hidden_size_1)
        print("* Hidden layer 2 size: ", self.hidden_size_2)
        print("* Action dimension: ", self.action_dim, "*",
              self.env.graph.number_of_nodes(), "=",
              self.action_dim*self.env.graph.number_of_nodes())

    ############################################################################
    #                            GRID SEARCH                                   #
    ############################################################################
    def grid_search(self) -> None:
        """Perform grid search on the hyperparameters"""
        for lr in self.lr_list:
            for gamma in self.gamma_list:
                for reward_weight in self.weight_list:
                    # Change Hyperparameters
                    self.reset_hyperparams(lr, gamma, reward_weight)
                    # Configure optimizers with the current learning rate
                    self.configure_optimizers()
                    # Training
                    log = self.training()
                    # Save results
                    file_path = self.file_path +\
                        f"lr-{lr}/gamma-{gamma}/reward_weight-{reward_weight}/"
                    self.save_plots(log, file_path)
                    gc.collect()

    def reset_hyperparams(self, lr: float, gamma: float, reward_weight: float) -> None:
        """
        Reset hyperparameters
        
        Parameters
        ----------
        lr : float
            Learning rate
        gamma : float
            Gamma parameter
        reward_weight : float
            Reward weight
        """
        self.lr = lr
        self.gamma = gamma
        self.reward_weight = reward_weight
        self.env.weight = reward_weight
        self.print_hyperparams()

        self.log_dict['train_reward'] = list()
        self.log_dict['train_steps'] = list()
        self.log_dict['train_avg_reward'] = list()
        self.log_dict['a_loss'] = list()
        self.log_dict['v_loss'] = list()
        self.saved_actions = []
        self.rewards = []

        self.obs = None
        self.episode_reward = 0
        self.done = False
        self.step = 0
        self.optimizers = dict()

    def print_hyperparams(self):
        print("*", "-"*18, "Model Hyperparameters", "-"*18)
        print("* Learning rate: ", self.lr)
        print("* Gamma parameter: ", self.gamma)
        print("* Reward weight: ", self.reward_weight)
        print("* Value for clipping the loss function: ", self.eps)

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
            # Change Target Node every 10 episodes
            # if i_episode % 10 == 0 and i_episode != 0:
            #    self.env.change_target_node()
            # Change Target Community every 100 episodes
            # if i_episode % 100 == 0 and i_episode != 0:
            #     self.env.change_target_community()
            # TEST: change target node and community every episode
            self.env.change_target_community()

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
                self.save_checkpoint(
                    self.env.env_name, self.env.detection_alg)
                self.best_reward = self.episode_reward

            # Log
            self.log_dict['train_reward'].append(self.episode_reward)
            self.log_dict['train_steps'].append(self.step)
            self.log_dict['train_avg_reward'].append(
                self.episode_reward/self.step)
            self.log_dict['a_loss'].append(self.a_loss)
            self.log_dict['v_loss'].append(self.v_loss)

        self.log(self.log_dict, self.env.env_name, self.env.detection_alg)
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
        # print("Action:", action_rl)
        torch.cuda.empty_cache()
        # Take action in environment
        self.obs, reward, self.done = self.env.step(action_rl)
        # Print rewiring action if we are testing
        if test:
            # Define the edge to be rewired, also his reverse
            edge = (self.env.node_target, action_rl)
            edge_reverse = (action_rl, self.env.node_target)
            if edge in self.env.possible_actions or\
                    edge_reverse in self.env.possible_actions:
                # Remove edge
                if edge in self.env.graph.edges() or\
                        edge_reverse in self.env.graph.edges():
                    print(f"Remove: {edge} | Reward: {reward}")
                # Add edge
                if edge not in self.env.graph.edges() or\
                        edge_reverse not in self.env.graph.edges():
                    print(f"Add: {edge} | Reward: {reward}")
            # Node is isolated from community
            if self.done:
                print(f"Node {self.env.node_target} is isolated from " +
                      f"community {self.env.community_target}")
        else:
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
        # print(action)
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

        # calculate the true value using rewards returned from the environment
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

        mean_a_loss = torch.stack(policy_losses).mean().item()
        mean_v_loss = torch.stack(value_losses).mean().item()

        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
        return mean_a_loss, mean_v_loss

    ############################################################################
    #                               TEST                                       #
    ############################################################################
    def test(self):
        """Hide a given node from a given community"""
        # Load best performing model
        self.load_checkpoint(self.env.env_name, self.env.detection_alg)
        # Set model in evaluation mode
        self.policy.eval()
        # Get the PyG graph
        self.obs = self.env.reset()
        # self.episode_reward = 0
        # self.done = False
        # self.step = 0
        while not self.done:
            self.rewiring(test=True)

    ############################################################################
    #                            CHECKPOINTING                                 #
    ############################################################################
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
        Utils.save_training(
            log,
            self.env.env_name,
            self.env.detection_alg,
            file_path)
        Utils.plot_training(
            log,
            self.env.env_name,
            self.env.detection_alg,
            file_path)

    def save_checkpoint(
            self,
            env_name: str = 'default',
            detection_alg: str = 'default',
            log_dir: str = FilePaths.TEST_DIR.value):
        """Save checkpoint"""
        log_dir = self.file_path +\
            f"lr-{self.lr}/gamma-{self.gamma}/reward_weight-{self.reward_weight}/"
        # Check if the directory exists, otherwise create it
        Utils.check_dir(log_dir)
        path = f'{log_dir}/{env_name}_{detection_alg}.pth'
        checkpoint = dict()
        checkpoint['model'] = self.policy.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(
            self,
            env_name: str = 'default',
            detection_alg: str = 'default',
            log_dir: str = FilePaths.LOG_DIR.value):
        """Load checkpoint
        
        Parameters
        ----------
        env_name : str, optional
            Environment name, by default 'default'
        detection_alg : str, optional
            Detection algorithm name, by default 'default'
        log_dir : str, optional
            Path to the log directory, by default FilePaths.LOG_DIR.value
        """
        log_dir = log_dir + env_name  # + '/' + detection_alg
        path = f'{log_dir}/{env_name}_{detection_alg}.pth'
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['model'])
        for key, _ in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(
            self,
            log_dict: dict,
            env_name: str = 'default',
            detection_alg: str = 'default',
            log_dir: str = FilePaths.LOG_DIR.value):
        """Log data
        
        Parameters
        ----------
        log_dict : dict
            Dictionary containing the data to be logged
        env_name : str, optional
            Environment name, by default 'default'
        detection_alg : str, optional
            Detection algorithm name, by default 'default'
        log_dir : str, optional
            Path to the log directory, by default FilePaths.LOG_DIR.value
        """
        log_dir = log_dir + env_name  # + '/' + detection_alg
        Utils.check_dir(log_dir)
        path = f'{log_dir}/{env_name}_{detection_alg}.pth'
        torch.save(log_dict, path)
