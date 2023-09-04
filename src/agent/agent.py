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
import os


class Agent:
    def __init__(self, state_dim, action_dim, action_std, lr, gamma, eps):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_std = action_std
        self.lr = lr
        # self.betas = betas
        self.gamma = gamma
        self.eps = eps

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.policy = ActorCritic(
            state_dim, action_dim, action_std).to(self.device)
        self.optimizers = self.configure_optimizers()
        
        # action & reward buffer
        # self.memory = Memory()
        self.SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        self.saved_actions = []
        self.rewards = []
        
        # Initialize lists for logging
        self.log_dict = {
            'train_reward': [],
            # Number of steps per episode
            'train_steps': [],
            # Average reward per step
            'train_avg_reward': [],
            # Average Actor loss per episode
            'a_loss': [],
            # Average Critic loss per episode
            'v_loss': [],
            # set max number of training episodes
            'train_episodes': HyperParams.MAX_EPISODES.value,
        }
        
        # Print Hyperparameters on console
        self.print_hyperparams()
        
    def configure_optimizers(self):
        """Configure optimizers
        
        Returns
        -------
        optimizers : dict
            Dictionary of optimizers
        """
        optimizers = dict()
        actor_params = list(self.policy.actor.parameters())
        critic_params = list(self.policy.critic.parameters())
        optimizers['a_optimizer'] = torch.optim.Adam(actor_params, lr=self.lr)
        optimizers['c_optimizer'] = torch.optim.Adam(critic_params, lr=self.lr)
        return optimizers

    def select_action(self, state: Data)->List[float]:
        concentration, value = self.policy.forward(state)
        dist = torch.distributions.Dirichlet(concentration)
        action = dist.sample()
        self.saved_actions.append(self.SavedAction(dist.log_prob(action), value))
        return list(action.cpu().numpy())
    
    def print_hyperparams(self):
        """Print hyperparameters"""
        # Print Hyperparameters
        print("*", "-"*18, "Hyperparameters", "-"*18)
        print("* State dimension: ", self.state_dim)
        print("* Action dimension: ", self.action_dim)
        print("* Action standard deviation: ", self.action_std)
        print("* Learning rate: ", self.lr)
        print("* Gamma parameter: ", self.gamma)
        # print("* Number of epochs when updating the policy: ", k_epochs)
        print("* Value for clipping the loss function: ", self.eps)

    def save_checkpoint(
        self,
        env_name: str = 'default',
        detection_alg: str = 'default',
        log_dir: str = FilePaths.TEST_DIR.value):
        """Save checkpoint"""
        log_dir = log_dir + env_name + '/' + detection_alg
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
            log_dir: str = FilePaths.TEST_DIR.value):
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
        log_dir = log_dir + env_name + '/' + detection_alg
        path = f'{log_dir}/{env_name}_{detection_alg}.pth'
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['model'])
        for key, _ in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(
        self, 
        log_dict:dict, 
        env_name: str = 'default',
        detection_alg: str = 'default',
        log_dir: str = FilePaths.TEST_DIR.value):
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
        log_dir = log_dir + env_name + '/' + detection_alg
        path = f'{log_dir}/{env_name}_{detection_alg}.pth'
        torch.save(log_dict, path)
    
    def training_step(self)->Tuple[float, float]:
        """Training step
        
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
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)
            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(
                value, torch.tensor([R]).to(self.device)))#.view(-1,1)))

        # take gradient steps
        self.optimizers['a_optimizer'].zero_grad()
        a_loss = torch.stack(policy_losses).sum()
        mean_a_loss = torch.stack(policy_losses).mean().item()
        a_loss.backward()
        self.optimizers['a_optimizer'].step()

        self.optimizers['c_optimizer'].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        mean_v_loss = torch.stack(value_losses).mean().item()
        v_loss.backward()
        self.optimizers['c_optimizer'].step()

        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
        return mean_a_loss, mean_v_loss
    

    def training(
        self,
        env: GraphEnvironment,
        env_name: str,
        detection_alg: str) -> dict:
        """
        Train the agent on the environment

        Parameters
        ----------
        env : GraphEnvironment
            Environment to train the agent on
        agent : Agent
            Agent to train
        env_name : str
            Name of the environment
        detection_alg : str
            Name of the detection algorithm
        
        Returns
        -------
        log_dict : dict
            Dictionary containing the training logs
        """
        # T = HyperParams.MAX_TIMESTEPS.value  # set max number of timesteps per episode
        # T = env.edge_budget*2
        epochs = trange(self.log_dict['train_episodes'])  # epoch iterator
        best_reward = -np.inf  # set best reward
        self.policy.train()  # set model in train mode

        for i_episode in epochs:
            # print("*" * 20, "Start Episode", i_episode, "*" * 20)
            obs = env.reset()  # initialize environment
            episode_reward = 0
            done = False
            step = 0
            while not done:
                # Select action: return a list of the probabilities of each action
                action_rl = self.select_action(obs)
                # Take action in environment
                obs, reward, done = env.step(action_rl)
                # Update reward
                episode_reward += reward
                # Store the transition in memory
                self.rewards.append(reward)
                step += 1
            # perform on-policy backprop
            a_loss, v_loss = self.training_step()
            # Send current statistics to screen
            epochs.set_description(
                f"Episode {i_episode+1} | Avg Reward: {episode_reward/step:.2f} | Avg Steps: {step} | Actor Loss: {a_loss:.2f} | Critic Loss: {v_loss:.2f}")
            # print("*"*60, "\n")
            # Checkpoint best performing model
            if episode_reward >= best_reward:
                self.save_checkpoint(env_name, detection_alg)
                best_reward = episode_reward
            # Log
            self.log_dict['train_reward'].append(episode_reward)
            self.log_dict['train_steps'].append(step)
            self.log_dict['train_avg_reward'].append(episode_reward/step)
            self.log_dict['a_loss'].append(a_loss)
            self.log_dict['v_loss'].append(v_loss)
            self.log(self.log_dict, env_name, detection_alg)
        return self.log_dict
