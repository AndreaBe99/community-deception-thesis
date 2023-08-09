"""Module for the agent class"""
import sys
sys.path.append('../../')

from src.agent.network import ActorCritic
from src.agent.memory import Memory
from torch import nn

import torch

class Agent:
    def __init__(self, state_dim, action_dim, action_std, lr, gamma, K_epochs, eps_clip):
        self.lr = lr
        # self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.memory = Memory()

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(self.device)
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.policy.parameters()), lr=lr)#, betas=betas)

        self.policy_old = ActorCritic(
            state_dim, action_dim, action_std).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        #  # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # ! OLD
        # return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
        
        # ! NEW
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        return action.item()
            
    def update(self):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # ! OLD
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # ! NEW
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        # # old_states = torch.squeeze(torch.stack(self.memory.states).to(device), 1).detach()
        # ! OLD
        # old_states = self.memory.states
        # old_actions = torch.squeeze(torch.stack(self.memory.actions).to(DEVICE), 1).detach()
        # old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs), 1).to(DEVICE).detach()
        # ! NEW
        old_states = torch.squeeze(torch.stack(
            self.memory.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(
            self.memory.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(
            self.memory.logprobs, dim=0)).detach().to(self.device)
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages
            
            # Final loss
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            if _ % 5 == 0:
                print('Epoches {} \t loss: {} \t '.format(_, loss.mean()))

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        # Clear memory
        self.memory.clear_memory()