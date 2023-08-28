"""Module for the agent class"""
import sys
sys.path.append('../../')

from src.agent.a2c.network import ActorCritic
from src.agent.a2c.memory import Memory
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

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

    def select_action(self, state: torch.Tensor, memory: Memory)-> list:
        """
        Select an action given the current state

        Parameters
        ----------
        state : _type_
            state
        memory : Memory
            Memory object

        Returns
        -------
        action: torch.Tensor
            Action to take
        """
        #  # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # ! OLD
        # return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
        
        # ! NEW
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state, memory)
        # memory.states.append(state)
        # memory.actions.append(action)
        # memory.logprobs.append(action_logprob)
        return action.tolist()
            
    def update(self, memory: Memory):
        """
        Update the policy

        Parameters
        ----------
        memory : Memory
            Memory object
        """
        rewards = []
        discounted_reward = 0
        # Compute the Monte Carlo estimate of the rewards for each time step in 
        # the episode. This involves iterating over the rewards in reverse order 
        # and computing the discounted sum of rewards from each time step to the 
        # end of the episode. 
        # The resulting rewards are then normalized by subtracting the mean
        # and dividing by the standard deviation.
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0

            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # ! OLD
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # ! NEW
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Prepares the data for training the policy network. 
        # The memory object contains lists of states, actions, log probabilities, 
        # and rewards for each time step in the episode.
        # ! OLD
        # old_states = memory.states
        # old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device), 1).detach()
        # old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(self.device).detach()
        # ! NEW
        # Find the maximum size of the tensors in memory.states
        max_size = max([s.size() for s in memory.states])
        # Pad all tensors to the maximum size
        padded_states = [pad(s, (-1, max_size[1]-s.size(1), 0, max_size[0]-s.size(0))) for s in memory.states]
        # padded_states = pad_sequence(memory.states, batch_first=True, padding_value=-1)
        
        old_states = torch.squeeze(torch.stack(padded_states, dim=1)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=1)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=1)).detach().to(self.device)
                
        # Optimize policy for K epochs
        for i in range(self.K_epochs):
            
            # The loss function is computed using the ratio of the probabilities 
            # of the actions under the new and old policies, multiplied by the 
            # advantage of taking the action. The advantage is the difference 
            # between the discounted sum of rewards and the estimated value of 
            # the state under the current policy. The loss is also augmented 
            # with a term that encourages the policy to explore different actions.
            
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages
            
            # Final loss
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            if (i+1) % 5 == 0 or i == 0:
                print('Epoches {} \t loss: {} \t '.format(i+1, loss.mean()))

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        # Clear memory
        memory.clear_memory()