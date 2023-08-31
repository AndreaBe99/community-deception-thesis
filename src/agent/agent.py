"""Module for the agent class"""
from src.agent.a2c.network import ActorCritic
from src.agent.a2c.memory import Memory
from src.environment.graph_env import GraphEnvironment
from src.utils.utils import HyperParams, FilePaths, Utils

from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch import nn

import torch


class Agent:
    def __init__(self, state_dim, action_dim, action_std, lr, gamma, K_epochs, eps_clip):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_std = action_std
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

    def select_action(self, state: Data, memory: Memory,) -> list:
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
        # ! OLD
        # return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
        
        # ! NEW
        with torch.no_grad():
            action = self.policy_old.act(state, memory)
        return action.cpu().data.numpy().flatten()
    
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
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Prepares the data for training the policy network. 
        # The memory object contains lists of states, actions, log probabilities, 
        # and rewards for each time step in the episode.
        
        # Each state is a PyG Data object
        old_states = Batch.from_data_list(memory.states).to(self.device)
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
            
            # Final loss: first term is Actor Loss, second term is Critic Loss
            act_loss = -torch.min(surr1, surr2) 
            crt_loss = self.MseLoss(state_values, rewards) * 0.5
            ent_loss = dist_entropy * 0.01
            loss = act_loss + crt_loss - ent_loss # Want to maximize
            
            if (i+1) % 5 == 0 or i == 0:
                print('* Epoches {} \t loss: {} \t '.format(i+1, loss.mean()))
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        # Clear memory
        memory.clear_memory()
    
    def train(
        self,
        env: GraphEnvironment,
        memory: Memory,
        max_episodes: int = HyperParams.MAX_EPISODES.value,
        max_timesteps: int = HyperParams.MAX_TIMESTEPS.value,
        update_timesteps: int = HyperParams.UPDATE_TIMESTEP.value,
        log_interval: int = HyperParams.LOG_INTERVAL.value,
        solved_reward: float = HyperParams.SOLVED_REWARD.value,
        save_model: int = HyperParams.SAVE_MODEL.value,
        log_dir: str = FilePaths.LOG_DIR.value,
        env_name: str = "Default") -> None:
        """
        Function to train the agent

        Parameters
        ----------
        max_episodes : int, optional
            Number of episodes,by default HyperParams.MAX_EPISODES.value
        max_timesteps : int, optional
            Number of timesteps, by default HyperParams.MAX_TIMESTEPS.value
        update_timesteps : int, optional
            Number of timesteps to update the policy, by default HyperParams.UPDATE_TIMESTEP.value
        log_interval : int, optional
            Interval for logging, by default HyperParams.LOG_INTERVAL.value
        solved_reward : float, optional
            Stop training if the reward is greater than this value, by default HyperParams.SOLVED_REWARD.value
        save_model : int, optional
            Each save_model episodes save the model, by default HyperParams.SAVE_MODEL.value
        log_dir : str, optional
            Directory for logging, by default FilePaths.LOG_DIR.value
        env_name : str, optional
            Environment name, by default "Default"
        """
        # Logging Variables
        running_reward = 0
        avg_length = 0
        time_step = 0

        episodes_avg_rewards = []
        episodes_length = []

        # ! Comment this line if you are on Kaggle or Colab
        log_dir = './' + log_dir

        # Training loop
        for episode in range(1, max_episodes + 1):
            # Reset the environment at each episode, state is a PyG Data object
            state = env.reset()
            done = False
            print("*" * 20, "Start Episode", episode, "*" * 20)

            avg_episode_reward = 0
            avg_episode_timesteps = 0
            for t in range(max_timesteps):
                # If the budget for the graph rewiring is exhausted, stop the episode
                if env.used_edge_budget == env.edge_budget - 1:
                    print("*", "-" * 19, "Budget exhausted", "-" * 19)
                    done = True
                time_step += 1
                # ° Running policy_old, return a distribution over the actions
                actions = self.select_action(state, memory)
                # ° Perform the step on the environment, i.e. add or remove an edge
                state, reward = env.step(actions)
                # ° Saving reward and is_terminals
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                # ° Update policy if its time
                if time_step % update_timesteps == 0:
                    print("*", "-" * 13, "Start training the RL agent ", "-" * 13)
                    self.update(memory)
                    memory.clear_memory()
                    time_step = 0
                    print("*", "-"*14, "End training the RL agent ", "-"*14)
                # Add the reward to the running reward
                running_reward += reward
                # Update the average episode reward and timesteps
                avg_episode_reward += reward
                avg_episode_timesteps += 1
                # Check if the episode is done
                if done:
                    break
            # Show Average Episode Reward and Timesteps
            print("* Average Episode Reward: ",
                avg_episode_reward/avg_episode_timesteps)
            print("* Episode Timesteps: ", avg_episode_timesteps)
            episodes_avg_rewards.append(avg_episode_reward/avg_episode_timesteps)
            episodes_length.append(avg_episode_timesteps)
            avg_episode_reward = 0
            avg_episode_timesteps = 0
            # Update the average length of episodes
            avg_length += t+1
            # ° Stop training if avg_reward > solved_reward
            if (episode % log_interval) > 10 and running_reward / avg_length > solved_reward:
                print("#"*20, "Solved", "#"*20)
                print("Running reward: ", running_reward/avg_length)
                torch.save(self.policy.state_dict(),
                        log_dir + '{}_rl_solved.pth'.format(env_name))
                break
            # ° Save model
            if episode % save_model == 0:
                print("*", "-"*19, "\tSaving Model  ", "-"*19)
                torch.save(self.policy.state_dict(),
                        log_dir + '{}_rl.pth'.format(env_name))
                torch.save(self.policy.actor.graph_encoder.state_dict(),
                        log_dir + '{}_rl_graph_encoder_actor.pth'.format(env_name))
                torch.save(self.policy.critic.graph_encoder_critic.state_dict(),
                        log_dir + '{}_rl_graph_encoder_critic.pth'.format(env_name))
            # ° Log details
            if episode % log_interval == 0:
                avg_length = int(avg_length / log_interval)
                running_reward = int((running_reward / log_interval))
                print("*", "-"*56)
                print('* Episode {}\t avg log length: {}\t avg log reward: {:.2f}'.format(
                    episode, avg_length, running_reward/avg_length))
                print("*", "-"*56)
                running_reward = 0
                avg_length = 0

            if env.debug:
                env.plot_graph()
            print("*"*57, "\n")

        hyperparams_dict = {
            "max_episodes": max_episodes,
            "max_timesteps": max_timesteps,
            "update_timesteps": update_timesteps,
            "log_interval": log_interval,
            "solved_reward": solved_reward,
            "save_model": save_model,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_std": self.action_std,
            "lr": self.lr,
            "gamma": self.gamma,
            "K_epochs": self.K_epochs,
            "eps_clip": self.eps_clip,
        }
        # Save lists and hyperparameters in a json file
        print("*", "-"*18, "Saving results", "-"*18)
        Utils.write_results_to_json(
            episodes_avg_rewards, episodes_length, hyperparams_dict, env_name)
        
        print("*", "-"*18, "Plotting results", "-"*18)
        # Plot the average reward per episode
        Utils.plot_avg_reward(episodes_avg_rewards, episodes_length, env_name)
