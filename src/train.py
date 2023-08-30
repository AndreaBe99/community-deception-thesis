from src.agent.agent import Agent
from src.agent.a2c.memory import Memory
from src.environment.graph_env import GraphEnvironment
from src.utils.utils import HyperParams, FilePaths, Utils
from typing import List, Tuple

import torch

def train(
    env: GraphEnvironment,
    agent: Agent,
    memory: Memory,
    max_episodes: int = HyperParams.MAX_EPISODES.value,
    max_timesteps: int = HyperParams.MAX_TIMESTEPS.value,
    update_timesteps: int = HyperParams.UPDATE_TIMESTEP.value,
    log_interval: int = HyperParams.LOG_INTERVAL.value,
    solved_reward: float = HyperParams.SOLVED_REWARD.value,
    save_model: int = HyperParams.SAVE_MODEL.value,
    log_dir: str = FilePaths.LOG_DIR.value,
    env_name: str = "Default") -> Tuple[List[float], List[int]]:
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
    
    Returns
    -------
    Tuple[List[float], List[int]]
        Average reward for each episode and the length of each episode
    """
    # Logging Variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    episodes_avg_rewards = []
    episodes_length = []
    
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
            #° Running policy_old, return a distribution over the actions
            actions = agent.select_action(state, memory)
            #° Perform the step on the environment, i.e. add or remove an edge
            state, reward = env.step(actions)
            #° Saving reward and is_terminals
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            #° Update policy if its time
            if time_step % update_timesteps == 0:
                print("*", "-" * 13, "Start training the RL agent ", "-" * 13)
                agent.update(memory)
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
        print("* Average Episode Reward: ", avg_episode_reward/avg_episode_timesteps)
        print("* Episode Timesteps: ", avg_episode_timesteps)
        episodes_avg_rewards.append(avg_episode_reward/avg_episode_timesteps)
        episodes_length.append(avg_episode_timesteps)
        avg_episode_reward = 0
        avg_episode_timesteps = 0
        # Update the average length of episodes
        avg_length += t+1
        #° Stop training if avg_reward > solved_reward
        if (episode % log_interval) > 10 and running_reward / avg_length > solved_reward:
            print("#"*20, "Solved", "#"*20)
            print("Running reward: ", running_reward/avg_length)
            torch.save(agent.policy.state_dict(), 
                './' + log_dir + '{}_rl_solved.pth'.format(env_name))
            break
        #° Save model 
        if episode % save_model == 0:
            print("*", "-"*19, "\tSaving Model  ", "-"*19)
            torch.save(agent.policy.state_dict(),
                './' + log_dir + '{}_rl.pth'.format(env_name))
            torch.save(agent.policy.actor.graph_encoder.state_dict(),
                './' + log_dir + '{}_rl_graph_encoder_actor.pth'.format(env_name))
            torch.save(agent.policy.critic.graph_encoder_critic.state_dict(),
                './' + log_dir + '{}_rl_graph_encoder_critic.pth'.format(env_name))
        #° Log details
        if episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))
            print("*", "-"*56)
            print('* Episode {}\t avg log length: {}\t avg log reward: {:.2f}'.format(episode,avg_length, running_reward/avg_length))
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
        "state_dim": agent.state_dim,
        "action_dim": agent.action_dim,
        "action_std": agent.action_std,
        "lr": agent.lr,
        "gamma": agent.gamma,
        "K_epochs": agent.K_epochs,
        "eps_clip": agent.eps_clip,
    }
    # Save lists and hyperparameters in a json file
    Utils.write_results_to_json(episodes_avg_rewards, episodes_length, hyperparams_dict, env_name)
    
    return episodes_avg_rewards, episodes_length