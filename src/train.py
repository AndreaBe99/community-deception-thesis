from torch_geometric.utils import from_networkx

import networkx as nx
import torch
import torch_geometric

import logging

import numpy as np

import sys
sys.path.append("../")

from src.agent.agent import Agent
from src.agent.memory import Memory
from src.environment.graph_env import GraphEnvironment
from src.utils import HyperParams, Utils, FilePaths

# Environment parameters
BETA = HyperParams.BETA.value
KAR_PATH = "../"+FilePaths.KARATE_PATH.value

# Agent parameters
G_IN_SIZE = HyperParams.G_IN_SIZE.value
ACTION_STD = HyperParams.ACTION_STD.value
EPS_CLIP = HyperParams.EPS_CLIP.value
LR = HyperParams.LR.value
GAMMA = HyperParams.GAMMA.value
K_EPOCHS = HyperParams.K_EPOCHS.value

# Hyperparameters for the training loop
SOLVED_REWARD = HyperParams.SOLVED_REWARD.value
LOG_INTERVAL = HyperParams.LOG_INTERVAL.value
MAX_EPISODES = HyperParams.MAX_EPISODES.value
MAX_TIMESTEPS = HyperParams.MAX_TIMESTEPS.value
UPDATE_TIMESTEP = HyperParams.UPDATE_TIMESTEP.value
RANDOM_SEED = None

# Name of the graph environment
ENV_NAME = "karate"



if __name__ == "__main__":
    # Define the environment
    env = GraphEnvironment(BETA)
    # Load the graph from the dataset folder
    kar = Utils.import_mtx_graph(KAR_PATH)
    
    # Print the graph
    print("Info on '{}' graph: ".format(ENV_NAME), kar)
    
    # Community to hide, from community_detection.ipynb file
    community_target = [4, 5, 6, 10, 16]
    # Setup the environment, by default we use the Louvain algorithm for detection
    env.setup(graph=kar, community=community_target, training=True)
    
    # G_IN_SIZE = kar.number_of_nodes()
    # Get list of possible actions
    possible_actions = env.get_possible_actions(kar, community_target)
    NUM_ACTIONS = len(possible_actions["ADD"]) + len(possible_actions["REMOVE"])
    print("Number of possible actions: ", NUM_ACTIONS)
    # Define the agent
    agent = Agent(
        state_dim=G_IN_SIZE, 
        action_dim=NUM_ACTIONS, 
        action_std=ACTION_STD, 
        lr=LR,
        gamma=GAMMA, 
        K_epochs=K_EPOCHS, 
        eps_clip=EPS_CLIP)
    
    if RANDOM_SEED:
        print("Random Seed: {}".format(RANDOM_SEED))
        torch.manual_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
    
    # Define Memory
    memory = Memory()
    
    # Logging Variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    # Training loop
    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        done = False
        
        for t in range(MAX_TIMESTEPS):
            time_step += 1
            # Running policy_old:
            action = agent.select_action(state.edge_index, memory)
            state, reward = env.step(action[0])
            
            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # Update if its time
            if time_step % UPDATE_TIMESTEP == 0:
                print("-*" * 10, "start training the RL agent", "-*" * 10)
                agent.update(memory)
                memory.clear_memory()
                time_step = 0
                print("-*" * 10, "start search the pruning policies", "-*" * 10)
            
            running_reward += reward
            
            # If the budget for the graph rewiring is exhausted, stop the episode
            if env.exhausted_budget:
                print("Budget exhausted")
                done = True

            # if time_step == MAX_TIMESTEPS:
            #    if not done:
            #        print("Max timestep reached, but not done")
            #        reward = -100
            #        done = True
            
            if done:
                break

        avg_length += t
        
        LOG_DIR = FilePaths.LOG_DIR.value + "/"
        # Stop training if avg_reward > solved_reward
        if (episode % LOG_INTERVAL) != 0 and running_reward / (episode % LOG_INTERVAL) > (SOLVED_REWARD):
            print("########## Solved! ##########")
            torch.save(agent.policy.state_dict(), './' + LOG_DIR + 'rl_solved_{}.pth'.format(ENV_NAME))
            break
        
        # Save model every 500 episodes
        if episode % 500 == 0:
            torch.save(agent.policy.state_dict(), './' + LOG_DIR + 'rl_{}.pth'.format(ENV_NAME))
            torch.save(agent.policy.actor.graph_encoder.state_dict(),
                        './' + LOG_DIR +'rl_graph_encoder_actor_{}.pth'.format(ENV_NAME))
            torch.save(agent.policy.critic.graph_encoder_critic.state_dict(),
                        './' + LOG_DIR +  'rl_graph_encoder_critic_{}.pth'.format(ENV_NAME))    
        
        # Log details
        if episode % LOG_INTERVAL == 0:
            avg_length = int(avg_length / LOG_INTERVAL)
            running_reward = int((running_reward / LOG_INTERVAL))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0        