# Agent

This Python module defines an agent class for reinforcement learning tasks. The agent uses the Advantage Actor-Critic (A2C) algorithm to learn policies for navigating a graph-based environment. The code is structured as follows:

## Dependencies

- The code imports various Python libraries and modules, including PyTorch, NetworkX, and others, necessary for reinforcement learning and graph-based tasks.

## Agent Class Definition

- `Agent` is the main class defined in this module. It is responsible for initializing and training an agent in a given environment.

- The agent is initialized with a graph-based environment, hyperparameters, and neural network architecture parameters.

- It contains methods for training, testing, hyperparameter grid search, and checkpointing.

- The agent uses the A2C algorithm to learn policies and perform actions in the environment.

- It also provides functionality for saving training logs and plotting training results.

- The agent can be configured with different sets of hyperparameters and tested on various scenarios within the environment.

## Training

- The `training` method trains the agent on the environment. It uses a training loop where the agent interacts with the environment, computes actor and critic losses, and updates the policy network.

- During training, the agent periodically changes the target node and target community within the environment.

- Training logs, including rewards, steps, and losses, are collected and stored.

## Testing

- The `test` method evaluates the agent's performance using specific hyperparameters. It loads a pre-trained model and tests the agent in a controlled scenario within the environment.

## Hyperparameter Grid Search

- The `grid_search` method performs a grid search over various hyperparameters, training the agent with different combinations of parameters.

- For each combination, the agent is trained, and the results are saved in separate folders.

## Checkpointing

- The agent supports checkpointing, allowing it to save and load trained models, optimizers, and training logs.

## Agent Information and Printing

- The agent class includes methods for printing information about the agent's architecture and hyperparameters.

- It also provides methods for printing the current set of hyperparameters during training and testing.

- The code is organized to facilitate experimentation with different environments, hyperparameters, and scenarios while training reinforcement learning agents.
