# GraphEnvironment Class

## Module Description

This module defines the `GraphEnvironment` class, which represents the environment where an agent will act. The environment is a graph with communities, and the agent's goal is to manipulate the graph while satisfying certain constraints.

## Imports

The module imports various libraries and modules:

- `community_algs.detection_algs`: Importing a community detection algorithm.
- `utils.utils`: Importing utility functions and constants.
- `utils.similarity`: Importing similarity functions.
- `math`, `networkx`, `random`, and `time`: Standard Python libraries for mathematical operations, graph manipulation, randomization, and time handling.
- `typing`: Used for type hinting.

## Class Definition

The `GraphEnvironment` class is defined with several methods and attributes to represent the environment and control agent interactions.

### Constructor

The constructor (`__init__`) initializes the environment with the following parameters:

- `graph_path` (optional): Path to the graph data file.
- `community_detection_algorithm`: Name of the community detection algorithm to use.
- `beta` (optional): Percentage of edges to remove.
- `tau` (optional): Strength of the deception constraint.
- `community_similarity_function` (optional): Name of the community similarity function to use.
- `graph_similarity_function` (optional): Name of the graph similarity function to use.

### Attributes

- `graph`: The current graph representing the environment.
- `original_graph`: A copy of the original graph to restart episodes.
- `old_graph`: The graph state before the agent's action.
- `n_connected_components`: Number of connected components in the graph.
- Various hyperparameters and similarity functions.
- Attributes related to community detection, community manipulation, and budget management.
- Attributes for tracking rewards, penalties, and episode status.
- Attributes for storing possible actions and episode step limits.

### Getter Methods

- `get_edge_budget()`: Computes the edge budget for the graph.
- `get_penalty()`: Computes a penalty based on community and graph distance metrics.
- `get_reward()`: Computes the reward for the agent based on community similarity and deception constraints.
- `get_possible_actions()`: Determines possible actions the agent can take.

### Episode Reset Methods

- `reset()`: Resets the environment to its initial state.
- `change_target_node()`: Changes the target node to remove from the community.
- `change_target_community()`: Changes the target community to hide the node.

### Episode Step Method

- `step(action)`: Executes a step in the environment based on the agent's action, updating the graph, rewards, and episode status.
- `apply_action(action)`: Applies the specified action to the graph, adding or removing edges.

### Environment Information

- `print_env_info()`: Prints information about the environment, including graph details, community detection algorithm, budget, and more.

This class provides a framework for simulating an environment where an agent can manipulate a graph while considering community structure and deception constraints.
