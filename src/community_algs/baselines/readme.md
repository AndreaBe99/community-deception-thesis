# Baselines Algorithms

## Degree Hiding

This Python module contains code for a Degree Hiding algorithm used in a graph-based research project. The algorithm aims to hide a target node from a target community within a given graph by rewiring its edges.

### Imports

The module imports the following libraries and modules:

- `GraphEnvironment` from `src.environment.graph_env`: This class provides the environment for the algorithm's operation.
- `DetectionAlgorithmsNames` and `Utils` from `src.utils.utils`: These modules provide detection algorithm names and utility functions.
- `CommunityDetectionAlgorithm` from `src.community_algs.detection_algs`: This class represents a community detection algorithm.
- `networkx` as `nx`: This library is used for graph operations.
- `List` and `random` from the `typing` module: These are used for type hints and randomization.

### Class: `DegreeHiding`

This class implements the Degree Hiding algorithm, which hides a target node from a target community by rewiring its edges. It includes the following methods:

#### `__init__(...)`

- Initializes the DegreeHiding object with the provided parameters, including the environment, number of steps, target community, and detection algorithm.

#### `hide_target_node_from_community(...)`

- Hides the target node from the target community by rewiring its edges.
- Chooses between adding or removing an edge based on the node with the highest degree.
- Continues the rewiring process until the specified number of steps is reached or the goal is achieved (target community is a subset of the new community).
- Returns the modified graph and new community structure.

### Example Usage

The module includes an example usage section. It demonstrates how to create a karate club graph, select a target community and node, and apply the Degree Hiding algorithm to hide the target node from the community. The original and new community structures are then printed for comparison.

Overall, the module provides the Degree Hiding algorithm as a means of hiding a target node from a community within a graph. It can be used to test and evaluate the algorithm's performance on different graphs and scenarios.

## Random Hiding

This Python module contains code for a Degree Hiding algorithm used in a graph-based research project. The algorithm aims to hide a target node from a target community within a given graph by rewiring its edges.

### Imports

The module imports the following libraries and modules:

- `GraphEnvironment` from `src.environment.graph_env`: This class provides the environment for the algorithm's operation.
- `DetectionAlgorithmsNames` and `Utils` from `src.utils.utils`: These modules provide detection algorithm names and utility functions.
- `CommunityDetectionAlgorithm` from `src.community_algs.detection_algs`: This class represents a community detection algorithm.
- `networkx` as `nx`: This library is used for graph operations.
- `List` and `random` from the `typing` module: These are used for type hints and randomization.

### Class: `DegreeHiding`

This class implements the Degree Hiding algorithm, which hides a target node from a target community by rewiring its edges. It includes the following methods:

#### `__init__(...)`

- Initializes the DegreeHiding object with the provided parameters, including the environment, number of steps, target community, and detection algorithm.

#### `hide_target_node_from_community(...)`

- Hides the target node from the target community by rewiring its edges.
- Chooses between adding or removing an edge based on the node with the highest degree.
- Continues the rewiring process until the specified number of steps is reached or the goal is achieved (target community is a subset of the new community).
- Returns the modified graph and new community structure.

### Example Usage

The module includes an example usage section. It demonstrates how to create a karate club graph, select a target community and node, and apply the Degree Hiding algorithm to hide the target node from the community. The original and new community structures are then printed for comparison.

Overall, the module provides the Degree Hiding algorithm as a means of hiding a target node from a community within a graph. It can be used to test and evaluate the algorithm's performance on different graphs and scenarios.


## Roam Heuristic

This Python module contains code for a Roam Hiding algorithm used in a graph-based research project. The algorithm aims to conceal the importance of a target node in a graph while preserving its influence over the network. The Roam Hiding algorithm is based on the article "Hiding Individuals and Communities in a Social Network."

### Imports

The module imports the following libraries and modules:

- `sys`: The system module for system-related operations.
- `CommunityDetectionAlgorithm`, `Safeness` from `src.community_algs.detection_algs` and `src.community_algs.metrics.safeness`: These modules provide community detection algorithms and metrics for measuring safeness.
- `Utils`, `FilePaths`, `DetectionAlgorithmsNames`, `HyperParams` from `src.utils.utils`: These modules provide utility functions, file paths, detection algorithm names, and hyperparameters.
- `matplotlib.pyplot` as `plt`: This library is used for plotting graphs.
- `networkx` as `nx`: This library is used for graph operations.
- `numpy` as `np`: This library is used for numerical operations.
- `random`: The random module for randomization.

### Class: `RoamHiding`

This class implements the Roam Hiding algorithm, which conceals the importance of a target node by decreasing its centrality without compromising its influence over the network. It includes the following methods:

#### `__init__(...)`

- Initializes the RoamHiding object with the provided graph, target node, and detection algorithm.

#### `get_edge_budget(...)`

- Computes the number of edges to add based on a budget and the graph's size.

#### `roam_heuristic(...)`

- Implements the ROAM heuristic given a budget:
  - Step 1: Removes the link between the target node and its neighbor with the most connections.
  - Step 2: Connects the neighbor to a specified number of nodes, preserving influence.

### Example Usage

The module includes an example usage section that sets up a graph environment, applies the Roam Hiding algorithm, and measures safeness metrics. The original and modified community structures, as well as centrality and NMI metrics, are printed for comparison.

Overall, the module provides the Roam Hiding algorithm for concealing the importance of a target node in a network while maintaining its influence. It also demonstrates its application and evaluation on different graph scenarios.
