# Community Detection Algorithms

This Python module contains code for community detection algorithms using CDLIB and iGraph libraries. It provides classes for detecting communities within a graph and offers flexibility in choosing various community detection algorithms. The module imports necessary libraries and allows the computation of community structures in both NetworkX and iGraph graph representations.

## Imports

The module imports the following libraries and modules:

- `sys`: The system module for system-related operations (commented out).
- `DetectionAlgorithmsNames` from `src.utils.utils`: Provides detection algorithm names as an enum.
- `List` from `typing`: Used for typing hints.
- `algorithms`, `NodeClustering` from `cdlib`: CDLIB libraries for community detection.
- `os`: The operating system module for file operations.
- `networkx` as `nx`: Used for NetworkX graph operations.
- `igraph` as `ig`: Used for iGraph graph operations.
- `matplotlib.pyplot` as `plt`: Used for plotting.

## Class: `CommunityDetectionAlgorithm`

This class provides community detection algorithms using the CDLIB library. It includes the following methods:

### `__init__(...)`

- Initializes the `CommunityDetectionAlgorithm` object with the provided algorithm name.

### `compute_community(...)`

- Computes the community partition of a given NetworkX graph based on the selected algorithm.

## Class: `DetectionAlgorithm`

This class offers community detection algorithms using the iGraph library. It includes methods for converting NetworkX graphs to iGraph graphs and for computing various community detection algorithms. The class contains the following methods:

### `__init__(...)`

- Initializes the `DetectionAlgorithm` object with the provided algorithm name.

### `networkx_to_igraph(...)`

- Converts a NetworkX graph to an iGraph graph for compatibility with iGraph's community detection algorithms.

### `compute_community(...)`

- Computes the community partition of a given NetworkX graph based on the selected algorithm.

### `vertexcluster_to_list(...)`

- Converts an iGraph `VertexClustering` object to a list of lists, where each inner list represents a community.

### `plot_graph(...)`

- Plots the graph using iGraph for visualization purposes.

### `compute_louv(...)`, `compute_walk(...)`, `compute_gre(...)`, `compute_inf(...)`, `compute_lab(...)`, `compute_eig(...)`, `compute_btw(...)`, `compute_spin(...)`, `compute_opt(...)`, `compute_scd(...)`

- Compute community detection algorithms using various iGraph methods.

### `write_graph_to_file(...)`

- Writes the graph to a text file, where each line represents an edge in the graph.

### `read_data_from_file(...)`

- Reads data from a file and returns a list of lists, where each inner list represents a community.

## Example Usage

The module includes an example usage section that demonstrates how to create a graph, initialize a community detection algorithm, compute communities, and print the results.

Overall, the module provides a comprehensive set of community detection algorithms using both CDLIB and iGraph libraries, making it suitable for various network analysis tasks.

