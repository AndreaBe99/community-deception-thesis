# Utils Module

## SIMILARITY

### Module Description

This module defines two classes, `CommunitySimilarity` and `GraphSimilarity`, that are used for computing similarity between communities and graphs, respectively. These classes provide different similarity metrics for assessing the similarity between lists of integers and graphs.

### Imports

The module imports the following libraries and modules:

- `typing`: Used for type hinting.
- `networkx`: A library for working with graphs.
- `numpy`: A library for numerical operations.
- `src.utils.utils.SimilarityFunctionsNames`: Importing constants for similarity function names.

### CommunitySimilarity Class

The `CommunitySimilarity` class is responsible for computing the similarity between two lists of integers representing communities.

#### Constructor

- `__init__(self, function_name: str)`: Initializes the class with a specified similarity function name.

#### Methods

- `select_similarity_function(self) -> Callable`: Selects the similarity function based on the provided function name.
- `jaccard_similarity(a: List[int], b: List[int]) -> float`: Computes the Jaccard similarity between two lists.
- `overlap_similarity(a: List[int], b: List[int]) -> float`: Computes the Overlap similarity between two lists.
- `sorensen_similarity(a: List[int], b: List[int]) -> float`: Computes the Sorensen similarity between two lists.

### GraphSimilarity Class

The `GraphSimilarity` class is responsible for computing the similarity between two graphs.

#### Constructor

- `__init__(self, function_name: str)`: Initializes the class with a specified similarity function name.

#### Methods

- `select_similarity_function(self) -> Callable`: Selects the similarity function based on the provided function name.
- `graph_edit_distance(self, g: nx.Graph, h: nx.Graph) -> float`: Computes the graph edit distance between two graphs and normalizes it.
- `jaccard_similarity_1(self, g: nx.Graph, h: nx.Graph) -> float`: Computes the Jaccard Similarity between two graphs.
- `jaccard_similarity_2(self, g: nx.Graph, h: nx.Graph) -> float`: Computes a second version of the Jaccard Similarity between two graphs.

### Main Section

- The main section of the code contains test cases for the `GraphSimilarity` class.
- It creates two sample graphs and calculates their similarity using the Jaccard similarity function.
- The results are printed to the console.

These classes provide a modular way to calculate different similarity metrics for communities and graphs, allowing for flexibility in assessing their similarity.


## TEST

This Python module contains code for testing the performance of an agent in a graph-based research project. It evaluates the agent's performance and compares it with several baseline algorithms:

- Random Hiding
- Degree Hiding
- Roam Heuristic

The module includes the following components:

### Imports

The module imports various libraries and modules, including `trange` for progress tracking and `time` for measuring execution time.

### Functions

#### `test(...)`

This function evaluates the performance of the agent and baseline algorithms. It performs the following tasks:

- Initializes a log dictionary to store evaluation metrics.
- Iterates through a specified number of evaluation episodes.
- For each episode, changes the target community and node.
- Tests the agent's performance on the current episode and records metrics.
- Applies the baseline algorithms (Random Hiding, Degree Hiding, and Roam Heuristic) to hide the target node from the community and records their performance metrics.
- Saves the evaluation results to the log dictionary.

#### `save_metrics(...)`

This utility function is used to save metrics for a specific algorithm (e.g., agent, Random Hiding, Degree Hiding, Roam Heuristic) in the log dictionary. It appends values such as goal achievement, NMI (Normalized Mutual Information), execution time, and steps to the respective metrics lists.

The module primarily focuses on evaluating the agent's performance in comparison to baseline algorithms and recording relevant metrics for analysis. The evaluation results are saved to a log dictionary and can be further analyzed or visualized.



## UTILS

### Module Description

This module defines several classes and enums that store file paths, hyperparameters, and utility functions for a graph-based research project. The module is structured as follows:

#### Enums

1. `FilePaths`: Enum class for storing file paths related to data and models.
2. `DetectionAlgorithmsNames`: Enum class for storing the names of detection algorithms.
3. `SimilarityFunctionsNames`: Enum class for storing the names of similarity functions.
4. `HyperParams`: Enum class for storing hyperparameters used in the project.

#### Classes

5. `Utils`: Class that contains static utility functions for various tasks.

### Imports

The module imports various libraries and modules, including `matplotlib`, `networkx`, `numpy`, `scipy`, `json`, and `os`, for data manipulation, visualization, and file handling.

### Enums

The module defines several enums to store constants and configuration options.

#### `FilePaths` Enum

- Stores file paths for datasets, logs, and tests.
- Provides options for different environments, such as local, Kaggle, and Google Colab.
- Specifies dataset file paths for various graph datasets.

#### `DetectionAlgorithmsNames` Enum

- Stores the names of detection algorithms as constants.

#### `SimilarityFunctionsNames` Enum

- Stores the names of similarity functions for community and graph similarity calculations.

#### `HyperParams` Enum

- Stores a wide range of hyperparameters for the environment, graph encoder, agent, training, evaluation, and graph generation.
- Contains hyperparameters for learning rates, discount factors, network architecture, and more.

### `Utils` Class

The `Utils` class contains various static utility methods used throughout the project:

#### Data Import

- `import_mtx_graph(file_path: str) -> nx.Graph`: Imports a graph from a .mtx file using SciPy and returns it as an `nx.Graph`.

#### Graph Generation

- `generate_lfr_benchmark_graph(...) -> Tuple[nx.Graph, str]`: Generates a synthetic graph using the LFR benchmark for community detection algorithms and saves it to a file.
  
#### Directory Handling

- `check_dir(path: str)`: Checks if a directory exists and creates it if it doesn't.

#### Training and Evaluation

- `plot_training(...)`: Plots training results, including reward and loss.
- `get_new_community(...) -> List[int]`: Searches for the new community containing the target node after deception.
- `check_goal(...) -> int`: Checks if the goal of hiding the target node was achieved during the training episode.
- `save_test(...)`: Saves and plots the results of evaluation metrics for different algorithms.

These classes and enums provide a structured and configurable way to manage file paths, hyperparameters, and utility functions for the graph-based research project.
