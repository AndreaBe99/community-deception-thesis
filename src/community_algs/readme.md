# Community Algorithms

## Community Detection Algorithms

The file `detection_algs.py` is a Python module for **community detection** algorithms. It imports the necessary libraries and defines a class called `DetectionAlgorithm`. This class has an init method that initializes the object with the name of the algorithm. It also has 3 methods

- The `networkx_to_igraph` method converts a `NetworkX` graph to an `iGraph` graph, which allows the use of iGraph's community detection algorithms. The converted graph is stored in the object's `ig_graph` attribute.

- The `compute_community` method computes the community detection algorithm for a given graph and returns a list of lists of integers representing the communities.

- The `vertexcluster_to_list` method converts an `iGraph` vertex cluster to a list of lists of integers representing the communities.

- The `plot_graph` method plots the graph using the `matplotlib` library.

- Then we have different methods that allow us to calculate the structure of the communities through a specific algorithm.

## Metrics

### Deception Score

The file `deception_score.py` is a Python module for calculating the **deception score** of a community, a value that measures how well a community is hidden in a graph, and it can be between 0 and 1, where 1 means that the community is completely hidden and 0 means that the community is completely visible.

It defines a class called `DeceptionScore` that has an init method that initializes the object with a list of integers representing the target community, i.e., the community that we want to hide in the graph. 
The class also has two static methods called `recall` and `precision` that calculate the recall and precision scores of a given community, respectively. The recall score measures the proportion of members in the target community that are also in the detected community, while the precision score measures the proportion of members in the detected community that are also in the target community. Finally, the class has a method called `compute_deception_score` that calculates the deception score of a community detection algorithm given the community structure and the number of connected components in the graph.

### Node Safeness

Let $G=(V,E)$ be a network, $C \subset V$ a community, and $u \in C$ a member of $C$. The safeness $\sigma(u,C)$ of $u$ in $G$ is defined as:
$$\begin{equation} \sigma ({u},{\mathcal{C}}):=\frac{1}{2}\frac{|{V}_{\mathcal{C}}^{u}|-|E(u,\mathcal{C})|}{|\mathcal{C}|-1}+\frac{1}{2} \frac{|\widetilde{E}(u,\mathcal{C})|}{deg(u)}, \end{equation}$$
where $V^u_C ⊆ C$ is the set of nodes reachable from $u$ passing only via nodes in $C$, and we indicate with $E(u,\bar{V})$ (resp., $\tilde{E}(u,\bar{V})$) the set of intra-community (resp., inter-community) edges for a node $u \in \bar{V}$.

### Permanece

The formulation of permanence is based on three factors: 

1. the internal pull $I(v)$ , denoted by the internal connections of a node $v$ within its own community; 
2. maximum external pull $E_{max}(v)$, denoted by the maximum connections of $v$ to its neighboring communities; 
3. internal clustering coefficient of $v$, $C_{in}(v)$, denoted by the fraction of actual and possible number of edges among the internal neighbors of $v$. 
The above three factors are then suitably combined to obtain the permanence of $v$ as
$$\begin{equation*} \text {Perm}(v,G) = \frac {I(v)}{E_{\text {max}}(v)}\times \frac {1}{\text {deg}(v)} - \big (1 - C_{\text {in}}(v)\big).\tag{2}\end{equation*}$$
This metric indicates that a vertex would remain in its own community as long as its internal pull is greater than the external pull or its internal neighbors are densely connected to each other, hence forming a near clique.

### Normalized Mutual Information (NMI)

The file `nmi.py` is a Python implementation of a class called `NMI` that calculates the **normalized mutual information** between two sets of communities.

The `NMI` class has three methods:

- The `calculate_confusion_matrix` method takes two lists of communities as input and returns a confusion matrix as a `Counter` object. The confusion matrix is a square matrix that shows the number of nodes that are in both communities. The method iterates over each pair of communities and computes the intersection of the two communities using the set data type. The intersection is then stored in the confusion matrix using a tuple of indices as the key.

**TODO**:
  - [ ] Avoid to process the same community twice


- The `calculate_sums` method takes a confusion matrix as input and returns the row sums, column sums, and total sum of the matrix as `Counter` objects. The method iterates over each element of the confusion matrix and updates the row sums, column sums, and total sum accordingly.

- The `compute_nmi` method takes two lists of communities as input and returns the normalized mutual information between the two sets of communities as a float. The method first calculates the confusion matrix using the `calculate_confusion_matrix` method, and then calculates the row sums, column sums, and total sum of the matrix using the `calculate_sums` method. The method then iterates over each element of the confusion matrix and computes the mutual information between the two communities using the formula for mutual information. The mutual information is then normalized using the formula for normalized mutual information.
