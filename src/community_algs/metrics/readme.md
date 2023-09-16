# Metrics

## Deception Score

The file `deception_score.py` is a Python module for calculating the **deception score** of a community, a value that measures how well a community is hidden in a graph, and it can be between 0 and 1, where 1 means that the community is completely hidden and 0 means that the community is completely visible.

It defines a class called `DeceptionScore` that has an init method that initializes the object with a list of integers representing the target community, i.e., the community that we want to hide in the graph.
The class also has two static methods called `recall` and `precision` that calculate the recall and precision scores of a given community, respectively. The recall score measures the proportion of members in the target community that are also in the detected community, while the precision score measures the proportion of members in the detected community that are also in the target community. Finally, the class has a method called `compute_deception_score` that calculates the deception score of a community detection algorithm given the community structure and the number of connected components in the graph.

## Node Safeness

Let $G=(V,E)$ be a network, $C \subset V$ a community, and $u \in C$ a member of $C$. The safeness $\sigma(u,C)$ of $u$ in $G$ is defined as:
$$\begin{equation} \sigma ({u},{\mathcal{C}}):=\frac{1}{2}\frac{|{V}_{\mathcal{C}}^{u}|-|E(u,\mathcal{C})|}{|\mathcal{C}|-1}+\frac{1}{2} \frac{|\widetilde{E}(u,\mathcal{C})|}{deg(u)}, \end{equation}$$
where $V^u_C ⊆ C$ is the set of nodes reachable from $u$ passing only via nodes in $C$, and we indicate with $E(u,\bar{V})$ (resp., $\tilde{E}(u,\bar{V})$) the set of intra-community (resp., inter-community) edges for a node $u \in \bar{V}$.

## Permanece

The formulation of permanence is based on three factors:

1. the internal pull $I(v)$ , denoted by the internal connections of a node $v$ within its own community;
2. maximum external pull $E_{max}(v)$, denoted by the maximum connections of $v$ to its neighboring communities;
3. internal clustering coefficient of $v$, $C_{in}(v)$, denoted by the fraction of actual and possible number of edges among the internal neighbors of $v$.
The above three factors are then suitably combined to obtain the permanence of $v$ as
$$\begin{equation*} \text {Perm}(v,G) = \frac {I(v)}{E_{\text {max}}(v)}\times \frac {1}{\text {deg}(v)} - \big (1 - C_{\text {in}}(v)\big).\tag{2}\end{equation*}$$
This metric indicates that a vertex would remain in its own community as long as its internal pull is greater than the external pull or its internal neighbors are densely connected to each other, hence forming a near clique.

## Normalized Mutual Information (NMI)

The file `nmi.py` is a Python implementation of a class called `NMI` that calculates the **normalized mutual information** between two sets of communities.

The `NMI` class has three methods:

- The `calculate_confusion_matrix` method takes two lists of communities as input and returns a confusion matrix as a `Counter` object. The confusion matrix is a square matrix that shows the number of nodes that are in both communities. The method iterates over each pair of communities and computes the intersection of the two communities using the set data type. The intersection is then stored in the confusion matrix using a tuple of indices as the key.
