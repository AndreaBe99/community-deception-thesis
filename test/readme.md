# Test

This folder contains the files for three different datasets:

- `kar`, $\sim 40$ nodes;
- `words`, $\sim 100$ nodes;
- `vote`, $\sim 1000$ nodes;

For each dataset, we have used three different Detection Algorithms (our $f(\cdot)$ function):

- **Greedy Modularity**;
- **Modularity**;
- **Walk Trap**;

For each combination of dataset and Detection Algorithm, we have performed two different experiments:

- **Node Deception**: Hide a node from its initial community;
  - In this case we test the Agent on:
    - three different values of the *Similarity Coefficient* $\tau = \{0.3, 0.5, 0.8\}$;
    - three different values of the *Budget Multiplier* $\beta = \{1, 2, 3\}$ (budget is equal to $\beta \times \text{avg degree of G}$);
  - Each combination of $\tau$ and $\beta$ is tested for 100 iterations, and we obtain:
    - **Goal**, percentage of achieved goal;
    - **NMI**, Normalized Mutual Information, value between 0 and 1;
    - Average number of **steps** to hide the node;
    - **Time** to hide the node;
- **Community Deception**: Hide a community, i.e. a set of nodes, distributing the nodes in the community among the other communities;
  - In this case we test the Agent on:
    - three different values of the *Similarity Coefficient* $\tau = \{0.3, 0.5, 0.8\}$;
    - three different values of the *Budget Percentage* $\beta = \{1\%, 3\%, 5\%\}$ (budget is equal to $\beta \times \text{number of edges}$);
  - Each combination of $\tau$ and $\beta$ is tested for 100 iterations, and we obtain:
  - **Deception Score**, metric defined in the paper, value between 0 and 1;
  - **NMI**, Normalized Mutual Information, value between 0 and 1;
  - Average number of **steps** to hide the node;
  - **Time** to hide the node;