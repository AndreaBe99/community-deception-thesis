# Graph Environment

The class `GraphEnvironment` represents an environment where an agent will act, which is a graph with a community. The class has an init method that initializes the object with the percentage of edges to rewire/update. The method also sets up the object's attributes, including the graph, community structure, deception score, detection algorithm, community to hide, and rewards.

- The `get_possible_actions` method returns the possible actions that can be applied to a given graph. The function takes two parameters: a `NetworkX` graph and a list of integers representing the community to hide. The function returns a dictionary of possible actions, whit two keys `ADD` and `REMOVE`, where each key has a list of tuples representing the edges that can be added or removed, respectively.

- The `get_edge_budget` method returns the number of edges that can be added or removed from a given graph, i.e. the number of actions that can be applied to the graph.

- The `get_reward` method returns the reward for a given action. 

- The `plot_graph` method plots the graph using the `matplotlib` library.

- The `delete_repeat_edges` method deletes repeated edges from a Data structure. Indeed, the `NetworkX` library does not allow repeated edges in a graph. The function takes a list of tuples representing the edges of a graph and returns a list of tuples without repeated edges.

- The `reset` method resets the environment to its initial state. 

- The `apply_action` method applies an action to a given graph, i.e. adds or removes an edge from the graph. To avoid to do the same action twice, the function substitutes the action applied with the invalid action `(-1,-1)`.

- The `setup` method sets up the environment with a given graph and community target, and a detection algorithm. 

- The `step` method applies an action to the graph, and compute the detection algorithm on the graph, to compute the deception score, and the nmi score. Then it uses the two scores to compute the reward, and returns the new state, and the reward.
