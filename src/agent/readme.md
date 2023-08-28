# Agent

The file `agent.py` is a Python module for the `Agent` class, that represents an agent that will act in the environment. The class has an init method that initializes the object with the state dimension, action dimension, action standard deviation, learning rate, discount factor, number of epochs, and clipping parameter. The method also sets up the object's attributes, including the memory, device, policy, optimizer, policy_old, and MseLoss.

The module uses the PyTorch library for deep learning and the typing library to specify the types of the input and output parameters. The Agent class provides a convenient way to set up an agent for an environment by creating an object with the necessary hyperparameters. The object's attributes are set up later with the select_action function.

The select_action function takes a state tensor and a memory object as input and returns a list of actions to take. The function uses the policy network to select an action given the current state and adds the state, action, reward, and next state to the memory object. The function then returns the selected action.

## TODO

- [ ] Avoid to pad the tensors in the updated of the Agent class, we can use a adjacency matrix to represent the graph, instead of a list of edges.