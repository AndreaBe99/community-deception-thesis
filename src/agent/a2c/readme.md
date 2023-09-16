# A2C

The A2C structure is taken from [gnn-rl-for-amod](https://github.com/DanieleGammelli/gnn-rl-for-amod/tree/main).

## Description

The Actor-Critic RL aims to find an optimal policy for the agent in an environment using two components: Actor and Critic.

- **Actor**: The Actor learns an optimal policy by exploring the environment

- **Critic**: The Critic assesses the value of each action taken by the Actor to determine whether the action will result in a better reward, guiding the Actor for the best course of action to take.

The Actor then uses the feedback from Critic to adjust its policy and make more informed decisions, leading to improved overall performance.

### Value-based vs Policy-based

The Actor-Critic is a combination of value-based, and policy-based methods where the Actor controls how our agent behaves using the Policy gradient, and the Critic evaluates how good the action taken by the Agent based on value-function.

In value-based methods, the value function is estimated to predict the expected future reward for a given state or action.

Policy-based methods directly map states to actions through a policy. The policy is updated using the policy gradient theorem, which updates the policy in the gradient direction to increase the expected reward.

### How it works

The Actor-Critic algorithm takes inputs from the environment and uses those states to determine the optimal actions.

The Actor component of the algorithm takes the current state as input from the environment. It uses a neural network, which serves as the policy, to output the probabilities of each action for the state.

The Critic network takes the current state and the Actor’s outputted actions as inputs and uses this information to estimate the expected future reward, also known as the Q-value. The Q-value represents the expected cumulative reward an agent can expect to receive if it follows a certain policy in a given state.

On the other hand, the value state represents the expected future reward for a given state, regardless of the action taken. It is calculated as the average of all the Q-values for a given state over all possible actions.

The difference between the expected reward and the average reward for the action is referred to as the advantage function or temporal difference.

$$Adv. = Q(s,a) — V(s)$$

The advantage function provides valuable information to guide the Actor’s policy, allowing it to determine which actions will lead to the best outcomes and adjust its policy accordingly.

If the advantage function for a particular state-action pair is positive, taking that action in that state is expected to yield a better outcome than the average action taken in that state.

The negative value of the advantage function indicates that the current action is less advantageous than expected, and the agent needs to explore other actions or update the policy to improve the performance.

As a result, the advantage function is backpropagated to both the Actor and the Critic, allowing both components to continuously update and improve their respective functions. This results in improved overall performance, as the Actor becomes more effective at making decisions that lead to better outcomes. Ultimately, the Actor-Critic algorithm learns an optimal policy that maximizes the expected future rewards.

The Actor-Critic algorithm is like a framework that forms as the base for several other algorithms like A2C, ACER, A3C, TRPO, and PPO.
