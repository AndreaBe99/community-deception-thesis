from src.utils.utils import HyperParams, Utils, FilePaths, DetectionAlgorithms
from src.community_algs.detection_algs import DetectionAlgorithm
from src.environment.graph_env import GraphEnvironment
from src.agent.a2c.memory import Memory
from src.agent.agent import Agent
from src.train import train
import os
os.environ['DGLBACKEND'] = 'pytorch'

if __name__ == "__main__":
    print("*"*20, "Setup Information", "*"*20)

    # ° ------ Graph Setup ------ ° #
    # ! Graph path (change the following line to change the graph)
    graph_path = FilePaths.KAR.value
    # Set the environment name as the graph name
    env_name = graph_path.split("/")[-1].split(".")[0]
    # Load the graph from the dataset folder
    graph = Utils.import_mtx_graph(graph_path)
    # Print the number of nodes and edges
    print("* Graph Name:", env_name)
    print("*", graph)

    # ° --- Environment Setup --- ° #
    # Define beta, i.e. the percentage of edges to add/remove
    beta = HyperParams.BETA.value
    # ! Define the detection algorithm to use (change the following line to change the algorithm)
    detection_alg = DetectionAlgorithms.WALK.value
    # Apply the community detection algorithm on the graph
    dct = DetectionAlgorithm(detection_alg)
    community_structure = dct.compute_community(graph)
    # Choose one of the communities found by the algorithm
    community_target = community_structure[0]
    print("* Community Detection Algorithm:", detection_alg)
    print("* Community Target:", community_target)
    
    # Define the environment and the number of possible actions
    env = GraphEnvironment(beta=beta, debug=False)
    # Setup the environment
    env.setup(
        graph=graph,
        community=community_target,
        training=True,
        community_detection_algorithm=detection_alg)
    # Get list of possible actions
    possible_actions = env.get_possible_actions(graph, community_target)
    n_actions = len(possible_actions["ADD"]) + len(possible_actions["REMOVE"])
    print("* Number of possible actions:", n_actions)

    # ° ------ Agent Setup ------ ° #
    # Dimensions of the state
    state_dim = HyperParams.G_IN_SIZE.value
    # Number of possible actions
    action_dim = n_actions
    # Standard deviation for the action
    action_std = HyperParams.ACTION_STD.value
    # Learning rate
    lr = HyperParams.LR.value
    # Gamma parameter
    gamma = HyperParams.GAMMA.value
    # Number of epochs when updating the policy
    k_epochs = HyperParams.K_EPOCHS.value
    # Value for clipping the loss function
    eps_clip = HyperParams.EPS_CLIP.value
    # Define the agent
    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_std=action_std,
        lr=lr,
        gamma=gamma,
        K_epochs=k_epochs,
        eps_clip=eps_clip)
    # Define Memory
    memory = Memory()
    # Print Hyperparameters
    print("*", "-"*18, "Hyperparameters", "-"*18)
    print("* State dimension: ", state_dim)
    print("* Action dimension: ", action_dim)
    print("* Action standard deviation: ", action_std)
    print("* Learning rate: ", lr)
    print("* Gamma parameter: ", gamma)
    print("* Number of epochs when updating the policy: ", k_epochs)
    print("* Value for clipping the loss function: ", eps_clip)
    print("*", "-"*53)

    random_seed = HyperParams.RANDOM_SEED.value
    # Set random seed
    # if random_seed:
    #     print("* Random Seed: {}".format(random_seed))
    #     torch.manual_seed(random_seed)
    #     env.seed(random_seed)
    #     np.random.seed(random_seed)
    print("*"*20, "End Information", "*"*20, "\n")

    # ° ------ Model Training ------ ° #
    # Set the maximum number of steps per episode to the double of the edge budget
    max_timesteps = env.edge_budget*2
    # Set the update timestep to 10 times then edge budget
    update_timesteps = env.edge_budget*10
    # Start training
    episodes_avg_reward, episode_length = train(
        env=env, 
        agent=agent, 
        memory=memory, 
        max_timesteps=max_timesteps,
        update_timesteps=update_timesteps, 
        env_name=env_name)
    
    # ° ------ Plot Results ------ °#
    # Plot the average reward per episode
    Utils.plot_avg_reward(episodes_avg_reward, episode_length, env_name)
