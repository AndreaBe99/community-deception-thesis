from typing import List
import matplotlib.pyplot as plt
import numpy as np
import json
import os

ALGS = ["rh", "dh", "di", "agent",]
COLORS = ["red", "green", "orange", "blue"]
MARKER = ["+", "x", "*", "o"]

METRICS = ["goal", "nmi", "time", "steps"]

def checl_dir(path: str)->None:
    """Check if the given directory exists, if not create it"""
    if not os.path.exists(path):
        os.makedirs(path)
        
def load_json(path: str)->dict:
    with open(path, "r") as f:
        log = json.load(f)
    return log

def plot_training_results(
    path: str,
    env_name: str,
    detection_algorithm: str,
    window_size: int = 10):
    """Plot the training results of the agent"""
    log = load_json(path + "/training_results.json")
    def plot_time_series(
        list_1: List[float],
        list_2: List[float],
        label_1: str,
        label_2: str,
        color_1: str,
        color_2: str,
        file_name: str):
        _, ax1 = plt.subplots()
        color = 'tab:'+color_1
        ax1.set_xlabel("Episode")
        ax1.set_ylabel(label_1, color=color)
        ax1.plot(list_1, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:'+color_2
        ax2.set_ylabel(label_2, color=color)
        ax2.plot(list_2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(
            f"Training on {env_name} graph with {detection_algorithm} algorithm")
        plt.savefig(file_name)
        # plt.show()

    plot_time_series(
        log['train_avg_reward'],
        log['train_steps'],
        'Avg Reward',
        'Steps per Epoch',
        'blue',
        'orange',
        path+"/training_reward.png",
    )
    plot_time_series(
        log["a_loss"],
        log["v_loss"],
        'Actor Loss',
        'Critic Loss',
        'green',
        'red',
        path+"/training_loss.png",
    )
    
    # Compute the rolling windows of the time series data using NumPy
    rolling_data_1 = np.convolve(np.array(log["train_avg_reward"]),
        np.ones(window_size) / window_size, mode='valid')
    rolling_data_2 = np.convolve(np.array(log["train_steps"]), 
        np.ones(window_size) / window_size, mode='valid')
    plot_time_series(
        rolling_data_1,
        rolling_data_2,
        'Avg Reward',
        'Steps per Epoch',
        'blue',
        'orange',
        path+"/training_rolling_reward.png",
    )
    # Compute the rolling windows of the time series data using NumPy
    rolling_data_1 = np.convolve(np.array(log["a_loss"]), 
        np.ones(window_size) / window_size, mode='valid')
    rolling_data_2 = np.convolve(np.array(log["v_loss"]), 
        np.ones(window_size) / window_size, mode='valid')
    plot_time_series(
        rolling_data_1,
        rolling_data_2,
        'Actor Loss',
        'Critic Loss',
        'green',
        'red',
        path+"/training_rolling_loss.png",
    )

def plot_evaluation_results(path: str)->None:
    """Plot the evaluation results of the agent"""
    # Load the log
    log = load_json(path + "/evaluation_results.json")

    # Plot the results
    for metric in METRICS:
        for i, alg in enumerate(ALGS):
            fig, ax = plt.subplots()
            ax.set_title(metric)
            # ax.plot(log[alg][metric], label=alg)
            # Scatter plot, each alg with a different marker
            ax.scatter(
                range(len(log[alg][metric])), 
                log[alg][metric], 
                marker=MARKER[i], 
                c=COLORS[i],
                label=alg)
            ax.legend()
            plt.savefig(f"{path}/{alg}_{metric}.png")

def plot_goal_nmi_evaluation(path: str)->None:
    """For eache algorithm, plot the goal and nmi in the same plot"""
    log = load_json(path + "/evaluation_results.json")
    checl_dir(path+"/goal_nmi")
    # Plot the results
    for i, alg in enumerate(ALGS):
        fig, ax = plt.subplots()
        ax.set_title(alg)
        # ax.plot(log[alg][metric], label=alg)
        # Scatter plot, each alg with a different marker
        ax.scatter(
            range(len(log[alg]["goal"])), 
            log[alg]["goal"], 
            marker=MARKER[3],
            label="goal")
        ax.scatter(
            range(len(log[alg]["nmi"])), 
            log[alg]["nmi"],
            marker=MARKER[1], 
            label="nmi")
        ax.legend()
        plt.savefig(f"{path}/goal_nmi/{alg}_goal_nmi.png")
    

if __name__ == "__main__":
    # path = "test/dol/infomap/lr-0.0001_gamma-0.95_lambda-0.1_alpha-0.7"
    # plot_goal_nmi_evaluation(path)

    path = "src/logs/dol/infomap/lr-0.01/gamma-0.95/lambda-1/alpha-0.9"
    plot_training_results(path, "dol", "infomap")
