from typing import List, Tuple
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from src.utils.utils import HyperParams
import numpy as np
import pandas as pd
import seaborn as sns
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


def plot_training_results_seaborn(
        file_path: str,
        env_name: str,
        detection_algorithm: str,
        window_size: int = 10):
    """Plot the training results of the agent using Seaborn"""
    log = load_json(file_path + "/training_results.json")

    if window_size < 1:
            window_size = 1
    df = pd.DataFrame({
        "Episode": range(len(log["train_avg_reward"])),
        "Avg Reward": log["train_avg_reward"],
        "Steps per Epoch": log["train_steps"],
        "Goal Reward": log["train_reward_mul"],
        "Goal Reached": [1/log["train_steps"][i] if log["train_reward_list"][i][-1]
            > 1 else 0 for i in range(len(log["train_steps"]))],
    })
    df["Rolling_Avg_Reward"] = df["Avg Reward"].rolling(window_size).mean()
    df["Rolling_Steps"] = df["Steps per Epoch"].rolling(window_size).mean()
    df["Rolling_Goal_Reward"] = df["Goal Reward"].rolling(window_size).mean()
    df["Rolling_Goal_Reached"] = df["Goal Reached"].rolling(window_size).mean()
    plot_seaborn(
        df,
        file_path+"/training_reward.png",
        env_name,
        detection_algorithm,
        ("Avg Reward", "Rolling_Avg_Reward"),
        ("lightsteelblue", "darkblue"),
    )
    plot_seaborn(
        df,
        file_path+"/training_steps.png",
        env_name,
        detection_algorithm,
        ("Steps per Epoch", "Rolling_Steps"),
        ("thistle", "purple"),
    )
    plot_seaborn(
        df,
        file_path+"/training_goal_reward.png",
        env_name,
        detection_algorithm,
        ("Goal Reward", "Rolling_Goal_Reward"),
        ("darkgray", "black"),
    )
    plot_seaborn(
        df,
        file_path+"/training_goal_reached.png",
        env_name,
        detection_algorithm,
        ("Goal Reached", "Rolling_Goal_Reached"),
        ("darkgray", "black"),
    )

    df = pd.DataFrame({
        "Episode": range(len(log["a_loss"])),
        "Actor Loss": log["a_loss"],
        "Critic Loss": log["v_loss"],
    })
    df["Rolling_Actor_Loss"] = df["Actor Loss"].rolling(window_size).mean()
    df["Rolling_Critic_Loss"] = df["Critic Loss"].rolling(window_size).mean()
    plot_seaborn(
        df,
        file_path+"/training_a_loss.png",
        env_name,
        detection_algorithm,
        ("Actor Loss", "Rolling_Actor_Loss"),
        ("palegreen", "darkgreen"),
    )
    plot_seaborn(
        df,
        file_path+"/training_v_loss.png",
        env_name,
        detection_algorithm,
        ("Critic Loss", "Rolling_Critic_Loss"),
        ("lightcoral", "darkred"),
    )

def plot_seaborn(
    df: pd.DataFrame,
    path: str,
    env_name: str,
    detection_algorithm: str,
    labels: Tuple[str, str],
    colors: Tuple[str, str])->None:
    sns.set_style("darkgrid")
    sns.lineplot(data=df, x="Episode", y=labels[0], color=colors[0])
    sns.lineplot(data=df, x="Episode", y=labels[1], color=colors[1],
                estimator="mean", errorbar=None)
    plt.title(
        f"Training on {env_name} graph with {detection_algorithm} algorithm")
    plt.xlabel("Episode")
    plt.ylabel(labels[0])
    plt.savefig(path)
    plt.clf()

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


def plot_evaluation_results_seaborn(
        path: str,
        env_name: str,
        detection_algorithm: str):
    """Plot the evaluation results of the algorithms using Seaborn"""
    log = load_json(path + "/evaluation_results.json")

    # Algorithms
    list_algs = HyperParams.ALGS_EVAL.value
    # Metrics for each algorithm
    metrics = HyperParams.METRICS_EVAL.value
    from statistics import mean
    for metric in metrics:
        # Create a DataFrame with the mean values of each algorithm for the metric
        df = pd.DataFrame({
            "Algorithm": list_algs,
            metric.capitalize(): [mean(log[alg][metric]) for alg in list_algs]
        })
        # Create the bar plot with the mean values of each algorithm for the metric
        sns.set_style("darkgrid")
        # If the metric is goal the y axis is a %, so we need to multiply by 100
        # and '%' on the y label axis
        if metric == "goal":
            df[metric.capitalize()] = df[metric.capitalize()] * 100
        
        sns.barplot(data=df, 
                    x="Algorithm",
                    y=metric.capitalize(),
                    palette=sns.color_palette("Set1"))
        plt.title(
            f"Evaluation on {log['env']['dataset']} graph with {log['env']['detection_alg']} algorithm")
        
        plt.xlabel("Algorithm")
        if metric == "goal":
            plt.ylabel(f"{metric.capitalize()} reached %")
        elif metric == "time":
            plt.ylabel(f"{metric.capitalize()} (s)")
        else:
            plt.ylabel(metric.capitalize())
        plt.savefig(f"{path}/{metric}.png")
        plt.clf()

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

    # path = "src/logs/lfr_benchmark_n-300/infomap/lr-0.0001/gamma-0.9/lambda-0.1/alpha-0.7"
    # plot_training_results_seaborn(path, "dol", "infomap")
    
    path = "../../test/kar/greedy/tau-0.5/beta-3/lr-0.0001/gamma-0.9/lambda-0.1/alpha-0.7"
    plot_evaluation_results_seaborn(path, "kar", "greedy")
