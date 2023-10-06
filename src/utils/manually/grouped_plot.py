from statistics import mean
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import os



def plot_singleDataset_singleTaus_allBetas(
    file_path: str,
    log_name: str,
    algs: List[str],
    metrics: List[str],
    betas: List[float],):
    """
    Given a tau folder make a plot for each metric, where each plot contains
    a group-plot with N groups (with N the number of betas) and each group
    contains a M bars (with M the number of algorithms).
    
    JSON file structure:
        - first level: algorithm (e.g. "Agent", "Random", "Degree", "Roam")
        - second level: metric (e.g. "goal", "steps", "reward", "time")
    
    Each metric value is a list of 100 values.
    """
    # Renae the "Agent" key to "DRL-Agent (our)"
    agent_renamed = "DRL-Agent (our)"
    # Create a json file, where each key is a metric, and the values are
    # dictionaries with keys the algorithms and values the mean of the metric
    dict_metrics = {}
    for metric in metrics:
        dict_metrics[metric] = {}
        for beta in betas:
            dict_metrics[metric][beta] = {}
            for alg in algs:
                if alg == "Agent":
                    dict_metrics[metric][beta][agent_renamed] = []
                else:
                    dict_metrics[metric][beta][alg] = []
    
    # Explore the folder structure
    for beta in betas:
        # Load the json file
        with open(f"{file_path}/beta_{beta}/{log_name}.json", "r") as f:
            data = json.load(f)
            
            # Copy the data to the dict_metrics
            for metric in metrics:
                for alg in algs:
                    # Replace the "Agent" key with "DRL-Agent (our)"
                    if alg == "Agent":
                        dict_metrics[metric][beta][agent_renamed] = data[alg][metric]
                    else:
                        dict_metrics[metric][beta][alg] = data[alg][metric]
    # Replace in the algs list the "Agent" with "DRL-Agent (our)"
    algs = [agent_renamed if alg == "Agent" else alg for alg in algs]
    
    # Save the mean and std of the metric for each algorithm and beta
    mean_std = {}
        
    # Make a plot for each metric
    for metric in metrics:
        mean_std[metric] = {}
        plot_data = []
        for beta in betas:
            # Create a dataframe from dict_metrics
            df = pd.DataFrame(dict_metrics[metric][beta])
            # Convert the column "goal" to percentages for each algorithm
            if metric == "goal":
                df = df.apply(lambda x: x * 100)
            # Rename the columns called "Agent" to "DRL-Agent (our)"
            df = df.rename(columns={"Agent": agent_renamed})
            # Add the dataframe to the plot_data
            plot_data.append(df)
        
        # Concatenate the dataframes
        df = pd.concat(plot_data, axis=1)
        df.columns = pd.MultiIndex.from_product([betas, algs])
        # Melt the dataframe
        df = df.melt(var_name=["Beta", "Algorithm"], value_name=metric)
        
        # Save the mean and std of the metric for each algorithm and beta
        for beta in betas:
            mean_std[metric][beta] = {}
            for alg in algs:
                mean_std[metric][beta][alg] = {
                    "mean": mean(dict_metrics[metric][beta][alg]),
                    "std": np.std(dict_metrics[metric][beta][alg])}
        
        # Plot the data
        sns.set_theme(style="darkgrid")
        # Increase the font size
        sns.set(font_scale=1.5)
        # Set palette
        if log_name == "evaluation_node_hiding":
            palette = sns.set_palette("Set1")
        elif log_name == "evaluation_community_hiding":
            palette = sns.set_palette("Set2")
            
        # if the metric is goal don't plot the error bars
        if metric == "nmi" or metric == "deception_score":
            errorbar="sd"
        else:
            errorbar=None

        # Plot the data
        g = sns.catplot(
            data=df, kind="bar", x="Beta", y=metric, hue="Algorithm",
            aspect=1, palette=palette, errorbar=errorbar)
        # Set labels as Betas and Metrics
        g.set_axis_labels("β Values", f"Mean {metric.capitalize()}")
        
        # Rename the x axis label values to "Nμ" where N is the previous name.
        if log_name == "evaluation_node_hiding":
            g.set_xticklabels(
                [f"{float(t.get_text())}μ" for t in g.ax.get_xticklabels()])
        
        # if the metric is goal set the y axis to percentages
        if metric == "goal":
            metric = "sr"
            g.set(ylim=(0, 100))
            g.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
            g.set_ylabels("Success Rate")
        elif metric == "nmi":
            g.set(ylim=(0, 1))
            g.set_ylabels("NMI (avg)")
        elif metric == "deception_score":
            metric = "ds"
            g.set(ylim=(0, 1))
            g.set_ylabels("Deception Score (avg)")
        elif metric == "steps":
            g.set_ylabels("Steps (avg)")
        elif metric == "time":
            g.set_ylabels("Time in sec. (avg)")
        
        # Save the plot
        g.savefig(
            f"{file_path}/{log_name}_{metric}_group.png",
            bbox_inches='tight', 
            dpi=300)

        # Save the mean and std to a json file
        with open(f"{file_path}/allBetas_{log_name}_mean_std.json", "w") as f:
            json.dump(mean_std, f)

def plot_singleBeta_singleTau_allDataset(
    file_path: str,
    log_name: str,
    algs: List[str],
    detection_alg: str,
    metrics: List[str],
    datasets: List[str],
    beta: float,
    tau: float):
    """
    Given a path, loop over all the datasets (given arguments) folders, and 
    given the beta and tau values, make a group-plot for each metric, whith
    N groups (with N the number of datasets) and each group contains a M bars
    (with M the number of algorithms).
    
    JSON file structure:
        - first level: algorithm (e.g. "Agent", "Random", "Degree", "Roam")
        - second level: metric (e.g. "goal", "steps", "reward", "time")
    
    FOLDER structure:
        - first level: 2 folder, one for each dataset (e.g. "words", "karate", "football")
        - second level: 3 folder, one for each  detection algorithms (greedy, louvain, walktrap)
        - third level: 2 folder, one for node hiding and one for community hiding
        - fourth level: 3 folders, one for each tau value
        - fifth level: 3 folders, one for each beta value
    """
    # Renae the "Agent" key to "DRL-Agent (our)"
    agent_renamed = "DRL-Agent (our)"
    
    # Save a dictionary with the first level keys is the metric, the second
    # level keys is the dataset, the third level keys is the algorithm
    metrics_dict = {}
    for metric in metrics:
        metrics_dict[metric] = {}
        for dataset in datasets:
            metrics_dict[metric][dataset] = {}
            
    for dataset in datasets:
        # Load the path of the json file: dataset/detection_alg/node_hiding/tau/beta
        if log_name == "evaluation_node_hiding":
            json_path = f"{file_path}/{dataset}/{detection_alg}/node_hiding/tau_{tau}/beta_{beta}/{log_name}.json"
        else:
            json_path = f"{file_path}/{dataset}/{detection_alg}/community_hiding/tau_{tau}/beta_{beta}/{log_name}.json"
        # Load the json file
        with open(json_path, "r") as f:
            data = json.load(f)
        
        for metric in metrics:
            for alg in algs:
                if alg == "Agent":
                    metrics_dict[metric][dataset][agent_renamed] = data[alg][metric]
                else:
                    metrics_dict[metric][dataset][alg] = data[alg][metric]

    for metric in metrics:
        plot_data = []
        for dataset in datasets:
            # Create a dataframe from dict_metrics
            df = pd.DataFrame(metrics_dict[metric][dataset])
            # Convert the column "goal" to percentages for each algorithm
            if metric == "goal":
                df = df.apply(lambda x: x * 100)
            # Rename the columns called "Agent" to "DRL-Agent (our)"
            df = df.rename(columns={"Agent": agent_renamed})
            # Add the dataframe to the plot_data
            plot_data.append(df)
        
        # Concatenate the dataframes
        df = pd.concat(plot_data, axis=1)
        df.columns = pd.MultiIndex.from_product([datasets, algs])
        # Melt the dataframe
        df = df.melt(var_name=["Dataset", "Algorithm"], value_name=metric)
        
        # Set theme
        sns.set_theme(style="darkgrid")
        # Increase the font size
        sns.set(font_scale=1.5)
        # Set palette
        if log_name == "evaluation_node_hiding":
            palette = sns.set_palette("Set1")
        elif log_name == "evaluation_community_hiding":
            palette = sns.set_palette("Set2")

        # if the metric is goal don't plot the error bars
        if metric == "nmi" or metric == "deception_score":
            errorbar = "sd"
        else:
            errorbar = None

        # Plot the data
        g = sns.catplot(
            data=df, kind="bar", x="Dataset", y=metric, hue="Algorithm",
            aspect=1, palette=palette, errorbar=errorbar)
        # Set labels as Betas and Metrics
        g.set_axis_labels("Datasets", f"Mean {metric.capitalize()}")
        # if the metric is goal set the y axis to percentages
        if metric == "goal":
            metric = "sr"
            g.set(ylim=(0, 100))
            g.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
            g.set_ylabels("Success Rate")
        elif metric == "nmi":
            g.set(ylim=(0, 1))
            g.set_ylabels("NMI (avg)")
        elif metric == "deception_score":
            metric = "ds"
            g.set(ylim=(0, 1))
            g.set_ylabels("Deception Score (avg)")
        elif metric == "steps":
            g.set_ylabels("Steps (avg)")
        elif metric == "time":
            g.set_ylabels("Time in sec. (avg)")

        # Save the plot
        g.savefig(
            f"{file_path}/allDataset_{log_name}_{metric}_tau{tau}_beta{beta}_group.png",
            bbox_inches='tight',
            dpi=300)

        

if __name__ == "__main__":
    ################# SINGLE DATASET - SINGLE TAU - ALL BETAS #################
    # NODE HIDING
    # PATH = "test/words/walktrap/node_hiding/tau_0.8"
    # plot_singleDataset_singleTaus_allBetas(
    #     file_path=PATH,
    #     log_name="evaluation_node_hiding",
    #     algs=["Agent", "Random", "Degree", "Roam"],
    #     metrics=["goal", "nmi", "steps", "time"],
    #     betas=[0.5, 1, 2],
    # )
    # # COMMUNITY HIDING
    # PATH = "test/words/walktrap/community_hiding/tau_0.8"
    # plot_singleDataset_singleTaus_allBetas(
    #     file_path=PATH,
    #     log_name="evaluation_community_hiding",
    #     algs=["Agent", "Safeness", "Modularity"],
    #     metrics=["goal", "nmi", "deception_score", "steps", "time"],
    #     betas=[1, 3, 5],
    # )
    
    ################# SINGLE BETA - SINGLE TAU - ALL DATASET #################
    DETECTION_ALG = "walktrap"
    PATH = "test"
    
    # NODE HIDING
    plot_singleBeta_singleTau_allDataset(
        PATH,
        log_name="evaluation_node_hiding",
        algs=["Agent", "Random", "Degree", "Roam"],
        detection_alg=DETECTION_ALG,
        metrics=["goal", "nmi", "steps", "time"],
        datasets=["kar", "words", "vote"],
        beta=1,
        tau=0.3,
    )
    # COMMUNITY HIDING
    plot_singleBeta_singleTau_allDataset(
        PATH,
        log_name="evaluation_community_hiding",
        algs=["Agent", "Safeness", "Modularity"],
        detection_alg=DETECTION_ALG,
        metrics=["goal", "nmi", "deception_score", "steps", "time"],
        datasets=["kar", "words", "vote"],
        beta=1,
        tau=0.3,
    )