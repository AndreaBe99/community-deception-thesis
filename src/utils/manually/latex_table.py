import json
import pandas as pd

def generate_latex_table(
    path: str,
    metrics: list = ["goal", "nmi"]):
    """
    Given a path to a JSON file, generate a LaTeX table, using pandas.
    The JSON file should be in the following format:
        - First level keys (Metrics): "goal", "nmi", "steps", "time"
        - Second level keys (Beta): "0.5", "1", "2"
        - Third level keys (Algorithm): "DRL-Agent (our)", "Random", "Degree", "Roam"
        - Fourth level keys (Values): "mean", "std"     

    Build a pandas DataFrame with the following columns and rows:
        |- - - - |- - - |- - - - - - - - -|- - - - |- - - - |- - - |
        | Metric | Beta |                   Algorithm              |
        |        |      |- - - - - - - - -|- - - - |- - - - |- - - |
        |        |      | DRL-Agent (our) | Random | Degree | Roam |
        |- - - - |- - - |- - - - - - - - -|- - - - |- - - - |- - - |
        | GOAL   | 0.5  |                 |        |        |      |
        |        |- - - |- - - - - - - - -|- - - - |- - - - |- - - |
        |        | 1    |                 |        |        |      |
        |        |- - - |- - - - - - - - -|- - - - |- - - - |- - - |
        |        | 2    |                 |        |        |      |
        |- - - - |- - - |- - - - - - - - -|- - - - |- - - - |- - - |
        | NMI    | 0.5  |                 |        |        |      |
        |        |- - - |- - - - - - - - -|- - - - |- - - - |- - - |
        |        | 1    |                 |        |        |      |
        |        |- - - |- - - - - - - - -|- - - - |- - - - |- - - |
        |        | 2    |                 |        |        |      |
        |- - - - |- - - |- - - - - - - - -|- - - - |- - - - |- - - |
    
    Print the DataFrame as a LaTeX table.
    
    Parameters
    ----------
    path : str
        _description_
    """

    # Load JSON data from file
    with open(path, "r") as f:
        data = json.load(f)

    
    # Create empty DataFrame with 3 columns: Metric, Beta, Algorithm.
    # and use multi column index for the Algorithm column, adding 4 columns: 
    # DRL-Agent (our), Random, Degree, Roam

    df = pd.DataFrame(columns=["Metric", "Beta", "DRL-Agent (our)", "Random", "Degree", "Roam"])
    
    values = ["mean", "std"]
    # Loop over metrics, betas, and algorithms to fill the DataFrame
    for metric in metrics:
        for beta in data[metric]:
            row = {"Metric": metric, "Beta": beta+r"$\mu$"}
            for algorithm in data[metric][beta]:
                if metric == "goal":
                    # Multiply by 100 to get percentage and round to 2 decimals
                    mean = int(round(data[metric][beta][algorithm][values[0]] * 100, 2))
                    row[algorithm] = str(mean) + "\\%"
                else:
                    # Round to 2 decimals
                    mean = round(data[metric][beta][algorithm][values[0]], 2)
                    std = round(data[metric][beta][algorithm][values[1]], 2)
                    row[algorithm] = str(mean) + " $\\pm$ " + str(std)
            # print(row)
            # Add row to DataFrame
            df.loc[len(df)] = row
        
    # Replace "goal" with "SR" and "nmi" with "NMI"
    df["Metric"] = df["Metric"].replace({"goal": "SR", "nmi": "NMI"})
    df.loc[(df['Metric'] == "SR").idxmax(), 'Metric'] = r"\multirow{3}{*}{\textbf{SR}}"
    df["Metric"] = df["Metric"].replace({"SR": ""})

    # Do the same for "NMI"
    df.loc[(df['Metric'] == "NMI").idxmax(), 'Metric'] = r"\multirow{3}{*}{\textbf{NMI}}"
    df["Metric"] = df["Metric"].replace({"NMI": ""})
    
    # Rename Metric column to "\multirow{2}{*}{\textbf{Metric}}"
    df.rename(columns={"Metric": r"\multirow{2}{*}{\textbf{Metric}}"}, inplace=True)
    # Rename Beta column to "\multirow{2}{*}{\textbf{$\beta$}}"
    df.rename(columns={"Beta": r"\multirow{2}{*}{\textbf{$\beta$}}"}, inplace=True)
    
    
    # Print the DataFrame as a LaTeX table
    latex_str = df.to_latex(
        index=False,
        column_format="|c|c|c|c|c|c|",
        multicolumn=True,
        multicolumn_format="c",
        multirow=True,
        )
    
    # Replace "DRL-Agent (our) & Random & Degree & Roam \\" with:
    # "\multicolumn{4}{c|}{\textbf{Node Deception Algorithm}} \\ \cline{3-6} & & \textit{DRL-Agent(ours)} & \textit{Random} & \textit{Degree} & \textit{Roam} \\"
    latex_str = latex_str.replace(
        "DRL-Agent (our) & Random & Degree & Roam \\\\", 
        r"\multicolumn{4}{c|}{\textbf{Node Deception Algorithm}} \\ \cline{3-6} & & \textit{DRL-Agent(ours)} & \textit{Random} & \textit{Degree} & \textit{Roam} \\"
        )
    # Replace "\toprule" with "\hline"
    latex_str = latex_str.replace(r"\toprule", r"\hline")
    # Replace "\midrule" with "\hline"
    latex_str = latex_str.replace(r"\midrule", r"\hline \hline")
    # Replace "\bottomrule" with "\hline"
    latex_str = latex_str.replace(r"\bottomrule", r"\hline")
    
    # Replace "\multirow{3}{*}{\textbf{NMI}}" to "\hline \hline \multirow{3}{*}{\textbf{NMI}}"
    latex_str = latex_str.replace(r"\multirow{3}{*}{\textbf{NMI}}", r"\hline \hline \multirow{3}{*}{\textbf{NMI}}")
    print(latex_str)




if __name__ == "__main__":
    PATH = "test/kar/greedy/node_hiding/tau_0.8/allBetas_evaluation_node_hiding_mean_std.json"
    generate_latex_table(PATH)
