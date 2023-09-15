# Community Deception

## Introduction

This repository contains the code for the thesis "Community Deception" for the Master's Degree in Computer Science at Università La Sapienza di Roma.

The goal of this repository is to provide a Reinforcement Learning agent that is able to learn how to hide a given node $u$ from its initial community $C$, computed through a given Community Detection algorithm $f(\cdot)$.

![Community Detection](images/community_detection.png)

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

## Training

To train the model, run the following command:

```bash
python main.py --mode "train"
```

While, to test the model, run the following command:

```bash
python main.py --mode "test"
```

## References


### Code References

The A2C structure is a reimplementation of the code found in the following repository:

```bibtex
@inproceedings{GammelliYangEtAl2021,
  author = {Gammelli, D. and Yang, K. and Harrison, J. and Rodrigues, F. and Pereira, F. C. and Pavone, M.},
  title = {Graph Neural Network Reinforcement Learning for Autonomous Mobility-on-Demand Systems},
  year = {2021},
  note = {Submitted},
}
```

## Directory Structure

```bash
├── dataset
│   ├── archives                          # Contains the archives of the datasets   
│   │   └── ...
│   ├── data                              # Contains the datasets
│   │   └── ...
│   └── readme.md
├── notebook                              # Contains the notebooks used for the analysis
│   └── ...
├── references                            # Contains articles used for the thesis
│   └── ...
├── src                                   # Contains the source code
│   ├── agent                             # Contains the agent code
│   │   ├── a2c
│   │   │   ├── a2c.py
│   │   │   ├── actor.py
│   │   │   ├── critic.py
│   │   │   ├── graph_encoder.py
│   │   │   ├── __init__.py
│   │   │   ├── memory.py
│   │   │   └── readme.md
│   │   ├── agent.py
│   │   ├── __init__.py
│   │   └── readme.md
│   ├── community_algs                  # Contains algorithms for community analysis
│   │   ├── baselines                   # Contains the baselines for community deception
│   │   │   ├── degree_hiding.py
│   │   │   ├── random_hiding.py
│   │   │   ├── readme.md
│   │   │   └── roam_hiding.py
│   │   ├── metrics                     # Contains an implementation of the metrics used for the evaluation
│   │   │   ├── deception_score.py
│   │   │   ├── nmi.py
│   │   │   ├── permanence.py
│   │   │   └── safeness.py
│   │   ├── detection_algs.py           # Contains the community detection algorithms
│   │   ├── __init__.py
│   │   └── readme.md
│   ├── environment                     # Contains the environment of the agent
│   │   ├── graph_env.py
│   │   ├── __init__.py
│   │   └── readme.md
│   ├── logs                            # Contains the logs of the training
│   │   ├── ...
│   │   └── readme.md
│   ├── utils                           # Contains utility functions
│   │   ├── similarity.py               # Contains the functions to measure the similarity
│   │   ├── test.py                     # Contains the functions to test the model
│   │   └── utils.py                    # Contains constants and other utility functions
│   └── __init__.py
├── test                                # Contains the output of the test                    
│   ├── ....
│   └── readme.md
├── main.py                             # Main file, used to train and test the model                 
├── README.md
└── requirements.txt
```