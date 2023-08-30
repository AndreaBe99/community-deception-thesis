# Community Deception

## Introduction

This repository contains the code for the thesis "Community Deception" for the Master's Degree in Computer Science at Università La Sapienza di Roma.

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

To install **Deep Graph Library** (`dgl`) for your system, follow the instructions [here](<https://www.dgl.ai/pages/start.html>).

## Training

To train the model, run the following command:

```bash
python main.py
```

## References

## Community Deception

Paper about community deception:

```bibtex
@article{8118127,
  author    = {Fionda, Valeria and Pirrò, Giuseppe},
  journal   = {IEEE Transactions on Knowledge and Data Engineering}, 
  title     = {Community Deception or: How to Stop Fearing Community Detection Algorithms}, 
  year      = {2018},
  volume    = {30},
  number    = {4},
  pages     = {660-673},
  doi       = {10.1109/TKDE.2017.2776133}
}
```

### Code References

A2C structure from:

```bibtex
@misc{yusx-swapp/gnn-rl-model-compression,
    url     = {<https://github.com/yusx-swapp/GNN-RL-Model-Compression/tree/master>},
    journal = {GitHub}, 
    year    = {2023} 
}
```

### Graph Rewiring

Paper about graph rewiring using reinforcement learning:

```bibtex
@article{DBLP:journals/corr/abs-2205-13578,
    author      = {Christoffel Doorman and Victor{-}Alexandru Darvariu and Stephen Hailes and Mirco Musolesi},
    title       = {Dynamic Network Reconfiguration for Entropy Maximization using Deep Reinforcement Learning},
    journal     = {CoRR},
    volume      = {abs/2205.13578},
    year        = {2022},
    url         = {https://doi.org/10.48550/arXiv.2205.13578},
    doi         = {10.48550/arXiv.2205.13578},
    eprinttype  = {arXiv},
    eprint      = {2205.13578},
    timestamp   = {Tue, 31 May 2022 15:14:51 +0200},
    biburl      = {https://dblp.org/rec/journals/corr/abs-2205-13578.bib},
    bibsource   = {dblp computer science bibliography, https://dblp.org}
}
```

They use a Structure2Vec model to encode the graph structure and obtain an embedding. The embedding is then used as input for a DQN that outputs the action to take.
‌
