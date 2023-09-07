# Community Deception

## Introduction

This repository contains the code for the thesis "Community Deception" for the Master's Degree in Computer Science at Università La Sapienza di Roma.

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

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
### Directory Structure

```bash
├── dataset
│   ├── archives
│   │   ├── amz.txt.gz
│   │   ├── astr.txt.gz
│   │   ├── dblp.zip
│   │   ├── dol.zip
│   │   ├── erdos.tar.bz2
│   │   ├── fb-75.zip
│   │   ├── lesm.tar.bz2
│   │   ├── mad.tar.bz2
│   │   ├── ork.ungraph.txt.gz
│   │   ├── polb.tar.bz2
│   │   ├── pow.tar.bz2
│   │   ├── words.tar.bz2
│   │   ├── you.ungraph.txt.gz
│   │   └── zachary.tar.bz2
│   ├── data
│   │   ├── astr.mtx
│   │   ├── dblp.mtx
│   │   ├── dol.mtx
│   │   ├── erdos.mtx
│   │   ├── fb-75.mtx
│   │   ├── kar.mtx
│   │   ├── lesm.mtx
│   │   ├── mad.mtx
│   │   ├── polb.mtx
│   │   ├── pow.mtx
│   │   ├── readme.md
│   │   ├── words.mtx
│   │   └── you.mtx
│   └── readme.md
├── notebook
│   ├── out
│   │   ├── communities.dat
│   │   └── output.txt
│   ├── community_deception.ipynb
│   ├── community_detection.ipynb
│   ├── graphs_analysis.ipynb
│   └── readme.md
├── references
│   ├── a2c
│   │   ├── medium_Actor-Critic_Beginner_Guide.pdf
│   │   ├── medium_Advantage_A2C_algorithm.pdf
│   │   ├── medium_Introduction_A2C.pdf
│   │   ├── medium_Understanding_A2C.pdf
│   │   └── medium_Understanding_Reinforce.pdf
│   ├── community_deception
│   │   ├── 20_multicomm.pdf
│   │   ├── 21_p15-nagaraja.pdf
│   │   ├── 35_1608.00375.pdf
│   │   ├── Adversarial Attack on Community Detection by Hiding.pdf
│   │   ├── Community deception: from undirected to directed networks.pdf
│   │   ├── Community_Deception_or_How_to_Stop_Fearing_Community_Detection_Algorithms.pdf
│   │   ├── community_deception_permanence.pdf
│   │   ├── community_deception_survey.pdf
│   │   ├── From Community Detection to Community Deception.pdf
│   │   ├── Hide_and_Seek_Outwitting_Community_Detection_Algorithms.pdf
│   │   ├── Node-Centric_Community_Deception_Based_on_Safeness.pdf
│   │   ├── overlapping_community_deception.pdf
│   │   └── rem-from-structural-entropy-to-community-structure-deception-Paper.pdf
│   ├── community_detection
│   │   ├── louvain.pdf
│   │   ├── readme.md
│   │   └── walktrap.pdf
│   ├── dqn
│   │   └── Deep Q-Network with Pytorch. DQN _ by Unnat Singh _ Medium.pdf
│   ├── gnn
│   │   └── medium_Understanding_Graph_Convolutional_Networks.pdf
│   ├── graph_representation_learning
│   │   ├── medium_DeepWalk.pdf
│   │   ├── medium_Graph_Embedding_for_Deep_Learning.pdf
│   │   ├── medium_Graph_Embedding.pdf
│   │   ├── medium_Graph_Representation_Learning-Network_Embeddings_1.pdf
│   │   ├── medium_Graph_Representation_Learning-Objective_Functions_and_Encoders_3.pdf
│   │   ├── medium_Graph_Representation_Learning.pdf
│   │   ├── medium_Graph_Representation_Learning-The_Encoder-Decoder_2.pdf
│   │   └── medium_Hands-on Graph Neural Networks with PyTorch & PyTorch.pdf
│   └── rl_on_graph
│       ├── dynamic_network_reconfiguration.pdf
│       ├── Learning_Policies_for_Effective_Graph_Sampling.pdf
│       ├── medium_Reinforcement_Learning_for_Combinatorial_Optimization.pdf
│       ├── Policy-GNN.pdf
│       └── RL_on_garph.pdf
├── src
│   ├── agent
│   │   ├── a2c
│   │   │   ├── a2c.py
│   │   │   ├── actor.py
│   │   │   ├── critic.py
│   │   │   ├── graph_encoder.py
│   │   │   ├── __init__.py
│   │   │   ├── memory.py
│   │   │   └── readme.md
│   │   ├── agent.py
│   │   ├── __init__.py
│   │   └── readme.md
│   ├── community_algs
│   │   ├── metrics
│   │   │   ├── deception_score.py
│   │   │   ├── nmi.py
│   │   │   ├── permanence.py
│   │   │   └── safeness.py
│   │   ├── detection_algs.py
│   │   ├── __init__.py
│   │   └── readme.md
│   ├── environment
│   │   ├── graph_env.py
│   │   ├── __init__.py
│   │   └── readme.md
│   ├── logs
│   │   ├── dol
│   │   │   └── dol_walktrap.pth
│   │   ├── kar
│   │   │   └── kar_walktrap.pth
│   │   └── readme.md
│   ├── utils
│   │   └── utils.py
│   └── __init__.py
├── test
│   ├── dol
│   │   ├── louvain
│   │   └── walktrap
│   │       ├── dol_walktrap_training_loss.png
│   │       └── dol_walktrap_training_reward.png
│   ├── kar
│   │   ├── louvain
│   │   └── walktrap
│   │       ├── kar_walktrap_results.json
│   │       ├── kar_walktrap_rolling_training_loss.png
│   │       ├── kar_walktrap_rolling_training_reward.png
│   │       ├── kar_walktrap_training_loss.png
│   │       └── kar_walktrap_training_reward.png
│   └── readme.md
├── main.py
├── README.md
└── requirements.txt
```


