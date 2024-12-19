# Explainability with Graphs

![image](https://github.com/user-attachments/assets/f2e550a7-ccc6-4d5b-b484-0a0835e22892)


## Overview

Graph algorithms are fundamental in AI systems design because they help model and solve problems involving networks, relationships, and structures. These algorithms enable efficient handling of tasks such as shortest path finding, clustering, and graph traversal, which are essential in areas like recommendation systems, knowledge graph construction, and neural network design.

This project focuses on **explainability with Graph Neural Networks (GNNs)** by implementing and visualizing state-of-the-art architectures by focusing on the task of node classification on different datasets showcased in the papers mentioned later. The project is divided into two key areas:

1. **Implementing GAT Architectures**:

   - Designed for both **homophilic** and **heterophilic** datasets (which I will explain later).
   - Attention weights are aggregated and visualized to demonstrate explainability.
   - Source:  [Graph Attention Network (GAT)](https://arxiv.org/abs/2310.01267), with code: [GAT on papers with code](https://paperswithcode.com/paper/graph-attention-networks)

2. **Implementing the CoGNN Architecture**:

   - A cutting-edge GNN model from 2024 called "Cooperative Graph Neural Networks".
   - Focused on heterophilic datasets and visualized information flow for enhanced understanding.
   - Source: [CoGNN](https://arxiv.org/abs/2310.01267), with code: [CoGNN on papers with code](https://paperswithcode.com/paper/cooperative-graph-neural-networks)
     
### Goals

- Provide **non-technical stakeholders** with intuitive visualizations of GNN sub-graphs and attention mechanisms.
- Offer **technical stakeholders** detailed insights into attention weights across heads and layers.
- Showcase the **coGNN architecture** with advanced visualizations of its information flow.

## Features

- **Explainability for Non-Technical Stakeholders**:
  - High-level sub-graph visualizations.
  - Aggregated attention weights across layers and heads.
- **Detailed Visualizations for Technical Stakeholders**:
  - Attention weights across all attention heads and layers.
  - Information flow visualizations for the coGNN architecture.
- **State-of-the-Art Model Implementations**:
  - GAT architectures.
  - coGNN reimplementation tailored for heterophilic datasets.

## Technology Stack

- **Languages**: Python
- **Frameworks and Libraries**:
  - Streamlit: For interactive visualization.
  - MLFlow: For model tracking and experimentation.
  - Pytorch Geometric: A library dedicated to graphs using python.
- **Visualization**:
  - HTML: For embedding dynamic and interactive visualizations.
- **Deployment**:
  - Docker: For containerization and reproducibility.
  - Google Cloud Run: For deploying the application.

## What are homophilic and heterophilic graphs?

![image](https://github.com/user-attachments/assets/d94d65ba-22f6-410d-9af4-a838fc67be11)

In a homophilic environment, nodes have a higher probability to connect via edges if they share the same attributes or class. In contrast, in a heterophilic environment nodes rather connect if they dont share the same class or attributes. Anyway, the graph structure remains the same for message passing graph neural networks, even when attention is used. 

CoGNN assigns weights (either 0 or 1) on the edges during inference, effectively cutting them out of the topology and therefore transforms a graph from a bidirectional into a directed graph (if the model decides that one direction might not contribute well to the prediction). The model can even completely isolate nodes from information exchange at a given layer, which can be seen on the right.

This effectively mitigates two of the most common shortcomings of Graph Neural Networks to some extend: Over-Squashing and Over-Smoothing.

### Over-Squashing
Means loss of distinct information by forcing  too much information into a single feature vector due to the graphs structure by repeated updates by the model. This often happens in highly connected topologies. 
The cause is that the receptive field of the model grows exponentially with the number of layers.

### Over-Smoothing
Means loss of information by updating entities too often - as a consequence the feature vectors for e.g. nodes will look very similar even if the nodes have different classes and therefore they lose their distinct nature.

## How do GNNs work in general

![image](https://github.com/user-attachments/assets/b334ae64-0340-4043-ba93-d8b6b546428d)

The model is able to "cluster" feature vectors (in this case node feature vectors, where different colors belong to different classes) into similar feature embeddings. 
The left image shows the embeddings of the node feature vectors in 2D, after running a trained model on the data we can see how the model is able to effectively distinguish between classes in the embedding space.


## Results

We trained several GAT models with different number of layers (3, 5, 10) on different datasets which were showcased in the papers, including Cora, Roman-Empire, Questions, Amazon-Ratings and Tolokers. For our project we only considered the task of node classification in these different datasets, mainly on heterophilic datasets:

![image](https://github.com/user-attachments/assets/7bd5608a-f232-4cfc-a788-2be45d8ff665)
(Source: [A critical look at the evaluations of GNNs under heterophily: Are we really making progress?](https://arxiv.org/abs/2302.11640)

- **GAT** and **CoGNN** architectures demonstrate state of the art performance on homophilic and heterophilic datasets.
- Both models can be adjusted to not only work on the task of node classification, but also link prediction and graph classification.
- Visualizations bridge the gap between technical and non-technical audience.

## References



## For installation:

- Download repository
- Create a new venv via conda with "conda create -p .envs/dev_env python=3.12"
- Activate venv via "conda activate .envs/dev_env"
- Install the requirements via "pip install -r requirements.txt"
- Navigate to the src.helpers.install_cora.py and execute it - under data/cora.npz you should have the cora dataset downloaded
