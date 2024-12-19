# xAI_with_graphs

## Overview

This project focuses on **explainability with Graph Neural Networks (GNNs)** by implementing and visualizing state-of-the-art architectures. The project is divided into two key areas:

1. **Implementing GAT Architectures**:

   - Designed for both **homophilic** and **heterophilic** datasets.
   - Attention weights are aggregated and visualized to demonstrate explainability.
   - Source:  [Graph Attention Network (GAT)](https://arxiv.org/abs/2310.01267)

2. **Reimplementing the coGNN Architecture**:

   - A cutting-edge GNN model from 2024.
   - Focused on heterophilic datasets and visualized information flow for enhanced understanding.
   - Source: [CoGNN](https://arxiv.org/abs/2310.01267)
     
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

## Results

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
