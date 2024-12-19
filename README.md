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

Obviously graph are neither completely heterophilic or homophilic, rather this attribute can be seen as a continous spectrum, still there are examples which meet the extreme like a bipartite graph of a recommender systems where only customers connect with items they might buy.

CoGNN effectively mitigates two of the most common shortcomings of Graph Neural Networks, at least to some extend: Over-Squashing and Over-Smoothing.

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

We trained several **GAT** models with different number of layers (3, 5, 10) on different datasets which were showcased in the papers, including Cora, Roman-Empire, Questions, Amazon-Ratings and Tolokers. For our project we only considered the task of node classification in these different datasets, mainly on heterophilic datasets:

![image](https://github.com/user-attachments/assets/7bd5608a-f232-4cfc-a788-2be45d8ff665)

Source: [A critical look at the evaluations of GNNs under heterophily: Are we really making progress?](https://arxiv.org/abs/2302.11640)

We trained several models using MLFlow for tracking the hyperparameters, especially:
  
- the hidden channels of the GAT layers
- the number of heads for each layer
- the dropout rate
- the learning rate (although we used a scheduler as well)
- different train dataset sizes (we observed that GNNs work very well with a small amount of training data to correctly classify the other nodes in a graph, but we only checked it on a small dataset)

The best performing models were mostly of different architectures when we evaluated them on the different datasets.
Note: For the purpose of fast experimentation and prototyping, no specific folds were created during the training, aka. no cross-validation.

![image](https://github.com/user-attachments/assets/2e00826b-db1e-4030-bf30-5bc285cc8df5)

For simple datasets, we quickly see huge overfitting to the training set. This is expected, as our model is already too complex for this easy task. 

![image](https://github.com/user-attachments/assets/60045ba3-de97-4755-a45b-4e4174df838d)

For different datasets like Tolokers the GAT model performed already quite well and we were able to replicate the results from the paper. As Tolokers is especially connected with an average node degree of 88, we can already conclude that the GAT model here effectively is tackling over-squashing by assigning low attention weights to neighboring nodes wich should not contribute to predictions. We observed that the attention weights seem to be very low in those cases, but never 0. CoGNN will go into another direction, by assigning weights to edges of either 0 or 1.


![image](https://github.com/user-attachments/assets/f2e550a7-ccc6-4d5b-b484-0a0835e22892)

For the visualization part, we came to the conclusion that it makes sense to aggregate over different attention heads and / or layers, depending on the stakeholders technical understanding. For a high level, simplified visualization, we can show for a prediction the graph which was considered by the model, as well as the contributing parts. This could facilitate a better understanding of the models inner workings or could be a basis for decision making.

![image](https://github.com/user-attachments/assets/7cb42e3e-d997-41dd-8ff6-91868f87be78)

In the case of technicak savy colleagues, stakeholders or regulatory bodys, the different attention weights for a prediction can also be shown per layer and per attention head. One could understand in detail, what contributed to a prediction in quite a detailed manner.
To have even more fun, we put the whole prediction pipeline for a specified model into a containerized application, put it on Google ClouD Run and then call its predictions for specific nodes of our graph into a streamlit application for interactivity.


## Cooperative Graph Neural Network
  
For CoGNN we implemented our own training pipeline. 
I will try to quickly describe the main parts of its architecture:

**Action Network:**

- Represents the individual behavior of nodes by determining its "state".
- Generates node-specific actions, determining how a node interacts or cooperates with its neighbors.
- Focuses on local decision-making, ensuring that each node learns an optimal way to collaborate based on the graph's structure and its features.

Usually a GCN with "MEAN" or "SUM" as the permutation invariant aggregation function and implemented as a NN with only 1 hidden layer with a size of 128. In 2 parallel steps, the action networks determine the graph topology by forcing 2 rules on each node: Keep incoming edges or discard them - and keep outgoing edges or discard them. A Node which is not allowed to communicate with any neighbor node would then be ISOLATED. The other remaining states are: LISTENING (just keep incoming edges), BROADCASTING (just keep outgoing edges) and STANDARD (keeping all edges).

**Environment Network:**

- Models the shared environment among nodes. This network applies message passing within the predicted topology of the Action Network.
- Captures global interactions and contextual relationships across the graph.
- Provides feedback to the action network by reflecting how actions taken by nodes impact the overall performance of the model for a specific task during training.

These are the sucessive layers of the actual model. It uses an embedding layer before entering the iteration of the layers, in each layer of the environment the action networks predict the states of the nodes and transform the graphs topology into a weightend graph, with edge weights being either 0 or 1. The action networks output is then transformed into a probability distribution using the gumbel-softmax estimator and will determine the edge weights. Finally we have another decoder layer for our task at hand, which maps to the different classes of the dataset. Last but not least, a skip connection is used in parallel to the layers.

**Gumbel-Softmax Estimator:**

- Used to transform the discrete action space for the nodes into a differentiable function for backpropagation.

In the original paper the authors use another model to learn and determine the "temperature" of the transformation. I wont go into details here, as we deactivated it in our project, still I want to point out that the authors achieved state of the art by enabling the training for this particular part of the model as well.


For our visualization aka. explainability part we considered one rather homophilic and one rather heterphilic dataset.

![image](https://github.com/user-attachments/assets/7c344cfc-2cbe-4dab-9538-bfa3c4254499)

On the homophilic dataset with the target node for prediction in red, we would observe that CoGNN transforms the original undirected (synonym for bidirectional in this context) graph into a directed one, assigning nodes different states. This is done in every layer.


![image](https://github.com/user-attachments/assets/d19db7d1-395e-4b18-a3c4-15263892ec1e)

In deeper layers, we observe that CoGNN increasingly isolates the target node. In our understanding this makes sense in a homophilic dataset and was also mentioned by the authors of the original paper.
  
- **GAT** and **CoGNN** architectures demonstrate state of the art performance on homophilic and heterophilic datasets.
- Both models can be adjusted to not only work on the task of node classification, but also link prediction and graph classification.
- Visualizations bridge the gap between technical and non-technical audience.

## 

## References



## For installation:

- Download repository
- Create a new venv via conda with "conda create -p .envs/dev_env python=3.12"
- Activate venv via "conda activate .envs/dev_env"
- Install the requirements via "pip install -r requirements.txt"
- Navigate to the src.helpers.install_cora.py and execute it - under data/cora.npz you should have the cora dataset downloaded
