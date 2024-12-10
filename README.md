# xAI_with_graphs

This project implements explainable Graph Neural Networks for node classification, including GAT (Graph Attention Networks) and CoGNN models, with a FastAPI backend for inference and a NextJS frontend for visualization.

## Installation Guide

1. **Clone the Repository**
```bash
git clone <your-repository-url>
cd xAI_with_graphs
```

2. **Create and Activate Conda Environment**
```bash
# Create new environment
conda create -p .envs/dev_env python=3.10

# Activate environment
conda activate .envs/dev_env
```

3. **Install Dependencies**
```bash
# First, install base requirements
pip install -r requirements/base.txt

# For development, also install dev requirements
pip install -r requirements/dev.txt
```

4. **Download the Cora Dataset**
```bash
# Run the dataset installation script
python -m src.data.download

# Verify installation - you should see cora.npz in data directory
ls data/cora.npz
```

## Model Training and Selection

The project supports training different graph neural network models. You can select and train models through the MLflow configuration:

### 1. Configuration Setup
Edit `src/config/mlflow_config.yml`:
```yaml
experiment_name: graph-neural-networks
model_type: gat  # Choose 'gat' or 'cognn'
```

### 2. Available Models

#### GAT (Graph Attention Network)
```yaml
model_type: gat
```
- Implementation: `src/models/gat.py`
- Features: 
  - Multi-head attention
  - Layer normalization
  - Skip connections

#### CoGNN (Conditional Graph Neural Network)
```yaml
model_type: cognn
```
- Implementation: `src/models/cognn.py`
- Features:
  - Gumbel-softmax attention
  - Edge weight learning
  - Conditional computation

### 3. Training Models
```bash
# Train selected model (specified in config)
python -m src.mlflow_main
```

### 4. Tracking Experiments
View training progress and results:
```bash
# Start MLflow UI
mlflow ui --port 5000
```
Access at http://localhost:5000


## Running the Application

1. **Start the API Server**
```bash
# Using the new structure
uvicorn src.api.app:app --host 0.0.0.0 --port 8080 --reload
```
The API will be available at http://localhost:8080

2. **Using Docker (Alternative)**
```bash
# Build the image
docker compose build

# Start the service
docker compose up
```

## Project Structure
```
.
├── src/
│   ├── api/              # FastAPI application
│   ├── data/             # Dataset handling
│   ├── models/           # Model implementations (GAT, CoGNN)
│   ├── training/         # Training scripts
│   ├── utils/           # Utility functions
│   └── visualization/   # Visualization tools
├── tests/               # Test files
├── requirements/        # Dependency files
├── models/             # Saved model artifacts
└── data/               # Dataset storage
    ├── raw/            # Raw downloaded data
    ├── processed/      # Processed datasets
    └── interim/        # Intermediate data
```

## Troubleshooting

1. **Conda Environment Issues**:
   - Make sure conda is properly installed
   - Try `conda init` if conda commands aren't recognized
   - Verify environment activation with `conda env list`

2. **Dataset Download Issues**:
   - Check if `data` directory exists
   - Verify internet connection
   - Check permissions in the data directory

3. **API Startup Issues**:
   - Ensure all requirements are installed
   - Verify port 8080 is available
   - Check if the Cora dataset is properly downloaded

## Next Steps

After installation:
1. Access the API documentation at http://localhost:8080/docs
2. Try the example predictions using the API
3. Explore the visualization capabilities

## License

[Your License]
