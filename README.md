# xAI_with_graphs

This project implements explainable Graph Attention Networks for node classification using PyTorch Geometric, with a FastAPI backend for inference.

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

## Running the Application

1. **Start the API Server**
```bash
# Using the new structure
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
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
│   ├── api/          # FastAPI application
│   ├── data/         # Dataset handling and download scripts
│   ├── models/       # Model definitions
│   ├── training/     # Training scripts
│   ├── utils/        # Utility functions
│   └── visualization/# Visualization tools
├── tests/            # Test files
├── requirements/     # Dependency files
│   ├── base.txt     # Core dependencies
│   ├── dev.txt      # Development dependencies
│   └── prod.txt     # Production dependencies
└── data/            # Dataset storage
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