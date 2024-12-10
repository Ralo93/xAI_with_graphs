import numpy as np
import torch
import logging
import mlflow
import optuna
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                           precision_score, recall_score)
import torch.nn.functional as F
import torch.nn as nn
from helpers.datasets import Dataset, CoraDataset
from typing import Dict, Any, Tuple
import yaml
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class GAT(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 num_heads: int = 4, dropout: float = 0.3, edge_dim: int = None):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Calculate output dimensions
        self.gat1_out_channels = hidden_channels * num_heads
        self.gat2_out_channels = hidden_channels * num_heads
        
        # Replace BatchNorm with LayerNorm
        self.norm1 = nn.LayerNorm(self.gat1_out_channels)
        self.norm2 = nn.LayerNorm(self.gat2_out_channels)

        # Input projection for skip connection
        self.proj_skip = nn.Linear(in_channels, self.gat2_out_channels)
        
        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim
        )
        
        self.gat2 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=hidden_channels,
            heads=num_heads,
            concat=True,
            dropout=dropout
        )
        
        self.gat3 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=out_channels,# // num_heads,
            heads=num_heads,
            concat=False,
            dropout=dropout
        )

    # This is what we expect in the predict method using fastapi
    def forward(self, x, edge_index):
            # Save input for skip connection
            x_skip = x

            # GAT layer 1
            x = self.gat1(x, edge_index)
            x = F.elu(x)
            x = self.norm1(x)
            x = self.dropout(x)

            # GAT layer 2
            x_skip = self.proj_skip(x_skip)  # Align dimensions for skip connection
            x = x + x_skip  # Add skip connection
            x = self.gat2(x, edge_index)
            x = F.elu(x)
            x = self.norm2(x)
            x = self.dropout(x)

            # GAT layer 3
            x = self.gat3(x, edge_index)

            return x

  
class GATTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config["training"]["device"])

    def prepare_data(self):
        # Load dataset
        print("Loading dataset...")

        set = config['dataset']
        #dataset = Dataset(name=set, device=str(self.device))
        dataset = CoraDataset("data/cora.npz", add_self_loops=False, device='cpu')
        
        #x = dataset.node_features does not work for cora
        x = dataset.node_features
        y = dataset.labels
        edge_index = dataset.edges
        
        # Print dataset statistics
        print(f"Node feature matrix shape (x): {x.shape}")
        print(f"Label tensor shape (y): {y.shape}")
        print(f"Edge index shape (original): {edge_index.shape}")
        print(f"Unique labels: {y.unique().tolist()}")
        print(f"Number of edges: {edge_index.size(1)}")
        
        # Make bidirectional and add self-loops
        edge_index = self.make_bidirectional(edge_index)
        print(f"Edge index shape (after make_bidirectional): {edge_index.shape}")

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            print(f"Transposing edge index. Current shape: {edge_index.shape}")
            edge_index = edge_index.t()

        edge_index = add_self_loops(edge_index=edge_index)[0]
        print(f"Edge index shape (after add_self_loops): {edge_index.shape}")

        # Split data into train, val, and test sets
        num_nodes = x.size(0)
        shuffled_indices = torch.randperm(num_nodes)

        print(f"Total nodes: {num_nodes}")
        num_train = int(self.config["training"]["train_split"] * num_nodes)
        num_val = int(self.config["training"]["val_split"] * num_nodes)
        print(f"Train size: {num_train}, Validation size: {num_val}, Test size: {num_nodes - num_train - num_val}")

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[shuffled_indices[:num_train]] = True
        val_mask[shuffled_indices[num_train:num_train + num_val]] = True
        test_mask[shuffled_indices[num_train + num_val:]] = True

        # Print mask shapes and counts
        print(f"Train mask shape: {train_mask.shape}, Count: {train_mask.sum().item()}")
        print(f"Validation mask shape: {val_mask.shape}, Count: {val_mask.sum().item()}")
        print(f"Test mask shape: {test_mask.shape}, Count: {test_mask.sum().item()}")

        data = Data(x=x, edge_index=edge_index, y=y,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        print(f"Data object: {data}")
        return data.to(self.device)
    
          
    def evaluate(self, model: GAT, data: Data, mask: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the GAT model using the specified data and mask.

        Args:
            model (GAT): The GAT model to evaluate.
            data (Data): The PyTorch Geometric Data object.
            mask (torch.Tensor): A boolean mask indicating which nodes to evaluate.

        Returns:
            Dict[str, float]: A dictionary of evaluation metrics.
        """
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            # Get model outputs
            outputs = model(data.x, data.edge_index)
            outputs = outputs[mask]  # Filter outputs based on mask
            targets = data.y[mask]  # Get ground truth labels for the masked nodes

            # For multi-class classification, use softmax for probabilities
            if outputs.size(-1) > 1:
                # evaluating softmax
                probs = torch.softmax(outputs, dim=-1)
                preds = outputs.argmax(dim=-1)
            else:
                # Evaluating sigmoid for binary classification
                probs = torch.sigmoid(outputs)  # Removed dim=-1
                preds = (probs > 0.5).long()  # Convert probabilities to binary predictions

            # Convert to CPU for metrics calculation
            targets_np = targets.cpu().numpy()
            preds_np = preds.cpu().numpy()
            probs_np = probs.cpu().numpy()

            # Compute metrics
            metrics = {
                "accuracy": accuracy_score(targets_np, preds_np),
                "f1_macro": f1_score(targets_np, preds_np, average="macro", zero_division=0),
                "f1_weighted": f1_score(targets_np, preds_np, average="weighted", zero_division=0),
                "precision_macro": precision_score(targets_np, preds_np, average="macro", zero_division=0),
                "recall_macro": recall_score(targets_np, preds_np, average="macro", zero_division=0),
            }


            #TODO Revise this for different datasets! Its always buggy for some datasets and way too big
            if outputs.size(-1) == 1:  # Binary classification
                # Ensure probs_np is 1D for the positive class
                probs_np = probs_np.squeeze()  # Make sure it's 1D
                if len(np.unique(targets_np)) > 1:  # Ensure at least two classes
                    metrics["auc"] = roc_auc_score(targets_np, probs_np)
                else:
                    metrics["auc"] = None  # AUC not defined for single-class
            elif outputs.size(-1) > 1:  # Multi-class classification
                if len(np.unique(targets_np)) > 1:  # Ensure at least two classes
                    
                    try:
                        if outputs.size(-1) > 1:  # Multi-class
                            probs_np = probs.cpu().numpy()
                            if probs_np.shape[1] > 1:
                                metrics["auc"] = roc_auc_score(targets_np, probs_np, multi_class="ovr")
                            else:
                                metrics["auc"] = None  # Skip if single-class
                        else:  # Binary classification
                            probs_np = probs.cpu().numpy().squeeze()  # Ensure 1D array
                            if len(np.unique(targets_np)) > 1:
                                metrics["auc"] = roc_auc_score(targets_np, probs_np)
                            else:
                                metrics["auc"] = None


                    except ValueError as e:
                        print(f"Skipping AUC calculation due to an error: {e}")
                        metrics["auc"] = None
                else:
                    metrics["auc"] = None  # AUC not defined for single-class
            #except ValueError as e:
            #    print(f"AUC calculation failed: {e}")
            #    metrics["auc"] = None
            #print(metrics)
            return metrics

    @staticmethod
    def make_bidirectional(edge_index):
        print(f"Original edge_index shape: {edge_index.shape}")
        edge_index = edge_index.t()
        print(f"Transposed edge_index shape: {edge_index.shape}")
        edge_index_reversed = edge_index.flip(0)
        print(f"Reversed edge_index shape: {edge_index_reversed.shape}")
        edge_index_bidirectional = torch.cat([edge_index, edge_index_reversed], dim=1)
        print(f"Concatenated bidirectional edge_index shape: {edge_index_bidirectional.shape}")
        edge_index_bidirectional = torch.unique(edge_index_bidirectional, dim=1)
        print(f"Unique bidirectional edge_index shape: {edge_index_bidirectional.shape}")
        return edge_index_bidirectional#.t() for not being cora


    def train_model(self, trial: optuna.Trial = None):
        
        data = self.prepare_data()
        print(f"Prepared data shapes:")
        print(f"x (features): {data.x.shape}, y (labels): {data.y.shape}, edge_index: {data.edge_index.shape}")
        print(f"Train mask: {data.train_mask.shape}, Val mask: {data.val_mask.shape}, Test mask: {data.test_mask.shape}")
        
        try:
         # Hyperparameter definition (either from config or trial)
            if trial is not None:
                hidden_channels = trial.suggest_int("hidden_channels", 4, 8)
                num_heads = trial.suggest_int("num_heads", 4, 8)
                dropout = trial.suggest_float("dropout", 0.1, 0.5)
                lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)

                print("Chosen parameters:")
                print(hidden_channels)
                print(num_heads)
                print(dropout)
                print(lr)


            else:
                hidden_channels = self.config["model"]["hidden_channels"]
                num_heads = self.config["model"]["num_heads"]
                dropout = self.config["model"]["dropout"]
                lr = self.config["optimizer"]["lr"]
            
            model = GAT(
                in_channels=data.x.size(1),
                hidden_channels=hidden_channels,
                out_channels= config["out_dimenstions"],  # CHANGE THIS FOR DATASET 18 for romans, 2 for tolokers
                num_heads=num_heads,
                dropout=dropout
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.CrossEntropyLoss()
            
            
        except Exception as e:
            print(f"Pre Trial failed due to: {e}")
            raise  # Allow Optuna to handle the exception
            
        best_val_acc = 0
        early_stopping_counter = 0
        
        # Start MLFlow run
        with mlflow.start_run():
            # Log parameters
            try:
                mlflow.log_params({
                    "hidden_channels": hidden_channels,
                    "num_heads": num_heads,
                    "dropout": dropout,
                    "learning_rate": lr
                })
            except Exception as e:
                print(f"MLflow logging failed: {e}")

            
            for epoch in range(self.config["training"]["epochs"]):
                
                # Training
                model.train()
                optimizer.zero_grad()

                #print("OUT!")
                out = model(data.x, data.edge_index)

                loss = criterion(out[data.train_mask], data.y[data.train_mask])
                
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    # Compute and log metrics
                    train_metrics = self.evaluate(model, data, data.train_mask)
                    val_metrics = self.evaluate(model, data, data.val_mask)
                    print(loss)
                    # Log all metrics with proper prefixes
                    metrics_to_log = {
                        f"train_{k}": v for k, v in train_metrics.items()
                    }
                    metrics_to_log.update({
                        f"val_{k}": v for k, v in val_metrics.items()
                    })
                    metrics_to_log["train_loss"] = loss.item()
                    
                    # Log to MLflow
                    mlflow.log_metrics(metrics_to_log, step=epoch)
                    
                   # Early stopping based on validation accuracy
                    if val_metrics['accuracy'] >= best_val_acc:
                        best_val_acc = val_metrics['accuracy']
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                        print(f"Early stopping counter incremented to {early_stopping_counter}")

                    # Check if early stopping criteria are met
                    if early_stopping_counter > self.config["training"]["early_stopping_patience"]:
                        logger.info(f"Early stopping triggered. Best validation accuracy: {best_val_acc}")
                        break
                    
                    # Log progress
                    logger.info(
                        f"Epoch {epoch:03d}, Loss: {loss:.4f}, "
                        f"Train Acc: {train_metrics['accuracy']:.4f}, "
                        f"Val Acc: {val_metrics['accuracy']:.4f}, "
                        f"Train F1: {train_metrics['f1_macro']:.4f}, "
                        f"Val F1: {val_metrics['f1_macro']:.4f}, "
                        f"ROC AUC F1: {val_metrics['auc']:.4f}"
                    )
            
            # Final evaluation on test set
            #test_metrics = self.evaluate(model, data, data.test_mask)
            #mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            
            # Save model
            mlflow.pytorch.log_model(model, "model")

                    # Ensure the result is valid
            if best_val_acc is None:
                raise ValueError("Validation accuracy is None.")

            
            return best_val_acc


def optimize(config: Dict[str, Any]):
    trainer = GATTrainer(config)
    study = optuna.create_study(
        study_name=config["hyperopt"]["study_name"],
        direction=config["hyperopt"]["direction"]
    )
    
    study.optimize(trainer.train_model, 
                  n_trials=config["hyperopt"]["n_trials"])
    
    logger.info(f"Best trial: {study.best_trial.params}")
    logger.info(f"Best accuracy: {study.best_trial.value}")

if __name__ == "__main__":


    from torch_geometric.datasets import Planetoid
    
    dataset = Planetoid(root='~/somewhere/Cora', name='Cora')

    # Load config from file if it exists, otherwise use default
    config_path = Path("config.yml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    #mlflow.set_tracking_uri("file:./mlruns")  # Use your tracking URI
    # Initialize MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Change the URI to the MLflow server
    mlflow.set_experiment(config["experiment_name"])

    # Run hyperparameter optimization
    optimize(config)



