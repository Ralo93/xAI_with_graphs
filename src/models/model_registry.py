# src/models/model_registry.py
import mlflow
import torch
from pathlib import Path
from typing import Optional, Union
from .gat import GAT, GATConfig
from .cognn import CoGNN, CoGNNConfig


class ModelRegistry:
    DEFAULT_MODEL_PATH = Path("models/artifacts/model.pth")
    MLFLOW_MODEL_PATH = "model"  # MLflow model artifact path

    @staticmethod
    def save_model(model: Union[GAT, CoGNN], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)

    @staticmethod
    def load_model(config: Union[GATConfig, CoGNNConfig],
                model_path: Optional[Path] = None,
                model_type: str = 'gat') -> Union[GAT, CoGNN]:
        # Try MLflow first
        try:
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            latest_model = None
            
            for exp in experiments:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=["attributes.start_time DESC"]
                )
                if runs:
                    latest_model = f"runs:/{runs[0].info.run_id}/model"
                    break
                    
            if latest_model:
                return mlflow.pytorch.load_model(latest_model)
        except:
            pass

        # Fallback to local path if specified
        if model_path and model_path.exists():
            model = GAT(config) if model_type == 'gat' else CoGNN(config)
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            return model.eval()
        
        # Create new model if no saved model found
        model = GAT(config) if model_type == 'gat' else CoGNN(config)
        return model.eval()

    @staticmethod
    def get_latest_model(
        experiment_name: str, metric: str = "val_accuracy"
    ) -> Optional[str]:
        """Get path to best model from MLflow based on metric"""
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
        )

        if not runs:
            return None

        best_run = runs[0]
        return f"runs:/{best_run.info.run_id}/{ModelRegistry.MLFLOW_MODEL_PATH}"
