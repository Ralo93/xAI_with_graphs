from typing import Dict, Any
import optuna
import mlflow
from .trainer import GATTrainer, TrainingConfig
from ..models.gat import GAT, GATConfig


class HyperoptOptimizer:
    """Hyperparameter optimization using Optuna"""

    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.trainer = GATTrainer(TrainingConfig(**base_config["training"]))

    def optimize(self, data: Any) -> None:
        """Run hyperparameter optimization"""
        study = optuna.create_study(
            study_name=self.base_config["hyperopt"]["study_name"],
            direction=self.base_config["hyperopt"]["direction"],
        )

        study.optimize(
            lambda trial: self._objective(trial, data),
            n_trials=self.base_config["hyperopt"]["n_trials"],
        )

        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_accuracy", study.best_trial.value)

    def _objective(self, trial: optuna.Trial, data: Any) -> float:
        """Objective function for optimization"""
        # Define hyperparameter search space
        model_config = GATConfig(
            in_channels=data.x.size(1),
            hidden_channels=trial.suggest_int("hidden_channels", 4, 8),
            out_channels=self.base_config["out_dimensions"],
            num_heads=trial.suggest_int("num_heads", 4, 8),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
        )

        model = GAT(model_config).to(self.trainer.device)
        return self.trainer.train(model, data, trial)
