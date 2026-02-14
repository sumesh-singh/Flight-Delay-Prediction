"""
Model Trainer - Unified Training Pipeline
Supports Logistic Regression and Random Forest with temporal data splitting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import logging
import json
import joblib
import hashlib
import time

# Sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Local imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.model_config import (
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    TRAINING_CONFIG,
    EXPERIMENT_TRACKING,
    get_model_config,
)
from config.data_config import MODELS_DIR
from src.features.target_generator import prepare_training_data
from src.models.evaluator import ModelEvaluator


class ModelTrainer:
    """
    Unified model trainer for Logistic Regression and Random Forest

    Features:
    - Temporal data splitting (no shuffling)
    - Target generation via target_generator.py
    - Model training with time tracking
    - Feature importance extraction (RF only)
    - Model persistence
    - Experiment logging (conforms to experiment_schema.json)

    Usage:
        trainer = ModelTrainer(model_type='random_forest')
        results = trainer.train_and_evaluate(df)
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        target_type: str = "binary",
        test_size: float = 0.2,
        log_level: str = "INFO",
    ):
        """
        Initialize trainer

        Args:
            model_type: 'logistic_regression' or 'random_forest'
            target_type: 'binary', 'multiclass', or 'regression' (binary is primary)
            test_size: Fraction for test set (default: 0.2 = 80/20 split)
            log_level: Logging level
        """
        # Validate model type
        valid_types = ["logistic_regression", "random_forest"]
        if model_type not in valid_types:
            raise ValueError(
                f"model_type must be one of {valid_types}, got '{model_type}'"
            )

        self.model_type = model_type
        self.target_type = target_type
        self.test_size = test_size

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Get model configuration
        self.config = get_model_config(model_type)
        self.model_params = self.config["params"]
        self.training_config = self.config["training_config"]

        # Initialize model
        self.model = None
        self.evaluator = ModelEvaluator(log_level=log_level)

        # Experiment tracking
        self.experiment_log = {}

        self.logger.info(
            f"ModelTrainer initialized: {model_type}, target={target_type}"
        )

    def temporal_split(
        self, df: pd.DataFrame, date_column: str = "FL_DATE"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Temporal train/test split (NO SHUFFLING)

        Args:
            df: Full dataset with FL_DATE column
            date_column: Column to use for temporal sorting

        Returns:
            (train_df, test_df)
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TEMPORAL DATA SPLIT")
        self.logger.info("=" * 60)

        # Ensure data is sorted by date
        if date_column in df.columns:
            df = df.sort_values(date_column).reset_index(drop=True)
            self.logger.info(f"Sorted by {date_column}")

            # Log date range
            date_min = pd.to_datetime(df[date_column]).min()
            date_max = pd.to_datetime(df[date_column]).max()
            self.logger.info(f"Date range: {date_min.date()} to {date_max.date()}")
        else:
            self.logger.warning(f"{date_column} not found - splitting by row order")

        # Calculate split point
        split_idx = int(len(df) * (1 - self.test_size))

        # Split
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        self.logger.info(
            f"\nSplit ratio: {1 - self.test_size:.0%} / {self.test_size:.0%}"
        )
        self.logger.info(f"Train set: {len(train_df):,} rows")
        self.logger.info(f"Test set:  {len(test_df):,} rows")

        if date_column in df.columns:
            train_end = pd.to_datetime(train_df[date_column]).max()
            test_start = pd.to_datetime(test_df[date_column]).min()
            self.logger.info(
                f"\nTrain period: {pd.to_datetime(train_df[date_column]).min().date()} to {train_end.date()}"
            )
            self.logger.info(
                f"Test period:  {test_start.date()} to {pd.to_datetime(test_df[date_column]).max().date()}"
            )

            # Verify no temporal overlap
            if train_end >= test_start:
                self.logger.warning(
                    f"⚠️  Train end ({train_end.date()}) >= Test start ({test_start.date()})"
                )

        self.logger.info("=" * 60)

        return train_df, test_df

    def prepare_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple:
        """
        Prepare features and targets

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame

        Returns:
            (X_train, X_test, y_train, y_test, feature_names)
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("DATA PREPARATION")
        self.logger.info("=" * 60)

        # Generate features and targets using target_generator
        self.logger.info("Preparing training data...")
        X_train, y_train = prepare_training_data(
            train_df, target_type=self.target_type, exclude_arr_delay=True
        )

        self.logger.info("Preparing test data...")
        X_test, y_test = prepare_training_data(
            test_df, target_type=self.target_type, exclude_arr_delay=True
        )

        # Get feature names
        feature_names = X_train.columns.tolist()

        self.logger.info(f"\nFeature matrix:")
        self.logger.info(f"  X_train: {X_train.shape}")
        self.logger.info(f"  X_test:  {X_test.shape}")
        self.logger.info(f"  Features: {len(feature_names)}")

        self.logger.info(f"\nTarget distribution:")
        if self.target_type == "binary":
            train_delay_rate = y_train.mean()
            test_delay_rate = y_test.mean()
            self.logger.info(
                f"  Train delay rate: {train_delay_rate * 100:.1f}% ({y_train.sum():,} delayed)"
            )
            self.logger.info(
                f"  Test delay rate:  {test_delay_rate * 100:.1f}% ({y_test.sum():,} delayed)"
            )

        self.logger.info("=" * 60)

        return X_train, X_test, y_train, y_test, feature_names

    def create_model(self) -> object:
        """
        Create model instance with configured parameters

        Returns:
            Untrained model
        """
        if self.model_type == "logistic_regression":
            model = LogisticRegression(**self.model_params)
            self.logger.info(
                f"Created LogisticRegression with params: {self.model_params}"
            )

        elif self.model_type == "random_forest":
            model = RandomForestClassifier(**self.model_params)
            self.logger.info(
                f"Created RandomForestClassifier with params: {self.model_params}"
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return model

    def train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[object, float]:
        """
        Train model

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            (trained_model, training_time_sec)
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"TRAINING {self.model_type.upper()}")
        self.logger.info("=" * 60)

        # Memory warning for large datasets
        n_samples = len(X_train)
        if n_samples > 1_000_000:
            self.logger.warning(
                f"⚠️  Large dataset: {n_samples:,} samples. Memory usage may be high."
            )

        # Create model
        model = self.create_model()

        # Train with timing
        self.logger.info(f"Training on {n_samples:,} samples...")
        start_time = time.time()

        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            self.logger.info(
                f"✓ Training complete in {training_time:.2f} seconds ({training_time / 60:.2f} minutes)"
            )

        except MemoryError as e:
            self.logger.error(f"❌ Out of memory during training: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            raise

        # Store model
        self.model = model

        self.logger.info("=" * 60)

        return model, training_time

    def extract_feature_importance(self, feature_names: list) -> Optional[list]:
        """
        Extract feature importance (Random Forest only)

        Args:
            feature_names: List of feature names

        Returns:
            List of {feature, importance} dicts (top 10) or None
        """
        if self.model is None:
            self.logger.warning("No model available for feature importance extraction")
            return None

        # Random Forest has feature_importances_
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_

            # Create list of {feature, importance}
            importance_list = [
                {"feature": name, "importance": float(imp)}
                for name, imp in zip(feature_names, importances)
            ]

            # Sort by importance
            importance_list = sorted(
                importance_list, key=lambda x: x["importance"], reverse=True
            )

            # Log top 10
            self.logger.info("\nTop 10 Feature Importances:")
            for i, item in enumerate(importance_list[:10], 1):
                self.logger.info(
                    f"  {i:2d}. {item['feature']:30s} {item['importance']:.4f}"
                )

            return importance_list[:10]  # Return top 10conforming to schema

        # Logistic Regression has coefficients (but not "importance" in schema)
        elif hasattr(self.model, "coef_"):
            self.logger.info(
                "Logistic Regression coefficients available (not in schema as importance)"
            )
            return None

        else:
            self.logger.warning("Model does not support feature importance")
            return None

    def save_model(
        self, model: object, feature_names: list, dataset_version: str = "unknown"
    ) -> Path:
        """
        Save model to disk

        Args:
            model: Trained model
            feature_names: List of feature names
            dataset_version: Dataset version identifier

        Returns:
            Path to saved model
        """
        # Create model directory
        model_dir = MODELS_DIR / self.model_type
        model_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        model_path = model_dir / f"{self.model_type}_model_{timestamp}.joblib"

        # Save model
        joblib.dump(model, model_path)
        self.logger.info(f"✓ Model saved to: {model_path}")

        # Save feature names separately (for inference)
        feature_path = model_dir / f"{self.model_type}_features_{timestamp}.json"
        with open(feature_path, "w") as f:
            json.dump({"features": feature_names}, f, indent=2)
        self.logger.info(f"✓ Feature names saved to: {feature_path}")

        return model_path

    def compute_data_hash(self, X: pd.DataFrame, y: np.ndarray) -> str:
        """
        Compute SHA256 hash of dataset for reproducibility

        Args:
            X: Features
            y: Labels

        Returns:
            Hex digest of dataset hash
        """
        # Concatenate X and y
        data_str = str(X.values.tobytes()) + str(y.values.tobytes())
        hash_obj = hashlib.sha256(data_str.encode())
        return hash_obj.hexdigest()[:16]  # First 16 chars

    def log_experiment(
        self,
        dataset_version: str,
        n_samples_train: int,
        n_samples_test: int,
        n_features: int,
        feature_names: list,
        hyperparameters: dict,
        metrics_train: dict,
        metrics_test: dict,
        training_time_sec: float,
        feature_importance: Optional[list],
        confusion_matrix: list,
        data_hash: str,
    ) -> Dict:
        """
        Create experiment log conforming to experiment_schema.json

        Returns:
            Experiment log dict
        """
        # Generate run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_abbrev = "lr" if self.model_type == "logistic_regression" else "rf"
        run_id = f"run_{timestamp}_{model_abbrev}_default"

        # Model type for schema
        model_type_schema = {
            "logistic_regression": "LogisticRegression",
            "random_forest": "RandomForestClassifier",
        }[self.model_type]

        # Build experiment log
        experiment_log = {
            "run_id": run_id,
            "model_type": model_type_schema,
            "dataset_version": dataset_version,
            "n_samples_train": n_samples_train,
            "n_samples_test": n_samples_test,
            "n_features": n_features,
            "feature_list": feature_names,
            "hyperparameters": hyperparameters,
            "metrics": {
                "train": metrics_train,
                "test": metrics_test,
                "training_time_sec": training_time_sec,
            },
            "confusion_matrix": confusion_matrix,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "config_version": "model_config_v1.0",
            "data_hash": data_hash,
        }

        # Add feature importance (RF only)
        if feature_importance is not None:
            experiment_log["feature_importance_top10"] = feature_importance

        # Add OOB score (RF only)
        if hasattr(self.model, "oob_score_"):
            experiment_log["metrics"]["oob_score"] = float(self.model.oob_score_)

        return experiment_log

    def save_experiment_log(self, experiment_log: Dict) -> Path:
        """
        Save experiment log to JSON

        Args:
            experiment_log: Experiment log dict

        Returns:
            Path to saved log
        """
        # Create log directory
        log_dir = EXPERIMENT_TRACKING["log_dir"]
        log_dir.mkdir(parents=True, exist_ok=True)

        # Save log
        log_path = log_dir / f"{experiment_log['run_id']}.json"
        with open(log_path, "w") as f:
            json.dump(experiment_log, f, indent=2)

        self.logger.info(f"✓ Experiment log saved to: {log_path}")

        return log_path

    def train_and_evaluate(
        self, df: pd.DataFrame, dataset_version: str = "features_unknown.parquet"
    ) -> Dict:
        """
        Complete training and evaluation pipeline

        Args:
            df: Full dataset with features and FL_DATE
            dataset_version: Dataset version identifier

        Returns:
            Complete experiment results dict
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info(f"MODEL TRAINING PIPELINE: {self.model_type.upper()}")
        self.logger.info("=" * 70)

        # Step 1: Temporal split
        train_df, test_df = self.temporal_split(df)

        # Step 2: Prepare data
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(
            train_df, test_df
        )

        # Step 3: Train model
        model, training_time = self.train(X_train, y_train)

        # Step 4: Evaluate on both sets
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EVALUATION")
        self.logger.info("=" * 60)

        # Train metrics
        self.logger.info("\nEvaluating on training set...")
        metrics_train = self.evaluator.evaluate_model(
            model, X_train, y_train, task_type="classification"
        )
        self.evaluator.log_metrics(metrics_train, split="train")

        # Test metrics
        self.logger.info("\nEvaluating on test set...")
        metrics_test = self.evaluator.evaluate_model(
            model, X_test, y_test, task_type="classification"
        )
        self.evaluator.log_metrics(metrics_test, split="test")

        # Step 5: Feature importance
        feature_importance = self.extract_feature_importance(feature_names)

        # Step 6: Save model
        model_path = self.save_model(model, feature_names, dataset_version)

        # Step 7: Compute data hash
        data_hash = self.compute_data_hash(X_train, y_train)

        # Step 8: Log experiment
        experiment_log = self.log_experiment(
            dataset_version=dataset_version,
            n_samples_train=len(X_train),
            n_samples_test=len(X_test),
            n_features=len(feature_names),
            feature_names=feature_names,
            hyperparameters=self.model_params,
            metrics_train={
                k: v for k, v in metrics_train.items() if k != "confusion_matrix_dict"
            },
            metrics_test={
                k: v for k, v in metrics_test.items() if k != "confusion_matrix_dict"
            },
            training_time_sec=training_time,
            feature_importance=feature_importance,
            confusion_matrix=metrics_test["confusion_matrix"],
            data_hash=data_hash,
        )

        log_path = self.save_experiment_log(experiment_log)

        # Summary
        self.logger.info("\n" + "=" * 70)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"Model: {self.model_type}")
        self.logger.info(f"Test F1-Score: {metrics_test['f1']:.4f}")
        self.logger.info(f"Test Accuracy: {metrics_test['accuracy']:.4f}")
        self.logger.info(f"Training Time: {training_time:.2f}s")
        self.logger.info(f"Model saved: {model_path}")
        self.logger.info(f"Log saved: {log_path}")
        self.logger.info("=" * 70)

        return {
            "model": model,
            "model_path": model_path,
            "metrics_train": metrics_train,
            "metrics_test": metrics_test,
            "training_time": training_time,
            "experiment_log": experiment_log,
            "log_path": log_path,
        }


# Convenience functions


def train_logistic_regression(
    df: pd.DataFrame, dataset_version: str = "features_unknown.parquet"
) -> Dict:
    """Quick LR training"""
    trainer = ModelTrainer(model_type="logistic_regression")
    return trainer.train_and_evaluate(df, dataset_version=dataset_version)


def train_random_forest(
    df: pd.DataFrame, dataset_version: str = "features_unknown.parquet"
) -> Dict:
    """Quick RF training"""
    trainer = ModelTrainer(model_type="random_forest")
    return trainer.train_and_evaluate(df, dataset_version=dataset_version)


if __name__ == "__main__":
    print("=" * 70)
    print("MODEL TRAINER - Configuration Test")
    print("=" * 70)

    # Test configuration loading
    for model_type in ["logistic_regression", "random_forest"]:
        print(f"\n{model_type.upper()}:")
        trainer = ModelTrainer(model_type=model_type)
        print(f"  Parameters: {len(trainer.model_params)} settings")
        print(f"  Random state: {trainer.model_params['random_state']}")

    print("\n" + "=" * 70)
    print("✓ Configuration test complete")
    print("=" * 70)
