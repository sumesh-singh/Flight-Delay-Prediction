"""
Model Evaluator - Metric Computation Utilities
Computes classification and regression metrics conforming to experiment_schema.json
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import logging


class ModelEvaluator:
    """
    Evaluates model performance with standard metrics

    Supports:
    - Binary classification (primary)
    - Multiclass classification (design-ready)
    - Regression (design-ready)

    All outputs conform to experiment_schema.json format
    """

    def __init__(self, log_level: str = "INFO"):
        """
        Initialize evaluator

        Args:
            log_level: Logging level
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute classification metrics

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for ROC-AUC)

        Returns:
            Metrics dict conforming to experiment_schema.json:
            {
                "accuracy": float,
                "precision": float,
                "recall": float,
                "f1": float,
                "roc_auc": float (optional)
            }
        """
        metrics = {}

        try:
            # Core metrics (required by schema)
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["precision"] = float(
                precision_score(y_true, y_pred, average="binary", zero_division=0)
            )
            metrics["recall"] = float(
                recall_score(y_true, y_pred, average="binary", zero_division=0)
            )
            metrics["f1"] = float(
                f1_score(y_true, y_pred, average="binary", zero_division=0)
            )

            # ROC-AUC (if probabilities provided)
            if y_prob is not None:
                try:
                    # For binary classification, use probabilities of positive class
                    if y_prob.ndim == 2:
                        y_prob = y_prob[:, 1]
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
                except ValueError as e:
                    self.logger.warning(f"Could not compute ROC-AUC: {e}")
                    metrics["roc_auc"] = None

            self.logger.info(f"Classification metrics computed: {len(metrics)} metrics")

        except Exception as e:
            self.logger.error(f"Error computing classification metrics: {e}")
            raise

        return metrics

    def compute_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute confusion matrix in dict format

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Dict with {TP, FP, FN, TN} and matrix format [[TN, FP], [FN, TP]]
        """
        try:
            # Get confusion matrix (sklearn format: [[TN, FP], [FN, TP]])
            cm = confusion_matrix(y_true, y_pred)

            # Extract components
            tn, fp, fn, tp = cm.ravel()

            return {
                "dict": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
                "matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
            }

        except Exception as e:
            self.logger.error(f"Error computing confusion matrix: {e}")
            raise

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute regression metrics (design-ready, not in schema yet)

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Metrics dict:
            {
                "rmse": float,
                "mae": float,
                "r2": float
            }
        """
        metrics = {}

        try:
            # Root Mean Squared Error
            mse = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = float(np.sqrt(mse))

            # Mean Absolute Error
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))

            # R² Score
            metrics["r2"] = float(r2_score(y_true, y_pred))

            self.logger.info(
                f"Regression metrics computed: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R²={metrics['r2']:.4f}"
            )

        except Exception as e:
            self.logger.error(f"Error computing regression metrics: {e}")
            raise

        return metrics

    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        task_type: str = "classification",
    ) -> Dict:
        """
        Complete model evaluation

        Args:
            model: Trained model with predict() and predict_proba() methods
            X_test: Test features
            y_test: Test labels
            task_type: 'classification' or 'regression'

        Returns:
            Complete metrics dict
        """
        self.logger.info(f"Evaluating {task_type} model...")

        # Get predictions
        y_pred = model.predict(X_test)

        if task_type == "classification":
            # Get probabilities if available
            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)
                except Exception as e:
                    self.logger.warning(f"Could not get prediction probabilities: {e}")

            # Compute classification metrics
            metrics = self.evaluate_classification(y_test, y_pred, y_prob)

            # Add confusion matrix
            cm_result = self.compute_confusion_matrix(y_test, y_pred)
            metrics["confusion_matrix"] = cm_result["matrix"]
            metrics["confusion_matrix_dict"] = cm_result["dict"]

        elif task_type == "regression":
            metrics = self.evaluate_regression(y_test, y_pred)
        else:
            raise ValueError(
                f"Unknown task type: {task_type}. Use 'classification' or 'regression'"
            )

        self.logger.info(f"Evaluation complete")
        return metrics

    def compare_models(
        self,
        metrics1: Dict,
        metrics2: Dict,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2",
    ) -> Dict:
        """
        Compare two models' metrics

        Args:
            metrics1: First model's metrics
            metrics2: Second model's metrics
            model1_name: Name of first model
            model2_name: Name of second model

        Returns:
            Comparison dict with deltas
        """
        comparison = {"model1": model1_name, "model2": model2_name, "deltas": {}}

        # Compare common metrics
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if metric in metrics1 and metric in metrics2:
                if metrics1[metric] is not None and metrics2[metric] is not None:
                    delta = metrics2[metric] - metrics1[metric]
                    comparison["deltas"][metric] = {
                        model1_name: metrics1[metric],
                        model2_name: metrics2[metric],
                        "delta": delta,
                        "improvement_%": (delta / metrics1[metric] * 100)
                        if metrics1[metric] != 0
                        else 0,
                    }

        # Determine winner (by F1-score)
        if "f1" in comparison["deltas"]:
            if comparison["deltas"]["f1"]["delta"] > 0:
                comparison["winner"] = model2_name
                comparison["improvement"] = comparison["deltas"]["f1"]["improvement_%"]
            else:
                comparison["winner"] = model1_name
                comparison["improvement"] = -comparison["deltas"]["f1"]["improvement_%"]

        return comparison

    def log_metrics(self, metrics: Dict, split: str = "test"):
        """
        Log metrics in readable format

        Args:
            metrics: Metrics dictionary
            split: 'train' or 'test'
        """
        self.logger.info(f"\n{split.upper()} SET METRICS:")
        self.logger.info(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        self.logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
        self.logger.info(f"  Recall:    {metrics.get('recall', 0):.4f}")
        self.logger.info(f"  F1-Score:  {metrics.get('f1', 0):.4f}")

        if "roc_auc" in metrics and metrics["roc_auc"] is not None:
            self.logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        if "confusion_matrix_dict" in metrics:
            cm = metrics["confusion_matrix_dict"]
            self.logger.info(f"\n  Confusion Matrix:")
            self.logger.info(f"    TN: {cm['TN']:,}  FP: {cm['FP']:,}")
            self.logger.info(f"    FN: {cm['FN']:,}  TP: {cm['TP']:,}")


# Convenience functions


def evaluate_binary_classifier(model, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
    """
    Quick evaluation for binary classifier

    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels

    Returns:
        Metrics dict
    """
    evaluator = ModelEvaluator()
    return evaluator.evaluate_model(model, X_test, y_test, task_type="classification")


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> Dict:
    """
    Quick metrics computation

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)

    Returns:
        Metrics dict
    """
    evaluator = ModelEvaluator()
    return evaluator.evaluate_classification(y_true, y_pred, y_prob)


if __name__ == "__main__":
    """Test evaluator with synthetic data"""
    print("=" * 60)
    print("MODEL EVALUATOR TEST")
    print("=" * 60)

    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000

    # Simulate predictions
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = y_true.copy()
    # Add some errors
    error_idx = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    y_pred[error_idx] = 1 - y_pred[error_idx]

    # Simulate probabilities
    y_prob = np.column_stack(
        [
            1 - (y_pred + np.random.normal(0, 0.1, n_samples)).clip(0, 1),
            (y_pred + np.random.normal(0, 0.1, n_samples)).clip(0, 1),
        ]
    )

    # Evaluate
    evaluator = ModelEvaluator()

    print("\n1. Classification Metrics:")
    metrics = evaluator.evaluate_classification(y_true, y_pred, y_prob)
    evaluator.log_metrics(metrics, split="test")

    print("\n2. Confusion Matrix:")
    cm_result = evaluator.compute_confusion_matrix(y_true, y_pred)
    print(f"  Dict: {cm_result['dict']}")
    print(f"  Matrix: {cm_result['matrix']}")

    print("\n" + "=" * 60)
    print("✓ Evaluator test complete")
    print("=" * 60)
