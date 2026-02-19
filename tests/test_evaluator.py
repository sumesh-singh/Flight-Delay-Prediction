"""
Unit Tests for Model Evaluator Module

Tests:
1. Classification metrics computation
2. Regression metrics computation
3. Metric value ranges
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from src.models.evaluator import ModelEvaluator


@pytest.fixture
def evaluator():
    return ModelEvaluator()


@pytest.fixture
def binary_classification_data():
    """Simple binary classification dataset."""
    np.random.seed(42)
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    return X, y


class TestClassificationMetrics:
    def test_all_metrics_present(self, evaluator, binary_classification_data):
        """Should compute all required classification metrics."""
        X, y = binary_classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:150], y[:150])

        metrics = evaluator.evaluate_model(
            model, X[150:], y[150:], task_type="classification"
        )

        required_metrics = ["accuracy", "precision", "recall", "f1"]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

    def test_accuracy_range(self, evaluator, binary_classification_data):
        """Accuracy should be between 0 and 1."""
        X, y = binary_classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:150], y[:150])

        metrics = evaluator.evaluate_model(
            model, X[150:], y[150:], task_type="classification"
        )

        assert 0 <= metrics["accuracy"] <= 1

    def test_f1_range(self, evaluator, binary_classification_data):
        """F1 should be between 0 and 1."""
        X, y = binary_classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:150], y[:150])

        metrics = evaluator.evaluate_model(
            model, X[150:], y[150:], task_type="classification"
        )

        assert 0 <= metrics["f1"] <= 1

    def test_confusion_matrix_present(self, evaluator, binary_classification_data):
        """Confusion matrix should be in metrics."""
        X, y = binary_classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:150], y[:150])

        metrics = evaluator.evaluate_model(
            model, X[150:], y[150:], task_type="classification"
        )

        assert "confusion_matrix" in metrics


class TestRegressionMetrics:
    def test_regression_metrics_computed(self, evaluator):
        """Should compute RMSE, MAE, R2 for regression."""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = X[:, 0] * 2 + X[:, 1] + np.random.randn(200) * 0.5

        model = LinearRegression()
        model.fit(X[:150], y[:150])

        metrics = evaluator.evaluate_model(
            model, X[150:], y[150:], task_type="regression"
        )

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics

    def test_rmse_non_negative(self, evaluator):
        """RMSE should always be non-negative."""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = X[:, 0] + np.random.randn(200)

        model = LinearRegression()
        model.fit(X[:150], y[:150])

        metrics = evaluator.evaluate_model(
            model, X[150:], y[150:], task_type="regression"
        )

        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0


class TestMetricLogging:
    def test_log_metrics_runs(self, evaluator, binary_classification_data):
        """log_metrics should not raise errors."""
        X, y = binary_classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:150], y[:150])

        metrics = evaluator.evaluate_model(
            model, X[150:], y[150:], task_type="classification"
        )

        # Should not raise
        evaluator.log_metrics(metrics, split="test")
