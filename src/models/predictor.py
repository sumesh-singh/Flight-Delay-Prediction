"""
Model Predictor - Inference-Only Wrapper
Loads trained models and makes predictions (no training)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import joblib
import json
import logging


class ModelPredictor:
    """
    Inference-only model wrapper

    Features:
    - Loads trained models from disk
    - Makes predictions on new data
    - Validates feature compatibility
    - Returns probabilities for classification

    Usage:
        predictor = ModelPredictor('models/random_forest/rf_model_20260210.joblib')
        predictions = predictor.predict(X_new)
    """

    def __init__(
        self,
        model_path: Path,
        feature_path: Optional[Path] = None,
        log_level: str = "INFO",
    ):
        """
        Initialize predictor

        Args:
            model_path: Path to saved model (.joblib)
            feature_path: Path to feature names JSON (optional)
            log_level: Logging level
        """
        self.model_path = Path(model_path)
        self.feature_path = Path(feature_path) if feature_path else None

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

        # Load model
        self.model = self._load_model()

        # Load feature names
        self.expected_features = self._load_features()

        self.logger.info(
            f"ModelPredictor initialized with model: {self.model_path.name}"
        )

    def _load_model(self) -> object:
        """Load model from disk"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        try:
            model = joblib.load(self.model_path)
            self.logger.info(f"✓ Model loaded: {self.model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _load_features(self) -> Optional[list]:
        """Load expected feature names"""
        # Try to auto-detect feature file
        if self.feature_path is None:
            # Look for feature file in same directory
            model_dir = self.model_path.parent
            model_stem = self.model_path.stem  # e.g., "rf_model_20260210"

            # Extract date from model name
            parts = model_stem.split("_")
            if len(parts) >= 3:
                date = parts[-1]  # Last part is date
                model_type = "_".join(parts[:-2])  # Everything except "model" and date

                feature_file = model_dir / f"{model_type}_features_{date}.json"
                if feature_file.exists():
                    self.feature_path = feature_file

        # Load features if path exists
        if self.feature_path and self.feature_path.exists():
            try:
                with open(self.feature_path, "r") as f:
                    data = json.load(f)
                    features = data.get("features", [])
                self.logger.info(
                    f"✓ Loaded {len(features)} expected features from {self.feature_path.name}"
                )
                return features
            except Exception as e:
                self.logger.warning(f"Could not load features: {e}")
                return None
        else:
            self.logger.warning(
                "No feature file found - cannot validate feature compatibility"
            )
            return None

    def validate_features(self, X: pd.DataFrame) -> bool:
        """
        Validate that input features match expected features

        Args:
            X: Input features

        Returns:
            True if valid, raises ValueError otherwise
        """
        if self.expected_features is None:
            self.logger.warning("No expected features loaded - skipping validation")
            return True

        input_features = X.columns.tolist()
        expected_features = self.expected_features

        # Check for missing features
        missing = set(expected_features) - set(input_features)
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")

        # Check for extra features
        extra = set(input_features) - set(expected_features)
        if extra:
            self.logger.warning(
                f"Extra features present (will be ignored): {sorted(extra)}"
            )

        # Check feature order
        if input_features != expected_features:
            self.logger.warning("Feature order differs - reordering to match training")
            # Reorder will be handled in predict()

        return True

    def predict(self, X: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features DataFrame
            return_proba: If True, return probabilities instead of labels

        Returns:
            Predictions array
        """
        # Validate features
        if self.expected_features is not None:
            self.validate_features(X)
            # Reorder to match expected features
            X = X[self.expected_features]

        # Make predictions
        if return_proba and hasattr(self.model, "predict_proba"):
            predictions = self.model.predict_proba(X)
        else:
            predictions = self.model.predict(X)

        self.logger.info(f"Generated {len(predictions):,} predictions")

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities (classification only)

        Args:
            X: Features DataFrame

        Returns:
            Probability array
        """
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Model does not support probability predictions")

        return self.predict(X, return_proba=True)

    def predict_with_threshold(
        self, X: pd.DataFrame, threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with custom probability threshold

        Args:
            X: Features DataFrame
            threshold: Probability threshold for positive class

        Returns:
            (predictions, probabilities)
        """
        # Get probabilities
        probabilities = self.predict_proba(X)

        # Apply threshold (for binary classification)
        if probabilities.ndim == 2:
            # Binary classification - use probability of positive class (column 1)
            pos_proba = probabilities[:, 1]
            predictions = (pos_proba >= threshold).astype(int)
        else:
            raise ValueError(
                "Custom threshold only supported for binary classification"
            )

        return predictions, probabilities


def load_model(model_path: Path, feature_path: Optional[Path] = None) -> ModelPredictor:
    """
    Convenience function to load model

    Args:
        model_path: Path to model file
        feature_path: Path to feature names JSON (optional)

    Returns:
        ModelPredictor instance
    """
    return ModelPredictor(model_path, feature_path)


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL PREDICTOR - Test")
    print("=" * 60)

    print("\nModelPredictor is ready for inference.")
    print("To use:")
    print("  predictor = ModelPredictor('path/to/model.joblib')")
    print("  predictions = predictor.predict(X_new)")

    print("\n" + "=" * 60)
