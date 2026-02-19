"""
Inference Component - Model Loading and Prediction
Uses optimal thresholds from training for accurate classification.
"""

import sys
import warnings
from pathlib import Path
import joblib
import json
import numpy as np
from typing import Tuple, Dict, Any

warnings.filterwarnings("ignore", message=".*fitted without feature names.*")

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from streamlit_app.feature_adapter import FeatureAdapter


class ModelPredictor:
    """Loads model and runs predictions using saved artifacts + optimal threshold."""

    def __init__(self, model_type: str):
        """
        Initialize predictor.

        Args:
            model_type: 'logistic_regression', 'random_forest', or 'sgd_classifier'
        """
        self.model_type = model_type
        self.model_dir = Path(f"models/{model_type}")

        # Load model
        self.model = self._load_model()
        self.metadata = self._load_metadata()
        self.optimal_threshold = self.metadata.get("optimal_threshold", 0.5)

        # Initialize feature adapter (handles scaling internally)
        self.feature_adapter = FeatureAdapter(model_type)

    def _load_model(self):
        """Load trained model."""
        model_files = list(self.model_dir.glob(f"{self.model_type}_model_*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"Model file not found in {self.model_dir}")

        return joblib.load(model_files[0])

    def _load_metadata(self) -> Dict:
        """Load model metadata."""
        metadata_path = self.model_dir / "metadata.json"
        if not metadata_path.exists():
            return {}

        with open(metadata_path, "r") as f:
            return json.load(f)

    def predict(
        self,
        flight_date,
        dep_time: int,
        arr_time: int,
        carrier: str,
        origin: str,
        dest: str,
        distance: float = None,
    ) -> Tuple[str, float]:
        """
        Make prediction using optimal threshold.

        Args:
            flight_date: Date of flight
            dep_time: Scheduled departure time (HHMM)
            arr_time: Scheduled arrival time (HHMM)
            carrier: Airline code
            origin: Origin airport code
            dest: Destination airport code
            distance: Route distance (optional)

        Returns:
            (prediction_label, probability)
        """
        # Compute features (already scaled by FeatureAdapter)
        X = self.feature_adapter.create_inference_features(
            flight_date, dep_time, arr_time, carrier, origin, dest, distance
        )

        # Get probability
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            delay_prob = float(proba[1])
        elif hasattr(self.model, "decision_function"):
            raw = self.model.decision_function(X)[0]
            delay_prob = float(1 / (1 + np.exp(-raw)))  # sigmoid
        else:
            # Fallback to raw predict
            prediction = int(self.model.predict(X)[0])
            return ("Delayed" if prediction == 1 else "On-Time", 0.5)

        # Apply optimal threshold
        is_delayed = delay_prob >= self.optimal_threshold

        label = "Delayed" if is_delayed else "On-Time"
        # Confidence = how far from threshold boundary
        confidence = delay_prob if is_delayed else (1 - delay_prob)

        return label, confidence

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for display."""
        return {
            "type": self.model_type,
            "threshold": self.optimal_threshold,
            "is_best": self.metadata.get("is_best_model", False),
            "f1": self.metadata.get("metrics", {}).get("f1", "N/A"),
            "trained_on": self.metadata.get("trained_on", "Unknown"),
        }
