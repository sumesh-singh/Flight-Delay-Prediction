"""
Inference Component - Model Loading and Prediction
"""

import sys
from pathlib import Path
import joblib
import json
from typing import Tuple, Dict, Any

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from streamlit_app.feature_adapter import FeatureAdapter


class ModelPredictor:
    """Loads model and runs predictions."""

    def __init__(self, model_type: str):
        """
        Initialize predictor.

        Args:
            model_type: 'logistic_regression' or 'random_forest'
        """
        self.model_type = model_type
        self.model_dir = Path(f"models/{model_type}")

        # Load model
        self.model = self._load_model()
        self.metadata = self._load_metadata()

        # Initialize feature adapter
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
        Make prediction.

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
        # Compute features
        X = self.feature_adapter.create_inference_features(
            flight_date, dep_time, arr_time, carrier, origin, dest, distance
        )

        # Predict
        prediction = self.model.predict(X)[0]

        # Get probability if available
        try:
            proba = self.model.predict_proba(X)[0]
            probability = proba[1] if prediction == 1 else proba[0]
        except:
            probability = 0.5  # Default if model doesn't support predict_proba

        label = "Delayed" if prediction == 1 else "On-Time"

        return label, probability
