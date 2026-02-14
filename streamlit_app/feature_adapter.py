"""
Feature Adapter for Inference-Time Feature Computation

Bridges user inputs → training feature space
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, Any, List
import json
import pickle

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit_app.ui_config import SAFE_DEFAULTS


class FeatureAdapter:
    """Computes all features needed for inference from minimal user inputs."""

    def __init__(self, model_type: str):
        """
        Initialize feature adapter.

        Args:
            model_type: 'logistic_regression' or 'random_forest'
        """
        self.model_type = model_type
        self.model_dir = Path(f"models/{model_type}")

        # Load artifacts
        self.feature_order = self._load_feature_order()
        self.label_encoders = self._load_label_encoders()
        self.historical_stats = self._load_historical_stats()

    def _load_feature_order(self) -> List[str]:
        """Load feature order from training."""
        features_file = list(self.model_dir.glob(f"{self.model_type}_features_*.json"))
        if not features_file:
            raise FileNotFoundError(f"Feature order file not found in {self.model_dir}")

        with open(features_file[0], "r") as f:
            data = json.load(f)

        return data.get("features", [])

    def _load_label_encoders(self) -> Dict:
        """Load label encoders for categorical features."""
        encoder_path = self.model_dir / "label_encoders.pkl"
        if not encoder_path.exists():
            return {}

        with open(encoder_path, "rb") as f:
            return pickle.load(f)

    def _load_historical_stats(self) -> Dict:
        """Load historical statistics for lookup features."""
        stats_path = self.model_dir / "historical_stats.json"
        if not stats_path.exists():
            return {}

        with open(stats_path, "r") as f:
            return json.load(f)

    def compute_temporal_features(
        self, flight_date: datetime.date, dep_time: int
    ) -> Dict[str, float]:
        """
        Compute temporal features from date and time.

        Args:
            flight_date: Date of flight
            dep_time: Departure time in HHMM format (e.g., 1430)

        Returns:
            Dictionary of temporal features
        """
        # Extract components
        hour = dep_time // 100
        minute = dep_time % 100
        month = flight_date.month
        day_of_week = flight_date.weekday()  # 0=Monday, 6=Sunday
        day_of_month = flight_date.day

        # Cyclical encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        # Binary flags
        is_weekend = 1 if day_of_week >= 5 else 0
        is_peak_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0

        # Season (approximate)
        season = (month % 12 + 3) // 3  # 1=Spring, 2=Summer, 3=Fall, 4=Winter

        return {
            "month": month,
            "day_of_week": day_of_week,
            "day_of_month": day_of_month,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_of_week_sin": dow_sin,
            "day_of_week_cos": dow_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "is_weekend": is_weekend,
            "is_peak_hour": is_peak_hour,
            "season": season,
        }

    def lookup_historical_features(
        self, carrier: str, origin: str, dest: str
    ) -> Dict[str, float]:
        """
        Lookup historical aggregate features.

        Args:
            carrier: Airline code
            origin: Origin airport code
            dest: Destination airport code

        Returns:
            Dictionary of historical features
        """
        features = {}

        # Carrier features
        carrier_avg = self.historical_stats.get(
            f"carrier_{carrier}_avg_delay",
            self.historical_stats.get("global_avg_delay", 0.0),
        )
        carrier_std = self.historical_stats.get(
            f"carrier_{carrier}_std_delay",
            self.historical_stats.get("global_std_delay", 0.0),
        )
        carrier_rate = self.historical_stats.get(
            f"carrier_{carrier}_delay_rate",
            self.historical_stats.get("global_delay_rate", 0.0),
        )

        features["carrier_avg_arr_delay"] = carrier_avg
        features["carrier_arr_delay_std"] = carrier_std
        features["carrier_delay_rate"] = carrier_rate
        features["carrier_avg_dep_delay"] = carrier_avg

        # Route features
        route_key = f"route_{origin}_{dest}"
        features["route_avg_delay"] = self.historical_stats.get(
            f"{route_key}_avg_delay", self.historical_stats.get("global_avg_delay", 0.0)
        )
        features["route_delay_rate"] = self.historical_stats.get(
            f"{route_key}_delay_rate",
            self.historical_stats.get("global_delay_rate", 0.0),
        )

        # Airport features
        origin_count = self.historical_stats.get(f"origin_{origin}_flight_count", 100)
        dest_count = self.historical_stats.get(f"origin_{dest}_flight_count", 100)

        features["origin_flight_count"] = origin_count
        features["dest_flight_count"] = dest_count

        # Size categories (based on flight count)
        features["origin_size_category"] = (
            2 if origin_count > 500 else (1 if origin_count > 100 else 0)
        )
        features["dest_size_category"] = (
            2 if dest_count > 500 else (1 if dest_count > 100 else 0)
        )

        return features

    def compute_network_features(self) -> Dict[str, float]:
        """
        Compute network features (unavailable at inference - use safe defaults).

        Returns:
            Dictionary of network features with safe defaults
        """
        return {
            "prev_flight_delay": SAFE_DEFAULTS["prev_flight_delay"],
            "turnaround_time": SAFE_DEFAULTS["turnaround_time"],
            "turnaround_stress": SAFE_DEFAULTS["turnaround_stress"],
            "same_day_carrier_delays": SAFE_DEFAULTS["same_day_carrier_delays"],
        }

    def create_inference_features(
        self,
        flight_date: datetime.date,
        dep_time: int,
        arr_time: int,
        carrier: str,
        origin: str,
        dest: str,
        distance: float = None,
    ) -> pd.DataFrame:
        """
        Create all features needed for inference.

        Args:
            flight_date: Date of flight
            dep_time: Scheduled departure time (HHMM)
            arr_time: Scheduled arrival time (HHMM)
            carrier: Airline code
            origin: Origin airport code
            dest: Destination airport code
            distance: Route distance (optional, median used if missing)

        Returns:
            DataFrame with 1 row and all training features in correct order
        """
        features = {}

        # Direct inputs
        features["CRS_DEP_TIME"] = dep_time
        features["CRS_ARR_TIME"] = arr_time
        features["CANCELLED"] = 0
        features["DIVERTED"] = 0

        # Distance (use median if not provided)
        if distance is None:
            distance = 800  # median approx
        features["DISTANCE"] = distance
        features["route_distance"] = distance
        features["route_distance_normalized"] = distance / 3000  # normalize

        # Temporal features
        temporal = self.compute_temporal_features(flight_date, dep_time)
        features.update(temporal)

        # Historical features
        historical = self.lookup_historical_features(carrier, origin, dest)
        features.update(historical)

        # Network features (safe defaults)
        network = self.compute_network_features()
        features.update(network)

        # Encode categorical features
        if "OP_CARRIER" in self.label_encoders:
            try:
                features["OP_CARRIER_encoded"] = self.label_encoders[
                    "OP_CARRIER"
                ].transform([carrier])[0]
            except:
                features["OP_CARRIER_encoded"] = 0  # Unknown carrier
        else:
            features["OP_CARRIER_encoded"] = 0

        if "ORIGIN" in self.label_encoders:
            try:
                features["ORIGIN_encoded"] = self.label_encoders["ORIGIN"].transform(
                    [origin]
                )[0]
            except:
                features["ORIGIN_encoded"] = 0
        else:
            features["ORIGIN_encoded"] = 0

        if "DEST" in self.label_encoders:
            try:
                features["DEST_encoded"] = self.label_encoders["DEST"].transform(
                    [dest]
                )[0]
            except:
                features["DEST_encoded"] = 0
        else:
            features["DEST_encoded"] = 0

        # Weather features (set to median/mean values - unavailable at inference)
        features["ORIGIN_TMAX"] = 70.0
        features["ORIGIN_PRCP"] = 0.0
        features["ORIGIN_AWND"] = 5.0
        features["ORIGIN_SNOW"] = 0.0
        features["DEST_TMAX"] = 70.0
        features["DEST_PRCP"] = 0.0
        features["DEST_AWND"] = 5.0
        features["DEST_SNOW"] = 0.0

        # Traffic feature (unavailable at inference)
        features["ORIGIN_AIRPORT_TRAFFIC"] = 0

        # Create DataFrame with exact feature order
        df = pd.DataFrame([features])

        # Reorder columns to match training
        missing_features = [f for f in self.feature_order if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            for f in missing_features:
                df[f] = 0

        df = df[self.feature_order]

        # Validate
        assert len(df.columns) == len(self.feature_order), (
            f"Feature count mismatch: {len(df.columns)} vs {len(self.feature_order)}"
        )
        assert list(df.columns) == self.feature_order, "Feature order mismatch"
        assert not df.isnull().any().any(), "NaN values detected in features"

        return df
