"""
Feature Adapter for Inference-Time Feature Computation

Bridges user inputs → training feature space
Supports all 45 features from the unified training pipeline.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, Any, List
import json
import pickle
import joblib

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit_app.ui_config import SAFE_DEFAULTS


class FeatureAdapter:
    """Computes all features needed for inference from minimal user inputs."""

    def __init__(self, model_type: str):
        """
        Initialize feature adapter.

        Args:
            model_type: 'logistic_regression', 'random_forest', or 'sgd_classifier'
        """
        self.model_type = model_type
        self.model_dir = Path(f"models/{model_type}")

        # Load artifacts
        self.feature_order = self._load_feature_order()
        self.label_encoders = self._load_label_encoders()
        self.historical_stats = self._load_historical_stats()
        self.scaler = self._load_scaler()

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

    def _load_scaler(self):
        """Load the fitted StandardScaler."""
        scaler_files = list(self.model_dir.glob(f"{self.model_type}_scaler_*.joblib"))
        if not scaler_files:
            return None
        return joblib.load(scaler_files[0])

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
        hour = dep_time // 100
        month = flight_date.month
        day_of_week = flight_date.weekday()  # 0=Monday, 6=Sunday
        day_of_month = flight_date.day
        quarter = (month - 1) // 3 + 1

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
        is_late_night = 1 if (hour >= 22 or hour <= 5) else 0

        # Holiday proximity check (simplified — major US holidays)
        major_holidays = [
            (1, 1),
            (1, 15),
            (2, 19),
            (5, 27),
            (7, 4),
            (9, 2),
            (10, 14),
            (11, 11),
            (11, 28),
            (12, 25),
            (12, 31),
        ]
        is_near_holiday = 0
        for h_month, h_day in major_holidays:
            try:
                from datetime import date as dt_date

                h_date = dt_date(flight_date.year, h_month, h_day)
                if abs((flight_date - h_date).days) <= 3:
                    is_near_holiday = 1
                    break
            except ValueError:
                continue

        return {
            "month": month,
            "Month": month,
            "day_of_week": day_of_week,
            "DayOfWeek": day_of_week + 1,  # BTS uses 1-indexed
            "day": day_of_month,
            "DayofMonth": day_of_month,
            "hour": hour,
            "quarter": quarter,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "is_weekend": is_weekend,
            "is_peak_hour": is_peak_hour,
            "is_near_holiday": is_near_holiday,
            "is_late_night_op": is_late_night,
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
        global_avg = self.historical_stats.get("global_avg_delay", 0.0)
        global_std = self.historical_stats.get("global_std_delay", 30.0)
        global_rate = self.historical_stats.get("global_delay_rate", 0.18)

        # Carrier features
        features["carrier_avg_arr_delay"] = self.historical_stats.get(
            f"carrier_{carrier}_avg_delay", global_avg
        )
        features["carrier_arr_delay_std"] = self.historical_stats.get(
            f"carrier_{carrier}_std_delay", global_std
        )
        features["carrier_delay_rate"] = self.historical_stats.get(
            f"carrier_{carrier}_delay_rate", global_rate
        )
        features["carrier_avg_dep_delay"] = features["carrier_avg_arr_delay"]

        # Route features
        route_key = f"route_{origin}_{dest}"
        features["route_avg_delay"] = self.historical_stats.get(
            f"{route_key}_avg_delay", global_avg
        )
        features["route_delay_rate"] = self.historical_stats.get(
            f"{route_key}_delay_rate", global_rate
        )
        features["route_avg_distance"] = 800  # approximate median

        # Airport features
        features["origin_flight_count"] = self.historical_stats.get(
            f"origin_{origin}_flight_count", 100
        )
        features["dest_flight_count"] = self.historical_stats.get(
            f"origin_{dest}_flight_count", 100
        )
        features["origin_delay_rate"] = self.historical_stats.get(
            f"origin_{origin}_delay_rate", global_rate
        )
        features["dest_delay_rate"] = self.historical_stats.get(
            f"origin_{dest}_delay_rate", global_rate
        )

        return features

    def compute_human_factors_features(self, dep_time: int) -> Dict[str, float]:
        """
        Compute human factors features with safe defaults.
        These are unknowable at booking time, so we use population averages.
        """
        hour = dep_time // 100
        is_late_night = 1 if (hour >= 22 or hour <= 5) else 0

        return {
            "aircraft_daily_legs": SAFE_DEFAULTS.get("aircraft_daily_legs", 3),
            "aircraft_leg_number": SAFE_DEFAULTS.get("aircraft_leg_number", 2),
            "crew_fatigue_index": SAFE_DEFAULTS.get("crew_fatigue_index", 0.5),
            "is_late_night_op": is_late_night,
            "origin_hourly_density": SAFE_DEFAULTS.get("origin_hourly_density", 10),
            "dest_hourly_density": SAFE_DEFAULTS.get("dest_hourly_density", 10),
            "aircraft_daily_util_min": SAFE_DEFAULTS.get(
                "aircraft_daily_util_min", 300
            ),
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
            distance: Route distance (optional, 800 used if missing)

        Returns:
            DataFrame with 1 row, scaled and in correct feature order
        """
        features = {}

        # Direct inputs
        features["CRS_DEP_TIME"] = dep_time
        features["CRS_ARR_TIME"] = arr_time

        # Distance
        if distance is None:
            distance = 800
        features["DISTANCE"] = distance
        features["DistanceGroup"] = min(int(distance / 250) + 1, 11)

        # Temporal features
        temporal = self.compute_temporal_features(flight_date, dep_time)
        features.update(temporal)

        # Historical features
        historical = self.lookup_historical_features(carrier, origin, dest)
        features.update(historical)

        # Network features (safe defaults)
        features["prev_flight_delay"] = SAFE_DEFAULTS.get("prev_flight_delay", 0)
        features["turnaround_stress"] = SAFE_DEFAULTS.get("turnaround_stress", 0)

        # Human factors features
        human = self.compute_human_factors_features(dep_time)
        features.update(human)

        # Encode categorical features
        for col_name in ["OP_CARRIER", "ORIGIN", "DEST"]:
            input_val = {"OP_CARRIER": carrier, "ORIGIN": origin, "DEST": dest}[
                col_name
            ]
            encoded_name = f"{col_name}_encoded"
            if col_name in self.label_encoders:
                try:
                    features[encoded_name] = int(
                        self.label_encoders[col_name].transform([input_val])[0]
                    )
                except (ValueError, KeyError):
                    features[encoded_name] = 0
            else:
                features[encoded_name] = 0

        # Weather features (benign defaults — unavailable at booking time)
        features["ORIGIN_TMAX"] = 70.0
        features["ORIGIN_PRCP"] = 0.0
        features["ORIGIN_AWND"] = 5.0
        features["ORIGIN_SNOW"] = 0.0
        features["DEST_TMAX"] = 70.0
        features["DEST_PRCP"] = 0.0
        features["DEST_AWND"] = 5.0
        features["DEST_SNOW"] = 0.0

        # Traffic feature
        features["ORIGIN_AIRPORT_TRAFFIC"] = SAFE_DEFAULTS.get(
            "ORIGIN_AIRPORT_TRAFFIC", 0
        )

        # Create DataFrame
        df = pd.DataFrame([features])

        # Fill any missing features with 0
        missing_features = [f for f in self.feature_order if f not in df.columns]
        if missing_features:
            for f in missing_features:
                df[f] = 0

        # Reorder to match training
        df = df[self.feature_order]

        # Apply scaler if available
        if self.scaler is not None:
            df = pd.DataFrame(
                self.scaler.transform(df),
                columns=self.feature_order,
            )

        # Validate
        assert len(df.columns) == len(self.feature_order), (
            f"Feature count mismatch: {len(df.columns)} vs {len(self.feature_order)}"
        )
        assert not df.isnull().any().any(), "NaN values detected in features"

        return df
