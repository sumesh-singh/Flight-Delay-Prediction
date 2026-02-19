"""
UI Configuration and Constants
"""

# App metadata
APP_TITLE = "Flight Delay Prediction System"
APP_ICON = "✈️"
APP_SUBTITLE = "ML-Powered Flight Delay Prediction with SMOTE & Human Factors"

# Model types available in the sidebar
MODELS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "SGD Classifier": "sgd_classifier",
}

# Validation rules
MIN_YEAR = 2020
MAX_YEAR = 2026

# Feature constants for inference (safe defaults for features unavailable at prediction time)
SAFE_DEFAULTS = {
    # Network features (unknown at booking time)
    "prev_flight_delay": 0,
    "turnaround_stress": 0,
    # Human factors features (use average values)
    "aircraft_daily_legs": 3,
    "aircraft_leg_number": 2,
    "crew_fatigue_index": 0.5,
    "is_late_night_op": 0,
    "origin_hourly_density": 10,
    "dest_hourly_density": 10,
    "aircraft_daily_util_min": 300,
    # External features (use benign defaults)
    "ORIGIN_AIRPORT_TRAFFIC": 0,
}

# UI text
MESSAGES = {
    "model_missing": """
    ❌ Model artifact missing: {path}
    
    To regenerate:
    1. Run: python train_pipeline.py
    
    This will train all models and save required artifacts.
    """,
    "inference_success": "Prediction complete!",
    "validation_error": "Please correct the highlighted errors before proceeding.",
}

# Delays threshold (minutes)
DELAY_THRESHOLD = 15
