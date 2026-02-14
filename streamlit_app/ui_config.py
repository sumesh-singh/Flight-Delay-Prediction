"""
UI Configuration and Constants
"""

# App metadata
APP_TITLE = "Flight Delay Prediction System"
APP_ICON = "✈️"
APP_SUBTITLE = "Phase 4: Inference & Model Comparison"

# Model types
MODELS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
}

# Validation rules
MIN_YEAR = 2020
MAX_YEAR = 2026

# Feature constants for inference
SAFE_DEFAULTS = {
    "prev_flight_delay": 0,
    "turnaround_time": 60,  # median minutes
    "turnaround_stress": 0,
    "same_day_carrier_delays": 0,
}

# UI text
MESSAGES = {
    "model_missing": """
    ❌ Model artifact missing: {path}
    
    To regenerate:
    1. cd experiments/
    2. python run_baseline_experiments.py
    
    This will create all required artifacts.
    """,
    "inference_success": "Prediction complete!",
    "validation_error": "Please correct the highlighted errors before proceeding.",
}

# Delays threshold (minutes)
DELAY_THRESHOLD = 15
