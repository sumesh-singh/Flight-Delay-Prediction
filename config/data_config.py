"""
Data Configuration Module
Centralizes all data-related parameters for the Flight Delay Prediction System
"""

from pathlib import Path

# ============================================================================
# PROJECT ROOT & PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"  # BTS CSV files go here
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"  # OurAirports CSVs
CACHE_DIR = DATA_DIR / "cache"
EXTERNAL_DATA_DIR = DATA_DIR / "external"  # API credentials, downloaded data

# Model artifacts
MODELS_DIR = PROJECT_ROOT / "models"
LR_MODEL_DIR = MODELS_DIR / "logistic_regression"
RF_MODEL_DIR = MODELS_DIR / "random_forest"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

# Logs
LOGS_DIR = PROJECT_ROOT / "logs"

# ============================================================================
# BTS DATA CONFIGURATION
# ============================================================================

# Years and months to process (2024-2025 as per available datasets)
BTS_YEAR_RANGE = [2024, 2025]
BTS_MONTHS = list(range(1, 13))  # All months 1-12

# BTS download URL template
BTS_BASE_URL = "https://transtats.bts.gov/PREZIP/"
BTS_FILENAME_TEMPLATE = (
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# Essential BTS columns (subset of 109 columns)
BTS_ESSENTIAL_COLUMNS = [
    # Flight identifiers
    "FL_DATE",
    "OP_CARRIER",
    "OP_CARRIER_FL_NUM",
    # Origin and Destination
    "ORIGIN",
    "DEST",
    # Scheduled times
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    # Actual times
    "DEP_TIME",
    "ARR_TIME",
    "ACTUAL_ELAPSED_TIME",
    # Delays (TARGET VARIABLES)
    "DEP_DELAY",
    "ARR_DELAY",
    # Flight status
    "CANCELLED",
    "DIVERTED",
    # Distance
    "DISTANCE",
    # Delay causes (for feature engineering)
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
]

# ============================================================================
# AIRPORT METADATA CONFIGURATION
# ============================================================================

# OurAirports CSV URLs
OURAIRPORTS_BASE_URL = "https://davidmegginson.github.io/ourairports-data/"
AIRPORT_METADATA_URLS = {
    "airports": f"{OURAIRPORTS_BASE_URL}airports.csv",
    "runways": f"{OURAIRPORTS_BASE_URL}runways.csv",
    "countries": f"{OURAIRPORTS_BASE_URL}countries.csv",
}

# ============================================================================
# API CONFIGURATION
# ============================================================================

# NOAA Weather API
NOAA_API_BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/"
NOAA_API_TOKEN_FILE = EXTERNAL_DATA_DIR / "APIs.txt"  # Store API token here
NOAA_RATE_LIMIT = 1000  # requests per day

# OpenSky Network API (for future global flight data)
OPENSKY_API_URL = "https://opensky-network.org"
OPENSKY_CREDENTIALS_FILE = EXTERNAL_DATA_DIR / "APIs.txt"

# ============================================================================
# DATA CLEANSING PARAMETERS
# ============================================================================

# Delay threshold for binary classification (minutes)
DELAY_THRESHOLD = 15  # ARR_DELAY > 15 minutes = DELAYED

# Outlier detection parameters (IEEE Table III methods)
OUTLIER_CONTAMINATION = 0.05  # 5% expected outliers
OUTLIER_DETECTION_COLUMNS = ["DEP_DELAY", "ARR_DELAY", "DISTANCE"]

# Missing value handling strategy
MISSING_VALUE_STRATEGY = {
    "critical_columns": ["DEP_DELAY", "ARR_DELAY"],  # Drop if missing
    "delay_causes": [
        "CARRIER_DELAY",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
    ],  # Fill with 0
    "temporal_columns": ["DEP_TIME", "ARR_TIME"],  # Drop if missing
}

# Multiclass delay categories (for future extension)
DELAY_CATEGORIES = {
    "bins": [-float("inf"), 0, 15, 45, float("inf")],
    "labels": ["Early/OnTime", "Minor", "Moderate", "Severe"],
}

# ============================================================================
# TRAIN/TEST SPLIT CONFIGURATION
# ============================================================================

# Temporal split (no shuffle to prevent data leakage)
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test
RANDOM_STATE = 42  # For reproducibility
SHUFFLE = False  # Temporal split (chronological order)

# Time Series Cross-Validation
TIME_SERIES_CV_FOLDS = 5

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================

# Peak hour ranges (for temporal features)
PEAK_HOURS = [
    (7 * 60, 9 * 60),  # 7:00 AM - 9:00 AM (morning rush)
    (17 * 60, 19 * 60),  # 5:00 PM - 7:00 PM (evening rush)
]

# Cyclical encoding columns
CYCLICAL_FEATURES = {
    "hour": 24,  # Hours in a day
    "month": 12,  # Months in a year
    "day_of_week": 7,  # Days in a week
}

# Categorical columns to encode
CATEGORICAL_COLUMNS = ["OP_CARRIER", "ORIGIN", "DEST"]

# ============================================================================
# DATA VALIDATION RULES
# ============================================================================

# Acceptable ranges for validation
VALIDATION_RULES = {
    "ARR_DELAY": {"min": -120, "max": 1500},  # -2 hours to 25 hours
    "DEP_DELAY": {"min": -120, "max": 1500},
    "DISTANCE": {"min": 0, "max": 5000},  # Max domestic flight distance
    "CRS_ELAPSED_TIME": {"min": 0, "max": 1440},  # Max 24 hours
}

# ============================================================================
# BATCH PROCESSING CONFIGURATION
# ============================================================================

# Chunk size for large file processing
CHUNK_SIZE = 100000  # Process 100K records at a time

# Parquet compression for processed data
PARQUET_COMPRESSION = "snappy"

# Cache settings
ENABLE_CACHE = True
CACHE_EXPIRY_DAYS = 7

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "data_pipeline.log"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_api_token(service: str = "noaa") -> str:
    """
    Read API token from APIs.txt file

    Args:
        service: Service name ('noaa' or 'opensky')

    Returns:
        API token string

    Raises:
        FileNotFoundError: If APIs.txt file doesn't exist
        ValueError: If token format is invalid or token not found
    """
    try:
        with open(NOAA_API_TOKEN_FILE, "r") as f:
            lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Check if this line contains the service we're looking for
                if service.lower() in line.lower():
                    # Expected format: "ServiceName: credentials"
                    if ":" not in line:
                        raise ValueError(
                            f"Invalid format in APIs.txt line {line_num}: '{line}'. "
                            f"Expected format: 'ServiceName: credentials'"
                        )

                    try:
                        # Split only on first colon to preserve credentials with colons
                        parts = line.split(":", 1)
                        if len(parts) != 2:
                            raise ValueError(f"Malformed line {line_num}")

                        token = parts[1].strip()
                        if not token:
                            raise ValueError(
                                f"Empty credentials for {service} in APIs.txt line {line_num}"
                            )

                        return token

                    except (IndexError, AttributeError) as e:
                        raise ValueError(
                            f"Error parsing APIs.txt line {line_num}: {str(e)}"
                        ) from e

    except FileNotFoundError:
        raise FileNotFoundError(
            f"API credentials file not found at {NOAA_API_TOKEN_FILE}. "
            f"Please create it with format: 'ServiceName: credentials'"
        ) from None

    # If we get here, token wasn't found
    raise ValueError(
        f"API token for '{service}' not found in {NOAA_API_TOKEN_FILE}. "
        f"Please add a line like: '{service.upper()}: your_token_here'"
    )


def ensure_directories():
    """
    Create all necessary directories if they don't exist
    """
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        METADATA_DIR,
        CACHE_DIR,
        EXTERNAL_DATA_DIR,
        LR_MODEL_DIR,
        RF_MODEL_DIR,
        ARTIFACTS_DIR,
        LOGS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("✓ All directories ensured")


if __name__ == "__main__":
    # Test configuration by printing key paths
    print("=" * 60)
    print("DATA CONFIGURATION TEST")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"BTS Year Range: {BTS_YEAR_RANGE}")
    print(f"Delay Threshold: {DELAY_THRESHOLD} minutes")
    print(f"Train/Test Split: {TRAIN_TEST_SPLIT}")
    print(f"Essential Columns: {len(BTS_ESSENTIAL_COLUMNS)}")
    print("=" * 60)

    # Ensure all directories exist
    ensure_directories()
