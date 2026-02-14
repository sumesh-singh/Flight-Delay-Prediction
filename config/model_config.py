"""
Model Configuration Module
Centralizes all model-related hyperparameters and training settings
Based on IEEE survey best practices and literature benchmarks
"""

from pathlib import Path
from typing import Dict, Any

# Import directory paths from data_config to avoid hardcoding
try:
    from .data_config import MODELS_DIR, LR_MODEL_DIR, RF_MODEL_DIR, ARTIFACTS_DIR
except ImportError:
    # Fallback for when running as standalone script
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from data_config import MODELS_DIR, LR_MODEL_DIR, RF_MODEL_DIR, ARTIFACTS_DIR

# ============================================================================
# MODEL TYPES
# ============================================================================

AVAILABLE_MODELS = ["logistic_regression", "random_forest"]
DEFAULT_MODEL = "random_forest"  # Primary model as per PRD

# ============================================================================
# LOGISTIC REGRESSION CONFIGURATION
# ============================================================================

LOGISTIC_REGRESSION_PARAMS = {
    # Regularization
    "penalty": "l2",  # L2 regularization (Ridge)
    "C": 1.0,  # Inverse of regularization strength
    # Optimization
    "solver": "lbfgs",  # Algorithm for optimization
    "max_iter": 1000,  # Maximum iterations
    "tol": 1e-4,  # Tolerance for stopping criterion
    # Reproducibility
    "random_state": 42,
    # Performance
    "n_jobs": -1,  # Use all CPU cores
    # Multiclass strategy (if extending to multiclass)
    "multi_class": "auto",
    # Verbosity
    "verbose": 0,
    # Class Weight
    "class_weight": "balanced",
}

# Logistic Regression training settings
LR_TRAINING_CONFIG = {
    "model_name": "logistic_regression_baseline",
    "expected_training_time": 60,  # seconds (for 1M records)
    "save_path": LR_MODEL_DIR / "lr_model.joblib",
    "metrics_path": LR_MODEL_DIR / "lr_metrics.json",
    "feature_importance_path": LR_MODEL_DIR / "lr_coefficients.csv",
}

# ============================================================================
# RANDOM FOREST CONFIGURATION
# ============================================================================

RANDOM_FOREST_PARAMS = {
    # Tree structure
    "n_estimators": 100,  # Number of trees in the forest
    "max_depth": 10,  # Maximum depth of each tree
    "min_samples_split": 50,  # Minimum samples to split a node
    "min_samples_leaf": 20,  # Minimum samples in a leaf node
    # Feature sampling
    "max_features": "sqrt",  # Number of features for best split
    "max_samples": None,  # Use all samples (bootstrap)
    # Bootstrap and bagging
    "bootstrap": True,  # Bootstrap samples for each tree
    "oob_score": True,  # Out-of-bag score estimation
    # Tree construction
    "criterion": "gini",  # Split criterion (gini or entropy)
    # Reproducibility
    "random_state": 42,
    # Performance
    "n_jobs": -1,  # Parallel processing (all cores)
    # Verbosity
    "verbose": 0,
    "warm_start": False,
    # Class Weight
    "class_weight": "balanced",
}

# Random Forest training settings
RF_TRAINING_CONFIG = {
    "model_name": "random_forest_primary",
    "expected_training_time": 300,  # seconds (5 minutes for 1M records)
    "save_path": RF_MODEL_DIR / "rf_model.joblib",
    "metrics_path": RF_MODEL_DIR / "rf_metrics.json",
    "feature_importance_path": RF_MODEL_DIR / "rf_feature_importance.csv",
}

# ============================================================================
# ALTERNATIVE HYPERPARAMETERS (FOR TUNING)
# ============================================================================

# Lighter Random Forest (faster training for prototyping)
RF_LIGHT_PARAMS = {
    **RANDOM_FOREST_PARAMS,
    "n_estimators": 50,
    "max_depth": 8,
    "min_samples_split": 100,
    "min_samples_leaf": 50,
}

# Heavy Random Forest (better performance, slower training)
RF_HEAVY_PARAMS = {
    **RANDOM_FOREST_PARAMS,
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
}

# ============================================================================
# HYPERPARAMETER TUNING CONFIGURATION
# ============================================================================

# Grid Search / Random Search parameter ranges
RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [8, 10, 12, 15],
    "min_samples_split": [20, 50, 100],
    "min_samples_leaf": [10, 20, 50],
    "max_features": ["sqrt", "log2"],
    "criterion": ["gini", "entropy"],
}

LOGISTIC_REGRESSION_PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "penalty": ["l1", "l2"],
    "solver": ["lbfgs", "liblinear", "saga"],
    "max_iter": [500, 1000, 2000],
}

# Hyperparameter tuning settings
HYPERPARAMETER_TUNING_CONFIG = {
    "search_method": "random",  # 'grid' or 'random'
    "cv_folds": 5,  # Cross-validation folds
    "n_iter": 20,  # Iterations for RandomizedSearchCV
    "scoring": "f1",  # Primary metric for optimization
    "n_jobs": -1,
    "verbose": 2,
    "random_state": 42,
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    # Data split
    "test_size": 0.2,  # 80/20 split
    "validation_size": 0.1,  # Optional validation set from training
    "shuffle": False,  # Temporal split (chronological)
    "stratify": None,  # No stratification for temporal data
    # Class balancing
    "handle_imbalance": True,  # Set True to use class weights
    "class_weight": "balanced",  # 'balanced' or None
    # Early stopping (for iterative models)
    "early_stopping": False,
    "patience": 5,
    # Checkpointing
    "save_checkpoints": True,
    "checkpoint_frequency": 10,  # Save every N epochs
    # Reproducibility
    "random_state": 42,
}

# ============================================================================
# EVALUATION METRICS CONFIGURATION
# ============================================================================

# IEEE Table VII standardized metrics
EVALUATION_METRICS = {
    "classification": [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "confusion_matrix",
        "classification_report",
    ],
    "regression": [
        "rmse",
        "mae",
        "r2_score",
        "mape",  # Mean Absolute Percentage Error
    ],
}

# Performance thresholds (from PRD success metrics)
PERFORMANCE_TARGETS = {
    "accuracy": 0.80,  # Target: >80%
    "f1_score": 0.75,  # Target: >0.75
    "rmse": 18.0,  # Target: <18 minutes
    "training_time": 600,  # Target: <10 minutes (seconds)
    "inference_latency": 0.1,  # Target: <100ms (seconds)
}

# IEEE literature benchmarks (for comparison)
IEEE_BENCHMARKS = {
    "accuracy_range": (0.75, 0.85),
    "f1_range": (0.70, 0.80),
    "rmse_range": (15, 25),  # minutes
    "reproducibility_rate": 0.10,  # <10% of papers provide code
}

# ============================================================================
# MODEL PERSISTENCE CONFIGURATION
# ============================================================================

MODEL_SAVE_CONFIG = {
    "format": "joblib",  # joblib or pickle
    "compression": 3,  # Compression level (0-9)
    "protocol": None,  # Pickle protocol version
    "save_metadata": True,  # Save training metadata
    "versioning": True,  # Enable model versioning
}

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

INFERENCE_CONFIG = {
    # Batch prediction
    "batch_size": 1000,  # Process N records at once
    # Probability thresholds
    "default_threshold": 0.5,  # Binary classification threshold
    "confidence_threshold": 0.7,  # High confidence predictions
    # Output format
    "return_probabilities": True,  # Include prediction probabilities
    "return_feature_importance": False,  # SHAP values (expensive)
    # Performance
    "use_cache": True,
    "max_cache_size": 10000,  # Cache last N predictions
}

# ============================================================================
# FEATURE SELECTION CONFIGURATION
# ============================================================================

FEATURE_SELECTION_CONFIG = {
    "method": "importance",  # 'importance', 'recursive', or 'l1'
    "n_features": None,  # None = use all, or specific number
    "importance_threshold": 0.01,  # Minimum feature importance
    "recursive_cv": 5,  # CV folds for recursive elimination
}

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

EXPERIMENT_TRACKING = {
    "enabled": True,
    "track_metrics": True,
    "track_parameters": True,
    "track_artifacts": True,
    "log_dir": MODELS_DIR.parent
    / "logs"
    / "experiments",  # PROJECT_ROOT/logs/experiments
    "experiment_name_template": "{model}_{timestamp}",
    "save_predictions": True,
    "save_errors": True,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    Get model configuration based on model type

    Args:
        model_type: 'logistic_regression' or 'random_forest'

    Returns:
        Dictionary with model parameters and training config
    """
    if model_type not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {AVAILABLE_MODELS}"
        )

    if model_type == "logistic_regression":
        return {
            "params": LOGISTIC_REGRESSION_PARAMS,
            "training_config": LR_TRAINING_CONFIG,
        }
    elif model_type == "random_forest":
        return {"params": RANDOM_FOREST_PARAMS, "training_config": RF_TRAINING_CONFIG}


def get_model_variant(model_type: str, variant: str = "default") -> Dict[str, Any]:
    """
    Get model configuration variant

    Args:
        model_type: 'random_forest'
        variant: 'default', 'light', or 'heavy'

    Returns:
        Model parameters dictionary
    """
    if model_type == "random_forest":
        variants = {
            "default": RANDOM_FOREST_PARAMS,
            "light": RF_LIGHT_PARAMS,
            "heavy": RF_HEAVY_PARAMS,
        }
        return variants.get(variant, RANDOM_FOREST_PARAMS)

    return LOGISTIC_REGRESSION_PARAMS


if __name__ == "__main__":
    # Test configuration
    print("=" * 60)
    print("MODEL CONFIGURATION TEST")
    print("=" * 60)
    print(f"Available Models: {AVAILABLE_MODELS}")
    print(f"Default Model: {DEFAULT_MODEL}")
    print()

    print("Logistic Regression Config:")
    lr_config = get_model_config("logistic_regression")
    print(f"  Parameters: {len(lr_config['params'])} settings")
    print(
        f"  Expected Training Time: {lr_config['training_config']['expected_training_time']}s"
    )
    print()

    print("Random Forest Config:")
    rf_config = get_model_config("random_forest")
    print(f"  Parameters: {len(rf_config['params'])} settings")
    print(
        f"  Expected Training Time: {rf_config['training_config']['expected_training_time']}s"
    )
    print(f"  Trees: {rf_config['params']['n_estimators']}")
    print(f"  Max Depth: {rf_config['params']['max_depth']}")
    print()

    print("Performance Targets:")
    for metric, target in PERFORMANCE_TARGETS.items():
        print(f"  {metric}: {target}")

    print("=" * 60)
