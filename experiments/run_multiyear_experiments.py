"""
Multi-Year Baseline Experiments with Temporal Drift Detection

Strategy 2: Year-Based Split (2023-2024 train, 2025 test)
+ Time-Series Cross-Validation
+ Temporal Drift Detection & Mitigation
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.multiyear_loader import load_train_val_test_split
from src.data.data_cleanser import DataCleanser
from src.features.feature_engineer import FeatureEngineer
from src.features.target_generator import TargetGenerator
from src.validation.drift_detection import analyze_temporal_drift
from src.validation.temporal_validation import TemporalValidator


class MultiYearExperimentRunner:
    """Runs multi-year experiments with drift detection."""

    def __init__(self, sample_size: int = None):
        """
        Initialize experiment runner.

        Args:
            sample_size: If provided, samples from each CSV (for testing)
        """
        self.sample_size = sample_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"experiments/multiyear_results_{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("MULTI-YEAR FLIGHT DELAY PREDICTION")
        print("=" * 70)
        print(f"Strategy: Year-Based Split + Temporal Drift Detection")
        print(f"Output: {self.output_dir}")
        print(f"Sample size: {sample_size if sample_size else 'Full dataset'}")
        print("=" * 70 + "\n")

    def normalize_bts_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize BTS column names to internal schema."""
        column_mapping = {
            "FlightDate": "FL_DATE",
            "Reporting_Airline": "OP_CARRIER",
            "Origin": "ORIGIN",
            "Dest": "DEST",
            "CRSDepTime": "CRS_DEP_TIME",
            "CRSArrTime": "CRS_ARR_TIME",
            "DepTime": "DEP_TIME",
            "ArrTime": "ARR_TIME",
            "DepDelay": "DEP_DELAY",
            "ArrDelay": "ARR_DELAY",
            "Cancelled": "CANCELLED",
            "Diverted": "DIVERTED",
            "Distance": "DISTANCE",
        }

        df = df.rename(columns=column_mapping)
        print(f"[OK] Normalized column names")
        return df

    def simple_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lightweight cleaning for BTS data - only removes cancelled/diverted flights."""
        initial_rows = len(df)

        # Remove cancelled and diverted flights
        if "CANCELLED" in df.columns:
            df = df[df["CANCELLED"] != 1.0]
        if "DIVERTED" in df.columns:
            df = df[df["DIVERTED"] != 1.0]

        removed = initial_rows - len(df)
        print(f"  Removed {removed:,} cancelled/diverted flights")
        print(f"  Remaining: {len(df):,} flights")

        return df

    def load_and_prepare_data(self):
        """Load multi-year data and prepare for training."""
        print("\n" + "=" * 70)
        print("STEP 1: LOADING MULTI-YEAR DATA")
        print("=" * 70)

        # Load data using Strategy 2
        train_df, test_df = load_train_val_test_split(
            sample_size=self.sample_size, verbose=True
        )

        # Normalize schema
        print("\nNormalizing BTS schema...")
        train_df = self.normalize_bts_schema(train_df)
        test_df = self.normalize_bts_schema(test_df)

        # Add month column for temporal analysis
        train_df["month"] = pd.to_datetime(train_df["FL_DATE"]).dt.month
        test_df["month"] = pd.to_datetime(test_df["FL_DATE"]).dt.month

        return train_df, test_df

    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Clean and engineer features."""
        print("\n" + "=" * 70)
        print("STEP 2: DATA PREPROCESSING")
        print("=" * 70)

        # Lightweight cleaning (only remove cancelled/diverted)
        print("\nCleaning training data...")
        train_clean = self.simple_clean(train_df.copy())

        print("Cleaning test data...")
        test_clean = self.simple_clean(test_df.copy())

        # Feature engineering
        print("\nEngineering features (training)...")
        engineer = FeatureEngineer()
        train_features = engineer.create_all_features(train_clean)

        print("Engineering features (test)...")
        test_features = engineer.create_all_features(test_clean, fit_encoders=False)

        # Generate targets
        print("\nGenerating targets...")
        target_gen = TargetGenerator()  # Uses default 15min threshold

        train_targets = target_gen.create_all_targets(train_features)
        test_targets = target_gen.create_all_targets(test_features)

        # Combine features and targets
        print("Combining features and targets...")
        train_with_target = pd.concat([train_features, train_targets], axis=1)
        test_with_target = pd.concat([test_features, test_targets], axis=1)

        print(f"\n[OK] Training: {len(train_with_target):,} flights")
        print(f"[OK] Test: {len(test_with_target):,} flights")

        return train_with_target, test_with_target

    def detect_temporal_drift(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list
    ):
        """Detect distribution shift between training and test data."""
        print("\n" + "=" * 70)
        print("STEP 3: TEMPORAL DRIFT DETECTION")
        print("=" * 70)

        # Run drift analysis
        drift_results = analyze_temporal_drift(
            train_df, test_df, feature_cols, output_dir=self.output_dir
        )

        return drift_results

    def train_and_evaluate_models(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Train models and evaluate on test set."""
        print("\n" + "=" * 70)
        print("STEP 4: MODEL TRAINING & EVALUATION")
        print("=" * 70)

        # Separate features and target
        exclude_cols = [
            "IS_DELAYED",
            "DELAY_CATEGORY",
            "ARR_DELAY",  # Targets
            "FL_DATE",
            "month",
            "year_month",
            "month_id",  # Identifiers/helpers
            "DEP_DELAY",
            "DEP_TIME",
            "ARR_TIME",  # Leakage prevention
            "CARRIER_DELAY",
            "WEATHER_DELAY",
            "NAS_DELAY",  # Post-flight only
            "SECURITY_DELAY",
            "LATE_AIRCRAFT_DELAY",
            "OP_CARRIER",
            "ORIGIN",
            "DEST",  # Use encoded versions
            # LEAKAGE PREVENTION - AGGREGATE FEATURES
            # These were calculated on the full dataset including the target
            "carrier_avg_arr_delay",
            "carrier_arr_delay_std",
            "carrier_delay_rate",
            "carrier_avg_dep_delay",
            "route_avg_delay",
            "route_delay_rate",
            # LEAKAGE PREVENTION - POST-FLIGHT / TARGET PROXIES
            "ArrDel15",
            "ArrivalDelayGroups",
            "CANCELLED",
            "DIVERTED",
            "ActualElapsedTime",
            "AirTime",
            "Flights",
            "TaxiIn",
            "TaxiOut",
            "WheelsOff",
            "WheelsOn",
            "DivAirportLandings",
            # LEAKAGE - SPECIFIC COLUMNS FOUND IN DEBUG
            "DepDel15",
            "DepartureDelayGroups",
            "ArrDelayMinutes",
            "DepDelayMinutes",
            "TotalAddGTime",
            "LongestAddGTime",
            "FirstDepTime",
            "LateAircraftDelay",
            "CarrierDelay",
            "WeatherDelay",
            "NasDelay",
            "SecurityDelay",
        ]

        # Select only numeric columns for training
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns

        # Filter out exclusion list AND any Div/Unnamed columns
        feature_cols = [
            col
            for col in numeric_cols
            if col not in exclude_cols
            and not col.startswith("Div")
            and not col.startswith("Unnamed")
        ]

        X_train = train_df[feature_cols]
        y_train = train_df["IS_DELAYED"]

        X_test = test_df[feature_cols]
        y_test = test_df["IS_DELAYED"]

        print(f"\nFeatures: {len(feature_cols)}")
        print(f"Training samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        print(f"Delay rate (train): {y_train.mean():.2%}")
        print(f"Delay rate (test): {y_test.mean():.2%}")

        # Drop columns with all NaNs in training
        X_train = X_train.dropna(axis=1, how="all")
        feature_cols = X_train.columns.tolist()
        # Align test set
        X_test = X_test[feature_cols]

        print(f"Features after dropping empty columns: {len(feature_cols)}")

        # Impute remaining missing values
        print("Imputing remaining missing values...")
        imputer = SimpleImputer(strategy="mean")

        # Preserve indices
        train_index = X_train.index
        test_index = X_test.index

        X_train = pd.DataFrame(
            imputer.fit_transform(X_train), columns=feature_cols, index=train_index
        )
        X_test = pd.DataFrame(
            imputer.transform(X_test), columns=feature_cols, index=test_index
        )

        # Train models
        models_config = {
            "logistic_regression": {
                "class": LogisticRegression,
                "params": {
                    "max_iter": 1000,
                    "random_state": 42,
                    "class_weight": "balanced",
                },
            },
            "random_forest": {
                "class": RandomForestClassifier,
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42,
                    "n_jobs": -1,
                    "class_weight": "balanced",
                },
            },
        }

        results = {}

        for model_name, config in models_config.items():
            print(f"\n{'=' * 70}")
            print(f"Training: {model_name.upper()}")
            print(f"{'=' * 70}")

            # Train
            model = config["class"](**config["params"])
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            }

            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1']:.4f}")

            results[model_name] = {
                "model": model,
                "metrics": metrics,
                "feature_cols": feature_cols,
            }

        # Create test_df_imputed with original columns + imputed features
        test_df_imputed = test_df.copy()
        test_df_imputed[feature_cols] = X_test

        return results, test_df_imputed

    def temporal_validation(self, test_df: pd.DataFrame, models_results: dict):
        """Run temporal validation on test set."""
        print("\n" + "=" * 70)
        print("STEP 5: TEMPORAL VALIDATION")
        print("=" * 70)

        validation_results = {}

        for model_name, result in models_results.items():
            print(f"\n{model_name.upper()}:")
            print("-" * 70)

            validator = TemporalValidator(result["model"], result["feature_cols"])

            # Month-by-month evaluation
            monthly_df = validator.evaluate_by_month(test_df)

            # Save monthly results
            monthly_path = self.output_dir / f"{model_name}_monthly_performance.csv"
            monthly_df.to_csv(monthly_path, index=False)
            print(f"[OK] Saved monthly results: {monthly_path}")

            validation_results[model_name] = {"monthly": monthly_df.to_dict("records")}

        return validation_results

    def save_final_report(
        self, models_results: dict, temporal_results: dict, drift_results: dict
    ):
        """Save comprehensive experiment report."""
        print("\n" + "=" * 70)
        print("SAVING FINAL REPORT")
        print("=" * 70)

        report = {
            "timestamp": self.timestamp,
            "strategy": "Year-Based Split (2023-2024 train, 2025 test)",
            "sample_size": self.sample_size,
            "models": {},
        }

        for model_name, result in models_results.items():
            report["models"][model_name] = {
                "test_metrics": result["metrics"],
                "n_features": len(result["feature_cols"]),
                "temporal_validation": temporal_results.get(model_name, {}),
            }

        # Add drift summary
        n_features = len(drift_results)
        n_drifted = sum(1 for r in drift_results.values() if r["significant"])

        report["drift_detection"] = {
            "total_features": n_features,
            "drifted_features": n_drifted,
            "drift_rate": n_drifted / n_features if n_features > 0 else 0,
        }

        # Save
        report_path = self.output_dir / "experiment_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[OK] Report saved: {report_path}")

        # Print summary
        print(f"\n{'=' * 70}")
        print("EXPERIMENT SUMMARY")
        print(f"{'=' * 70}")
        print(
            f"Drift: {n_drifted}/{n_features} features ({n_drifted / n_features * 100:.1f}%)"
        )

        for model_name, data in report["models"].items():
            print(f"\n{model_name.upper()}:")
            print(f"  F1 Score: {data['test_metrics']['f1']:.4f}")
            print(f"  Accuracy: {data['test_metrics']['accuracy']:.4f}")

        print(f"\n{'=' * 70}\n")


def main():
    """Main execution function."""
    # Create runner (use sample for testing, None for full dataset)
    runner = MultiYearExperimentRunner(sample_size=1000)  # Change to None for full run

    # Step 1: Load data
    train_df, test_df = runner.load_and_prepare_data()

    # Step 2: Preprocess (FIXED: Now uses correct methods)
    train_processed, test_processed = runner.preprocess_data(train_df, test_df)

    # Get feature columns
    exclude_cols = [
        "IS_DELAYED",
        "DELAY_CATEGORY",
        "ARR_DELAY",
        "FL_DATE",
        "month",
        "year_month",
        "month_id",
        "DEP_DELAY",
        "DEP_TIME",
        "ARR_TIME",
        "CARRIER_DELAY",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
        "OP_CARRIER",
        "ORIGIN",
        "DEST",
    ]
    feature_cols = [col for col in train_processed.columns if col not in exclude_cols]

    # Step 3: Detect drift
    drift_results = runner.detect_temporal_drift(
        train_processed, test_processed, feature_cols
    )

    # Step 4: Train & evaluate
    models_results, test_df_imputed = runner.train_and_evaluate_models(
        train_processed, test_processed
    )

    # Step 5: Temporal validation
    temporal_results = runner.temporal_validation(test_df_imputed, models_results)

    # Step 6: Save report
    runner.save_final_report(models_results, temporal_results, drift_results)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {runner.output_dir}")


if __name__ == "__main__":
    main()
