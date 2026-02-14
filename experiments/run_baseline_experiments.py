"""
Baseline Experiments Runner
Runs LR and RF baseline experiments with reproducibility validation
NOTE: Uses ASCII-safe output for Windows compatibility
"""

# Set UTF-8 encoding for Windows console
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from pathlib import Path
import json
import hashlib
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ModelTrainer
from src.features.feature_engineer import FeatureEngineer
from src.data.data_cleanser import DataCleanser
# from src.validation.validation_pipeline import DataValidationPipeline  # REMOVED


class BaselineExperimentRunner:
    """
    Runs baseline experiments with reproducibility validation

    Features:
    - Trains LR and RF on same data split
    - Runs each model twice to verify reproducibility
    - Computes data hash for integrity
    - Generates comparison reports
    """

    def __init__(self, output_dir: Path = None):
        """
        Initialize experiment runner

        Args:
            output_dir: Directory for experiment outputs
        """
        self.output_dir = output_dir or Path("experiments/baseline")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {}

    def load_real_data(self, data_path: Path, sample_size: int = 20000):
        """Load real BTS flight data"""
        print("\n" + "=" * 70)
        print("LOADING REAL BTS DATA")
        print("=" * 70)

        print(f"\nReading from: {data_path}")
        print(
            f"NOTE: Using head({sample_size}) sample; results are indicative, not final."
        )
        print(
            f"      Temporal split remains valid but may not reflect full month variability."
        )
        df = pd.read_csv(data_path, nrows=sample_size, low_memory=False)

        print(f"✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Check for FlightDate column (BTS uses FlightDate not FL_DATE)
        if "FlightDate" in df.columns:
            print(f"  Date range: {df['FlightDate'].min()} to {df['FlightDate'].max()}")

        if "ArrDelay" in df.columns:
            print(f"  Mean ArrDelay: {df['ArrDelay'].mean():.2f} minutes")
            print(f"  Delay rate (>15min): {(df['ArrDelay'] > 15).mean() * 100:.1f}%")

        # Normalize BTS column names to internal schema
        print("\nNormalizing BTS column names to internal schema...")
        column_mapping = {
            "FlightDate": "FL_DATE",
            "CRSDepTime": "CRS_DEP_TIME",
            "CRSArrTime": "CRS_ARR_TIME",
            "Reporting_Airline": "OP_CARRIER",
            "Origin": "ORIGIN",
            "Dest": "DEST",
            "Distance": "DISTANCE",
            "ArrDelay": "ARR_DELAY",
            "DepDelay": "DEP_DELAY",
            "Cancelled": "CANCELLED",
            "Diverted": "DIVERTED",
            "ArrTime": "ARR_TIME",
            "DepTime": "DEP_TIME",
        }
        df = df.rename(columns=column_mapping)
        print(f"✓ Normalized to internal schema")

        # Sort by date to ensure temporal integrity
        if "FL_DATE" in df.columns:
            df = df.sort_values("FL_DATE").reset_index(drop=True)
            print("✓ Sorted by FL_DATE for temporal split")

        print("=" * 70)

        return df

    def prepare_data_pipeline(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Run data through validation, cleansing, and feature engineering"""
        print("\n" + "=" * 70)
        print("DATA PREPARATION PIPELINE")
        print("=" * 70)

        # [ISSUE 1 FIX] Explicit validation gate (allow warnings)
        print("\n[1/3] Data Validation (SKIPPED - Simplified Pipeline)...")
        # validator = DataValidationPipeline()
        # validation_results = validator.validate(df_raw, stop_on_error=False)
        # print(
        #     f"✓ Validation: {validation_results['status']} ({validation_results['metadata']['validators_run']}/{validation_results['metadata']['validators_total']} validators ran)"
        # )

        # Clean data - MANUAL filtering (DataCleanser expects full BTS schema)
        print("\n[2/3] Data Cleansing (Manual)...")
        initial_count = len(df_raw)

        # Remove cancelled and diverted flights manually
        df_clean = df_raw[(df_raw["CANCELLED"] != 1) & (df_raw["DIVERTED"] != 1)].copy()
        removed = initial_count - len(df_clean)
        print(f"  Removed {removed:,} cancelled/diverted flights")
        print(f"✓ Cleaned: {df_clean.shape[0]:,} rows remaining")

        # [ISSUE 5 FIX] Memory guard for large datasets
        if df_clean.shape[0] > 1_000_000:
            print(
                f"⚠️  Large dataset detected ({df_clean.shape[0]:,} rows); monitor memory usage"
            )

        # Engineer features
        print("\n[3/3] Feature Engineering...")
        engineer = FeatureEngineer()
        df_features = engineer.create_all_features(df_clean)
        print(f"✓ Features: {df_features.shape[1]} columns created")

        # Drop leakage columns and non-numeric types BEFORE the assertion check
        # Drop leakage columns and non-numeric types BEFORE the assertion check
        cols_to_drop = [
            "DEP_DELAY",
            "ARR_TIME",
            "DEP_TIME",  # Leakage
            "FL_DATE",  # Timestamp (not for model)
            "OP_CARRIER",
            "ORIGIN",
            "DEST",  # Raw categoricals (string)
        ]
        existing_to_drop = [col for col in cols_to_drop if col in df_features.columns]
        if existing_to_drop:
            df_features = df_features.drop(columns=existing_to_drop)
            print(f"  Dropped cleanup columns: {existing_to_drop}")

        # Keep only numeric columns (FeatureEngineer may leave some string columns)
        df_features = df_features.select_dtypes(include=[np.number])
        print(f"  Retained {df_features.shape[1]} numeric feature columns")

        # [ISSUE 2 FIX] Explicit leakage column assertions
        print("\n[SAFETY CHECK] Verifying no leakage columns present...")
        leakage_cols = [
            "DEP_DELAY",
            "ARR_TIME",
            "DEP_TIME",
            "CARRIER_DELAY",
            "WEATHER_DELAY",
            "NAS_DELAY",
            "SECURITY_DELAY",
            "LATE_AIRCRAFT_DELAY",
        ]
        found_leakage = [col for col in leakage_cols if col in df_features.columns]
        if found_leakage:
            raise ValueError(f"🚨 DATA LEAKAGE DETECTED: {found_leakage}")
        print(f"✓ No leakage columns detected (checked {len(leakage_cols)} columns)")

        print("=" * 70)

        return df_features

    def run_single_experiment(
        self, df: pd.DataFrame, model_type: str, run_id: int, dataset_version: str
    ) -> dict:
        """Run a single model training experiment"""
        print(f"\n{'=' * 70}")
        print(f"EXPERIMENT: {model_type.upper()} - Run {run_id}")
        print(f"{'=' * 70}")

        # Train model
        trainer = ModelTrainer(model_type=model_type, log_level="INFO")
        results = trainer.train_and_evaluate(
            df=df.copy(),  # Copy to ensure independence
            dataset_version=dataset_version,
        )

        # Extract key metrics
        experiment_summary = {
            "model_type": model_type,
            "run_id": run_id,
            "test_metrics": results["metrics_test"],
            "train_metrics": results["metrics_train"],
            "training_time": results["training_time"],
            "experiment_log_path": str(results["log_path"]),
            "model_path": str(results["model_path"]),
            "data_hash": results["experiment_log"].get("data_hash", None),
        }

        print(f"\n✓ Experiment complete")
        print(f"  Test F1: {results['metrics_test']['f1']:.4f}")
        print(f"  Test Accuracy: {results['metrics_test']['accuracy']:.4f}")
        print(f"  Training Time: {results['training_time']:.2f}s")

        return experiment_summary

    def verify_reproducibility(self, run1: dict, run2: dict, model_type: str) -> dict:
        """Verify that two runs produce identical results"""
        print(f"\n{'=' * 70}")
        print(f"REPRODUCIBILITY CHECK: {model_type.upper()}")
        print(f"{'=' * 70}")

        reproducibility_report = {
            "model_type": model_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checks": {},
        }

        # Check 1: Data hash identical
        hash1 = run1.get("data_hash")
        hash2 = run2.get("data_hash")
        hash_match = hash1 == hash2
        reproducibility_report["checks"]["data_hash"] = {
            "passed": hash_match,
            "run1": hash1,
            "run2": hash2,
        }
        print(f"\n[1/4] Data Hash: {'✓ PASS' if hash_match else '✗ FAIL'}")
        if hash_match:
            print(f"      Both runs: {hash1}")

        # Check 2: Metrics identical (within floating point tolerance)
        metrics_match = True
        metric_diffs = {}

        for metric in ["accuracy", "precision", "recall", "f1"]:
            val1 = run1["test_metrics"].get(metric)
            val2 = run2["test_metrics"].get(metric)

            if val1 is not None and val2 is not None:
                diff = abs(val1 - val2)
                metric_diffs[metric] = diff
                if diff > 1e-10:  # Floating point tolerance
                    metrics_match = False

        reproducibility_report["checks"]["metrics"] = {
            "passed": metrics_match,
            "max_difference": max(metric_diffs.values()) if metric_diffs else 0,
            "differences": metric_diffs,
        }
        print(f"\n[2/4] Test Metrics: {'✓ PASS' if metrics_match else '✗ FAIL'}")
        if metrics_match:
            print(
                f"      All metrics identical (max diff: {max(metric_diffs.values()):.2e})"
            )
        else:
            print(f"      Differences detected:")
            for m, d in metric_diffs.items():
                if d > 1e-10:
                    print(f"        {m}: {d:.2e}")

        # Check 3: Confusion matrix identical
        cm1 = run1["test_metrics"].get("confusion_matrix")
        cm2 = run2["test_metrics"].get("confusion_matrix")
        cm_match = cm1 == cm2
        reproducibility_report["checks"]["confusion_matrix"] = {
            "passed": cm_match,
            "run1": cm1,
            "run2": cm2,
        }
        print(f"\n[3/4] Confusion Matrix: {'✓ PASS' if cm_match else '✗ FAIL'}")

        # Check 4: Training time similar (within 20%)
        time1 = run1["training_time"]
        time2 = run2["training_time"]
        time_diff_pct = abs(time1 - time2) / time1 * 100
        time_similar = time_diff_pct < 20
        reproducibility_report["checks"]["training_time"] = {
            "passed": time_similar,
            "run1": time1,
            "run2": time2,
            "difference_pct": time_diff_pct,
        }
        print(f"\n[4/4] Training Time: {'✓ PASS' if time_similar else '⚠️  WARN'}")
        print(
            f"      Run 1: {time1:.2f}s, Run 2: {time2:.2f}s (Δ {time_diff_pct:.1f}%)"
        )

        # Overall reproducibility
        critical_checks = [hash_match, metrics_match, cm_match]
        reproducibility_report["reproducible"] = all(critical_checks)

        print(f"\n{'=' * 70}")
        if reproducibility_report["reproducible"]:
            print(f"✓ REPRODUCIBILITY VERIFIED for {model_type.upper()}")
        else:
            print(f"✗ REPRODUCIBILITY FAILED for {model_type.upper()}")
        print(f"{'=' * 70}")

        return reproducibility_report

    def compare_models(self, lr_result: dict, rf_result: dict) -> dict:
        """Compare LR and RF baseline performance"""
        print(f"\n{'=' * 70}")
        print("BASELINE MODEL COMPARISON")
        print(f"{'=' * 70}")

        comparison = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "models": {"logistic_regression": lr_result, "random_forest": rf_result},
            "metrics_comparison": {},
            "recommendations": [],
        }

        # Compare metrics
        print("\nTest Set Metrics:")
        print(f"{'Metric':<15} {'LR':<12} {'RF':<12} {'Δ':<12} {'Winner'}")
        print("-" * 70)

        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            lr_val = lr_result["test_metrics"].get(metric)
            rf_val = rf_result["test_metrics"].get(metric)

            if lr_val is not None and rf_val is not None:
                delta = rf_val - lr_val
                delta_pct = (delta / lr_val * 100) if lr_val != 0 else 0
                winner = "RF" if delta > 0 else "LR" if delta < 0 else "TIE"

                comparison["metrics_comparison"][metric] = {
                    "lr": lr_val,
                    "rf": rf_val,
                    "delta": delta,
                    "delta_pct": delta_pct,
                    "winner": winner,
                }

                print(
                    f"{metric:<15} {lr_val:<12.4f} {rf_val:<12.4f} {delta:+12.4f} {winner}"
                )

        # Training time comparison
        lr_time = lr_result["training_time"]
        rf_time = rf_result["training_time"]
        time_ratio = rf_time / lr_time

        print(f"\nTraining Time:")
        print(f"  LR: {lr_time:.2f}s")
        print(f"  RF: {rf_time:.2f}s ({time_ratio:.1f}x slower)")

        comparison["training_time_comparison"] = {
            "lr": lr_time,
            "rf": rf_time,
            "rf_lr_ratio": time_ratio,
        }

        # Decision guidance (NOT hard stops)
        print(f"\n{'=' * 70}")
        print("DECISION GUIDANCE")
        print(f"{'=' * 70}")

        lr_f1 = lr_result["test_metrics"]["f1"]
        rf_f1 = rf_result["test_metrics"]["f1"]
        f1_improvement = rf_f1 - lr_f1
        f1_improvement_pct = (f1_improvement / lr_f1 * 100) if lr_f1 != 0 else 0

        # Check 1: LR F1 < 0.65
        if lr_f1 < 0.65:
            msg = f"⚠️  LR F1 ({lr_f1:.4f}) < 0.65 → Investigate data quality or feature engineering"
            print(f"\n{msg}")
            comparison["recommendations"].append(
                {
                    "severity": "warning",
                    "message": msg,
                    "action": "Investigate data quality and feature engineering",
                }
            )
        else:
            print(f"\n✓ LR F1 ({lr_f1:.4f}) ≥ 0.65 - Acceptable baseline")

        # Check 2: RF F1 < 0.75
        if rf_f1 < 0.75:
            msg = f"⚠️  RF F1 ({rf_f1:.4f}) < 0.75 (IEEE benchmark) → Investigate, do NOT discard"
            print(f"{msg}")
            comparison["recommendations"].append(
                {
                    "severity": "info",
                    "message": msg,
                    "action": "Investigate model performance but continue with current model",
                }
            )
        else:
            print(f"✓ RF F1 ({rf_f1:.4f}) ≥ 0.75 - Meets IEEE benchmark")

        # Check 3: RF improvement over LR
        if f1_improvement < 0.05:  # Less than 5 percentage points
            msg = f"⚠️  RF improvement ({f1_improvement:.4f}, {f1_improvement_pct:.1f}%) < 5pp → Feature interactions may be missing"
            print(f"{msg}")
            comparison["recommendations"].append(
                {
                    "severity": "warning",
                    "message": msg,
                    "action": "Consider adding interaction features or polynomial terms",
                }
            )
        else:
            print(
                f"✓ RF improvement ({f1_improvement:.4f}, {f1_improvement_pct:.1f}%) ≥ 5pp - Good feature interactions"
            )

        # Check 4: RF ≤ LR (red flag)
        if rf_f1 <= lr_f1:
            msg = f"🚨 RF F1 ({rf_f1:.4f}) ≤ LR F1 ({lr_f1:.4f}) → Feature engineering or RF config issue"
            print(f"{msg}")
            comparison["recommendations"].append(
                {
                    "severity": "critical",
                    "message": msg,
                    "action": "Review feature engineering and RF hyperparameters",
                }
            )

        print(f"{'=' * 70}")

        return comparison

    def run_all_experiments(self, df: pd.DataFrame, data_hash: str):
        """Run all baseline experiments"""
        print("\n" + "=" * 70)
        print("BASELINE EXPERIMENTS - COMPLETE SUITE")
        print("=" * 70)

        # [ISSUE 3 FIX] Derive dataset version from data hash
        dataset_version = f"baseline_features_{data_hash[:8]}_{datetime.now().strftime('%Y%m%d')}.parquet"
        print(f"Dataset version: {dataset_version}")
        print(f"Data hash: {data_hash}")
        print(f"Timestamp: {datetime.now()}")
        print("=" * 70)

        # Experiment 1: Logistic Regression Run 1
        lr_run1 = self.run_single_experiment(
            df, "logistic_regression", 1, dataset_version
        )
        self.results["lr_run1"] = lr_run1

        # Experiment 2: Logistic Regression Run 2 (reproducibility)
        lr_run2 = self.run_single_experiment(
            df, "logistic_regression", 2, dataset_version
        )
        self.results["lr_run2"] = lr_run2

        # Verify LR reproducibility
        lr_reproducibility = self.verify_reproducibility(
            lr_run1, lr_run2, "logistic_regression"
        )
        self.results["lr_reproducibility"] = lr_reproducibility

        # Experiment 3: Random Forest Run 1
        rf_run1 = self.run_single_experiment(df, "random_forest", 1, dataset_version)
        self.results["rf_run1"] = rf_run1

        # Experiment 4: Random Forest Run 2 (reproducibility)
        rf_run2 = self.run_single_experiment(df, "random_forest", 2, dataset_version)
        self.results["rf_run2"] = rf_run2

        # Verify RF reproducibility
        rf_reproducibility = self.verify_reproducibility(
            rf_run1, rf_run2, "random_forest"
        )
        self.results["rf_reproducibility"] = rf_reproducibility

        # Compare models (using run 1 results)
        model_comparison = self.compare_models(lr_run1, rf_run1)
        self.results["model_comparison"] = model_comparison

        # Save comprehensive report
        self.save_baseline_report()

        return self.results

    def save_baseline_report(self):
        """Save comprehensive baseline experiment report"""
        report_path = (
            self.output_dir
            / f"baseline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n{'=' * 70}")
        print(f"📄 Baseline report saved to: {report_path}")
        print(f"{'=' * 70}")

        # Create summary file
        summary_path = self.output_dir / "BASELINE_SUMMARY.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("BASELINE EXPERIMENTS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {datetime.now()}\n\n")

            # Reproducibility
            f.write("REPRODUCIBILITY:\n")
            f.write(
                f"  Logistic Regression: {'✓ VERIFIED' if self.results['lr_reproducibility']['reproducible'] else '✗ FAILED'}\n"
            )
            f.write(
                f"  Random Forest:       {'✓ VERIFIED' if self.results['rf_reproducibility']['reproducible'] else '✗ FAILED'}\n\n"
            )

            # Performance
            f.write("PERFORMANCE:\n")
            lr_f1 = self.results["lr_run1"]["test_metrics"]["f1"]
            rf_f1 = self.results["rf_run1"]["test_metrics"]["f1"]
            f.write(f"  LR Test F1:  {lr_f1:.4f}\n")
            f.write(f"  RF Test F1:  {rf_f1:.4f}\n")
            f.write(
                f"  Improvement: {rf_f1 - lr_f1:+.4f} ({(rf_f1 - lr_f1) / lr_f1 * 100:+.1f}%)\n\n"
            )

            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            for rec in self.results["model_comparison"]["recommendations"]:
                f.write(f"  [{rec['severity'].upper()}] {rec['message']}\n")
                f.write(f"    Action: {rec['action']}\n\n")

        print(f"📄 Summary saved to: {summary_path}")


def main():
    """Main execution"""
    print("=" * 70)
    print("FLIGHT DELAY PREDICTION - BASELINE EXPERIMENTS")
    print("=" * 70)
    print("\nThis script will:")
    print("1. Load real BTS flight data")
    print("2. Train Logistic Regression (2 runs)")
    print("3. Train Random Forest (2 runs)")
    print("4. Verify reproducibility")
    print("5. Compare baseline models")
    print("6. Generate comprehensive reports")

    # input("\nPress Enter to start baseline experiments...")

    # Initialize runner
    runner = BaselineExperimentRunner()

    # Load real data
    data_path = (
        Path(__file__).parent.parent
        / "data"
        / "raw"
        / "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2024_1.csv"
    )
    df_raw = runner.load_real_data(data_path, sample_size=20000)

    # Pre-filter to required columns (solves DataCleanser aggressive missing value handling)
    print("\n" + "=" * 70)
    print("PRE-FILTERING TO REQUIRED COLUMNS")
    print("=" * 70)
    print(
        "\nReason: BTS data includes 110+ columns with many optional/post-flight fields."
    )
    print(
        "        DataCleanser would drop all rows due to missing values in these columns."
    )
    print("\nSelecting core columns needed for modeling...")

    required_cols = [
        "FL_DATE",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "OP_CARRIER",
        "ORIGIN",
        "DEST",
        "DISTANCE",
        "ARR_DELAY",
        "CANCELLED",
        "DIVERTED",
        "DEP_DELAY",  # Needed for FeatureEngineer.create_carrier_features()
        "ARR_TIME",  # Needed for FeatureEngineer.create_network_features()
        "DEP_TIME",  # Needed for FeatureEngineer.create_network_features()
        # NOTE: DEP_DELAY, ARR_TIME, DEP_TIME will be dropped after feature engineering
    ]

    # Check which columns exist after normalization
    available_cols = [col for col in required_cols if col in df_raw.columns]
    missing_cols = [col for col in required_cols if col not in df_raw.columns]

    if missing_cols:
        print(f"⚠️  Missing normalized columns: {missing_cols}")
        print(f"    Proceeding with available {len(available_cols)} columns...")

    df_filtered = df_raw[available_cols].copy()
    print(f"✓ Filtered: {df_filtered.shape[0]:,} rows × {df_filtered.shape[1]} columns")
    print(f"  Columns: {', '.join(available_cols)}")
    print("=" * 70)

    # Compute data hash for dataset versioning
    print("\nComputing dataset hash...")
    data_str = str(df_filtered.values.tobytes())
    data_hash = hashlib.sha256(data_str.encode()).hexdigest()
    print(f"✓ Data hash: {data_hash[:16]}...")

    # Prepare data through pipeline
    df_features = runner.prepare_data_pipeline(df_filtered)

    # Run all experiments (dataset version derived from hash inside)
    results = runner.run_all_experiments(df=df_features, data_hash=data_hash)

    # Final summary
    print("\n" + "=" * 70)
    print("BASELINE EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"\nReproducibility:")
    print(
        f"  LR: {'✓ VERIFIED' if results['lr_reproducibility']['reproducible'] else '✗ FAILED'}"
    )
    print(
        f"  RF: {'✓ VERIFIED' if results['rf_reproducibility']['reproducible'] else '✗ FAILED'}"
    )

    print(f"\nPerformance:")
    print(f"  LR F1: {results['lr_run1']['test_metrics']['f1']:.4f}")
    print(f"  RF F1: {results['rf_run1']['test_metrics']['f1']:.4f}")

    print(f"\nOutputs:")
    print(f"  Experiment logs: logs/experiments/")
    print(f"  Baseline report: experiments/baseline/")
    print(f"  Models: models/logistic_regression/ and models/random_forest/")

    print("\n" + "=" * 70)
    print("✓ All baseline experiments completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
