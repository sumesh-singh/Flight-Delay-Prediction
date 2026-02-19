"""
FIXED: Multi-Year Experiment Runner with Class Balancing

CRITICAL FIX: Added class_weight='balanced' to handle imbalanced data
Previous results: Recall=0.0000 (model always predicted on-time)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gc
import warnings
import joblib

warnings.filterwarnings("ignore")

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.multiyear_loader import MultiYearDataLoader
from src.features.feature_engineer import FeatureEngineer
from src.features.target_generator import TargetGenerator
from src.validation.temporal_validation import TemporalValidator


class MemoryOptimizedExperimentRunner:
    """
    MEMORY-OPTIMIZED experiment runner with CLASS BALANCING.
    Processes data in chunks to avoid memory overflow.
    """

    def __init__(self, chunk_size: int = 1_000_000):
        """
        Initialize runner.

        Args:
            chunk_size: Number of rows to process at once
        """
        self.chunk_size = chunk_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"experiments/multiyear_results_{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup Logging
        self.log_dir = Path("experiments/logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"run_{self.timestamp}.log"

        # Redirect stdout/stderr to tee to file
        self.logger = self._setup_logger()

        self.logger.info("=" * 70)
        self.logger.info("MEMORY-OPTIMIZED MULTI-YEAR FLIGHT DELAY PREDICTION")
        self.logger.info("=" * 70)
        self.logger.info(f"Chunk size: {chunk_size:,} rows")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info(f"Logs: {self.log_file}")
        self.logger.info("=" * 70 + "\n")

    def _setup_logger(self):
        import logging

        logger = logging.getLogger("ExperimentRunner")
        logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(message)s"))

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def log(self, msg):
        self.logger.info(msg)

    def normalize_bts_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize BTS column names."""
        column_mapping = {
            "FlightDate": "FL_DATE",
            "Reporting_Airline": "OP_CARRIER",
            "Tail_Number": "TAIL_NUM",
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
        return df

    def simple_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove cancelled/diverted flights."""
        df = df[(df["CANCELLED"] != 1.0) & (df["DIVERTED"] != 1.0)]
        return df

    def process_chunk_to_features(
        self,
        chunk: pd.DataFrame,
        engineer: FeatureEngineer,
        target_gen: TargetGenerator,
        fit: bool = False,
    ) -> tuple:
        """
        Process a single chunk through full pipeline.

        Args:
            chunk: Raw data chunk
            engineer: FeatureEngineer instance
            target_gen: TargetGenerator instance
            fit: Whether to fit encoders (training only)

        Returns:
            (X_chunk, y_chunk) ready for training
        """
        # Normalize
        chunk = self.normalize_bts_schema(chunk)

        # Clean
        chunk = self.simple_clean(chunk)

        if len(chunk) == 0:
            return None, None

        # Create target
        chunk = target_gen.create_target_variables(chunk)

        # Engineer features
        chunk = engineer.create_all_features(chunk, fit_encoders=fit)

        # Select features
        X, y = engineer.select_features_for_training(chunk, target_col="IS_DELAYED")

        # Handle NaN
        X = X.fillna(0)

        return X, y

    def train_with_chunks(self):
        """
        Train model by processing data in chunks.
        Uses SGDClassifier for incremental learning.
        """
        self.log("\n" + "=" * 70)
        self.log("STEP 1: CHUNKED TRAINING WITH CLASS BALANCING")
        self.log("=" * 70)

        # Initialize components
        loader = MultiYearDataLoader()
        # ENABLE EXTERNAL DATA
        engineer = FeatureEngineer(use_external_data=True)
        self.log("External Data Integration: ENABLED")
        target_gen = TargetGenerator()

        # ADDED: StandardScaler (Critical for SGD)
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        # CRITICAL FIX: Use manual sample weights instead of class_weight='balanced'
        # (class_weight='balanced' is not supported with partial_fit in this version)
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
            # class_weight='balanced',  # REMOVED: Causes error with partial_fit
            random_state=42,
            n_jobs=-1,
        )

        feature_names = None
        chunk_num = 0
        total_samples = 0

        # Track class distribution
        total_delayed = 0
        total_ontime = 0

        # Training: 2023-2024
        years = [2023, 2024]

        from sklearn.utils.class_weight import compute_sample_weight

        for year in years:
            self.log(f"\nProcessing year {year}...")
            csv_files = loader.get_available_files(year)

            for csv_file in csv_files:
                self.log(f"\n  Loading {csv_file.name}...")

                # Read in chunks
                for chunk_df in pd.read_csv(
                    csv_file,
                    chunksize=self.chunk_size,
                    usecols=loader.ESSENTIAL_COLS,
                    low_memory=False,
                ):
                    chunk_num += 1
                    self.log(f"    Chunk {chunk_num}: {len(chunk_df):,} rows -> ")

                    # Process chunk
                    X_chunk, y_chunk = self.process_chunk_to_features(
                        chunk_df,
                        engineer,
                        target_gen,
                        fit=(chunk_num == 1),  # Only fit encoders on first chunk
                    )

                    if X_chunk is None or len(X_chunk) == 0:
                        self.log("Skipped (empty after cleaning)")
                        continue

                    # Store feature names from first chunk
                    if feature_names is None:
                        feature_names = X_chunk.columns.tolist()
                    else:
                        # Ensure consistent columns
                        X_chunk = X_chunk.reindex(columns=feature_names, fill_value=0)

                    # Track class distribution
                    total_delayed += y_chunk.sum()
                    total_ontime += (1 - y_chunk).sum()

                    # SCALE FEATURES (Incremental)
                    scaler.partial_fit(X_chunk)
                    X_scaled = scaler.transform(X_chunk)

                    # Compute sample weights for balancing (approximate per chunk)
                    # This replaces class_weight='balanced'
                    weights = compute_sample_weight(class_weight="balanced", y=y_chunk)

                    # Incremental fit with weights
                    model.partial_fit(
                        X_scaled, y_chunk, classes=[0, 1], sample_weight=weights
                    )
                    total_samples += len(X_chunk)

                    delay_rate = y_chunk.mean()
                    delay_rate = y_chunk.mean()
                    self.log(
                        f"    Trained on {len(X_chunk):,} samples (Delay rate: {delay_rate:.1%}, Total: {total_samples:,})"
                    )

                    # Clean up
                    del chunk_df, X_chunk, X_scaled, y_chunk
                    gc.collect()

        self.log(f"\n{'=' * 70}")
        self.log(f"TRAINING COMPLETE")
        self.log(f"Total samples processed: {total_samples:,}")
        self.log(f"Class distribution:")
        self.log(
            f"  On-Time: {total_ontime:,} ({total_ontime / total_samples * 100:.1f}%)"
        )
        self.log(
            f"  Delayed: {total_delayed:,} ({total_delayed / total_samples * 100:.1f}%)"
        )
        self.log(f"Features: {len(feature_names)}")
        self.log(f"{'=' * 70}")

        return model, engineer, target_gen, scaler, feature_names

    def evaluate_on_test_data(
        self,
        model,
        engineer: FeatureEngineer,
        target_gen: TargetGenerator,
        scaler,  # Added scaler
        feature_names: list,
    ):
        """
        Evaluate model on 2025 test data (processed in chunks).
        """
        self.log("\n" + "=" * 70)
        self.log("STEP 2: EVALUATING ON 2025 TEST DATA")
        self.log("=" * 70)

        loader = MultiYearDataLoader()

        all_y_true = []
        all_y_pred = []
        chunk_num = 0

        # Test: 2025 Jan-Nov
        csv_files = loader.get_available_files(2025)[:11]  # Jan-Nov

        for csv_file in csv_files:
            self.log(f"\nEvaluating {csv_file.name}...")

            for chunk_df in pd.read_csv(
                csv_file,
                chunksize=self.chunk_size,
                usecols=loader.ESSENTIAL_COLS,
                low_memory=False,
            ):
                chunk_num += 1

                # Process chunk
                X_chunk, y_chunk = self.process_chunk_to_features(
                    chunk_df,
                    engineer,
                    target_gen,
                    fit=False,  # Don't refit encoders
                )

                if X_chunk is None or len(X_chunk) == 0:
                    continue

                # Ensure consistent columns
                X_chunk = X_chunk.reindex(columns=feature_names, fill_value=0)

                # SCALE FEATURES (Use trained scaler)
                X_scaled = scaler.transform(X_chunk)

                # Predict
                y_pred = model.predict(X_scaled)

                all_y_true.extend(y_chunk.tolist())
                all_y_pred.extend(y_pred.tolist())

                self.log(f"  Chunk {chunk_num}: {len(X_chunk):,} samples evaluated")

                # Clean up
                del chunk_df, X_chunk, X_scaled, y_chunk, y_pred
                gc.collect()

        # Calculate final metrics
        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        self.log(f"\n{'=' * 70}")
        self.log("OVERALL PERFORMANCE (2025 Test Set)")
        self.log(f"{'=' * 70}")
        self.log(f"Samples: {len(y_true):,}")
        self.log(f"Accuracy:  {accuracy:.4f}")
        self.log(f"Precision: {precision:.4f}")
        self.log(f"Recall:    {recall:.4f}")
        self.log(f"F1-Score:  {f1:.4f}")
        self.log(f"\nConfusion Matrix:")
        self.log(f"  True Negatives (On-Time correctly predicted):  {tn:,}")
        self.log(f"  False Positives (On-Time predicted as Delayed): {fp:,}")
        self.log(f"  False Negatives (Delayed predicted as On-Time): {fn:,}")
        self.log(f"  True Positives (Delayed correctly predicted):   {tp:,}")
        self.log(f"\nPrediction Distribution:")
        self.log(
            f"  Predicted On-Time: {(y_pred == 0).sum():,} ({(y_pred == 0).sum() / len(y_pred) * 100:.1f}%)"
        )
        self.log(
            f"  Predicted Delayed: {(y_pred == 1).sum():,} ({(y_pred == 1).sum() / len(y_pred) * 100:.1f}%)"
        )
        self.log(f"{'=' * 70}")

        # Save results
        results = {
            "n_samples": int(len(y_true)),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        }

        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save artifacts
        self.log(f"\nSaving artifacts to {self.output_dir}...")
        joblib.dump(model, self.output_dir / "sgd_model.joblib")
        joblib.dump(scaler, self.output_dir / "scaler.joblib")
        joblib.dump(feature_names, self.output_dir / "feature_names.joblib")
        # Save engineer (for label encoders)
        joblib.dump(engineer, self.output_dir / "feature_engineer.joblib")

        self.log(f"Results saved to {self.output_dir / 'results.json'}")

        return results


def main():
    """
    Main execution function.
    """
    # MEMORY OPTIMIZATION: Process 1M rows at a time
    runner = MemoryOptimizedExperimentRunner(chunk_size=1_000_000)

    # Train model incrementally
    model, engineer, target_gen, scaler, feature_names = runner.train_with_chunks()

    # Evaluate on test data
    results = runner.evaluate_on_test_data(
        model, engineer, target_gen, scaler, feature_names
    )

    runner.log("\n" + "=" * 70)
    runner.log("EXPERIMENT COMPLETE")
    runner.log("=" * 70)
    runner.log(f"Output directory: {runner.output_dir}")
    runner.log(f"Final F1-Score: {results['f1_score']:.4f}")
    runner.log("=" * 70)


if __name__ == "__main__":
    main()
