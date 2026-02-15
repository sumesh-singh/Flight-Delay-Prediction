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
warnings.filterwarnings('ignore')

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
        
        print("=" * 70)
        print("MEMORY-OPTIMIZED MULTI-YEAR FLIGHT DELAY PREDICTION")
        print("=" * 70)
        print(f"Chunk size: {chunk_size:,} rows")
        print(f"Output: {self.output_dir}")
        print("=" * 70 + "\n")

    def normalize_bts_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize BTS column names."""
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
        fit: bool = False
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
        X, y = engineer.select_features_for_training(chunk, target_col='IS_DELAYED')
        
        # Handle NaN
        X = X.fillna(0)
        
        return X, y

    def train_with_chunks(self):
        """
        Train model by processing data in chunks.
        Uses SGDClassifier for incremental learning.
        """
        print("\n" + "=" * 70)
        print("STEP 1: CHUNKED TRAINING WITH CLASS BALANCING")
        print("=" * 70)
        
        # Initialize components
        loader = MultiYearDataLoader()
        engineer = FeatureEngineer()
        target_gen = TargetGenerator()
        
        # CRITICAL FIX: Added class_weight='balanced'
        model = SGDClassifier(
            loss='log_loss',  # Logistic regression
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            class_weight='balanced',  # FIXES RECALL=0 ISSUE
            random_state=42,
            n_jobs=-1
        )
        
        feature_names = None
        chunk_num = 0
        total_samples = 0
        
        # Track class distribution
        total_delayed = 0
        total_ontime = 0
        
        # Training: 2023-2024
        years = [2023, 2024]
        
        for year in years:
            print(f"\nProcessing year {year}...")
            csv_files = loader.get_available_files(year)
            
            for csv_file in csv_files:
                print(f"\n  Loading {csv_file.name}...")
                
                # Read in chunks
                for chunk_df in pd.read_csv(csv_file, chunksize=self.chunk_size, low_memory=False):
                    chunk_num += 1
                    print(f"    Chunk {chunk_num}: {len(chunk_df):,} rows", end=" -> ")
                    
                    # Process chunk
                    X_chunk, y_chunk = self.process_chunk_to_features(
                        chunk_df, 
                        engineer, 
                        target_gen,
                        fit=(chunk_num == 1)  # Only fit encoders on first chunk
                    )
                    
                    if X_chunk is None or len(X_chunk) == 0:
                        print("Skipped (empty after cleaning)")
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
                    
                    # Incremental fit
                    model.partial_fit(X_chunk, y_chunk, classes=[0, 1])
                    total_samples += len(X_chunk)
                    
                    delay_rate = y_chunk.mean()
                    print(f"Trained on {len(X_chunk):,} samples (Delay rate: {delay_rate:.1%}, Total: {total_samples:,})")
                    
                    # Clean up
                    del chunk_df, X_chunk, y_chunk
                    gc.collect()
        
        print(f"\n{'=' * 70}")
        print(f"TRAINING COMPLETE")
        print(f"Total samples processed: {total_samples:,}")
        print(f"Class distribution:")
        print(f"  On-Time: {total_ontime:,} ({total_ontime/total_samples*100:.1f}%)")
        print(f"  Delayed: {total_delayed:,} ({total_delayed/total_samples*100:.1f}%)")
        print(f"Features: {len(feature_names)}")
        print(f"{'=' * 70}")
        
        return model, engineer, target_gen, feature_names

    def evaluate_on_test_data(
        self, 
        model, 
        engineer: FeatureEngineer,
        target_gen: TargetGenerator,
        feature_names: list
    ):
        """
        Evaluate model on 2025 test data (processed in chunks).
        """
        print("\n" + "=" * 70)
        print("STEP 2: EVALUATING ON 2025 TEST DATA")
        print("=" * 70)
        
        loader = MultiYearDataLoader()
        
        all_y_true = []
        all_y_pred = []
        chunk_num = 0
        
        # Test: 2025 Jan-Nov
        csv_files = loader.get_available_files(2025)[:11]  # Jan-Nov
        
        for csv_file in csv_files:
            print(f"\nEvaluating {csv_file.name}...")
            
            for chunk_df in pd.read_csv(csv_file, chunksize=self.chunk_size, low_memory=False):
                chunk_num += 1
                
                # Process chunk
                X_chunk, y_chunk = self.process_chunk_to_features(
                    chunk_df,
                    engineer,
                    target_gen,
                    fit=False  # Don't refit encoders
                )
                
                if X_chunk is None or len(X_chunk) == 0:
                    continue
                
                # Ensure consistent columns
                X_chunk = X_chunk.reindex(columns=feature_names, fill_value=0)
                
                # Predict
                y_pred = model.predict(X_chunk)
                
                all_y_true.extend(y_chunk.tolist())
                all_y_pred.extend(y_pred.tolist())
                
                print(f"  Chunk {chunk_num}: {len(X_chunk):,} samples evaluated")
                
                # Clean up
                del chunk_df, X_chunk, y_chunk, y_pred
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
        
        print(f"\n{'=' * 70}")
        print("OVERALL PERFORMANCE (2025 Test Set)")
        print(f"{'=' * 70}")
        print(f"Samples: {len(y_true):,}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives (On-Time correctly predicted):  {tn:,}")
        print(f"  False Positives (On-Time predicted as Delayed): {fp:,}")
        print(f"  False Negatives (Delayed predicted as On-Time): {fn:,}")
        print(f"  True Positives (Delayed correctly predicted):   {tp:,}")
        print(f"\nPrediction Distribution:")
        print(f"  Predicted On-Time: {(y_pred == 0).sum():,} ({(y_pred == 0).sum()/len(y_pred)*100:.1f}%)")
        print(f"  Predicted Delayed: {(y_pred == 1).sum():,} ({(y_pred == 1).sum()/len(y_pred)*100:.1f}%)")
        print(f"{'=' * 70}")
        
        # Save results
        results = {
            'n_samples': int(len(y_true)),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
        }
        
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {self.output_dir / 'results.json'}")
        
        return results


def main():
    """
    Main execution function.
    """
    # MEMORY OPTIMIZATION: Process 1M rows at a time
    runner = MemoryOptimizedExperimentRunner(chunk_size=1_000_000)
    
    # Train model incrementally
    model, engineer, target_gen, feature_names = runner.train_with_chunks()
    
    # Evaluate on test data
    results = runner.evaluate_on_test_data(model, engineer, target_gen, feature_names)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Output directory: {runner.output_dir}")
    print(f"Final F1-Score: {results['f1_score']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
