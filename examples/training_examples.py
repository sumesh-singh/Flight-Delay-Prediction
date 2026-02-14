"""
Example: Training Baseline Models
Demonstrates end-to-end model training pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ModelTrainer
from src.features.feature_engineer import FeatureEngineer
from src.features.target_generator import create_target_variable
from src.data.data_cleanser import DataCleanser


def create_sample_dataset():
    """Create realistic sample flight data for demonstration"""
    print("\n" + "=" * 70)
    print("CREATING SAMPLE DATASET")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 10000

    # Create date range
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="10min")

    # Create sample flight data
    df = pd.DataFrame(
        {
            "FL_DATE": dates,
            "CRS_DEP_TIME": np.random.randint(500, 2200, n_samples),
            "CRS_ARR_TIME": np.random.randint(600, 2300, n_samples),
            "OP_CARRIER": np.random.choice(["AA", "DL", "UA", "WN", "B6"], n_samples),
            "ORIGIN": np.random.choice(
                ["ATL", "ORD", "DFW", "LAX", "DEN", "JFK"], n_samples
            ),
            "DEST": np.random.choice(
                ["ATL", "ORD", "DFW", "LAX", "DEN", "JFK"], n_samples
            ),
            "DISTANCE": np.random.randint(200, 2500, n_samples),
            "ARR_DELAY": np.random.normal(10, 25, n_samples),  # Minutes
        }
    )

    print(f"✓ Created dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Date range: {df['FL_DATE'].min().date()} to {df['FL_DATE'].max().date()}")
    print(f"  Mean delay: {df['ARR_DELAY'].mean():.2f} minutes")

    return df


def example_1_logistic_regression():
    """Example 1: Train Logistic Regression baseline"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: LOGISTIC REGRESSION BASELINE")
    print("=" * 70)

    # Create sample data
    df_raw = create_sample_dataset()

    # Clean data
    print("\n📋 Step 1: Data Cleansing")
    cleanser = DataCleanser()
    df_clean = cleanser.full_pipeline(df_raw)
    print(f"✓ Cleaned: {df_clean.shape[0]:,} rows remaining")

    # Engineer features
    print("\n⚙️  Step 2: Feature Engineering")
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_clean)
    print(f"✓ Features: {df_features.shape[1]} columns")

    # Train model
    print("\n🎯 Step 3: Model Training")
    trainer = ModelTrainer(model_type="logistic_regression")
    results = trainer.train_and_evaluate(
        df_features, dataset_version="sample_features_20260210.parquet"
    )

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Test F1-Score: {results['metrics_test']['f1']:.4f}")
    print(f"Test Accuracy: {results['metrics_test']['accuracy']:.4f}")
    print(f"Test Precision: {results['metrics_test']['precision']:.4f}")
    print(f"Test Recall: {results['metrics_test']['recall']:.4f}")
    print(f"Training Time: {results['training_time']:.2f}s")
    print("=" * 70)

    return results


def example_2_random_forest():
    """Example 2: Train Random Forest primary model"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: RANDOM FOREST PRIMARY MODEL")
    print("=" * 70)

    # Create sample data
    df_raw = create_sample_dataset()

    # Clean data
    print("\n📋 Step 1: Data Cleansing")
    cleanser = DataCleanser()
    df_clean = cleanser.full_pipeline(df_raw)

    # Engineer features
    print("\n⚙️  Step 2: Feature Engineering")
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_clean)

    # Train model
    print("\n🎯 Step 3: Model Training")
    trainer = ModelTrainer(model_type="random_forest")
    results = trainer.train_and_evaluate(
        df_features, dataset_version="sample_features_20260210.parquet"
    )

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Test F1-Score: {results['metrics_test']['f1']:.4f}")
    print(f"Test Accuracy: {results['metrics_test']['accuracy']:.4f}")
    print(f"Test Precision: {results['metrics_test']['precision']:.4f}")
    print(f"Test Recall: {results['metrics_test']['recall']:.4f}")

    if results["metrics_test"].get("roc_auc"):
        print(f"Test ROC-AUC: {results['metrics_test']['roc_auc']:.4f}")

    print(f"\nTraining Time: {results['training_time']:.2f}s")

    # Display OOB score if available
    if "oob_score" in results["experiment_log"]["metrics"]:
        print(f"OOB Score: {results['experiment_log']['metrics']['oob_score']:.4f}")

    print("=" * 70)

    return results


def example_3_compare_models():
    """Example 3: Train both models and compare"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: MODEL COMPARISON")
    print("=" * 70)

    # Create sample data (shared for fair comparison)
    df_raw = create_sample_dataset()

    # Prepare data once
    cleanser = DataCleanser()
    df_clean = cleanser.full_pipeline(df_raw)

    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_clean)

    # Train Logistic Regression
    print("\n[1/2] Training Logistic Regression...")
    lr_trainer = ModelTrainer(model_type="logistic_regression")
    lr_results = lr_trainer.train_and_evaluate(
        df_features, dataset_version="comparison_features_20260210.parquet"
    )

    # Train Random Forest
    print("\n[2/2] Training Random Forest...")
    rf_trainer = ModelTrainer(model_type="random_forest")
    rf_results = rf_trainer.train_and_evaluate(
        df_features, dataset_version="comparison_features_20260210.parquet"
    )

    # Compare results
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    from src.models.evaluator import ModelEvaluator

    evaluator = ModelEvaluator()

    comparison = evaluator.compare_models(
        lr_results["metrics_test"],
        rf_results["metrics_test"],
        model1_name="Logistic Regression",
        model2_name="Random Forest",
    )

    print(f"\nWinner: {comparison['winner']}")
    print(f"F1 Improvement: {comparison['improvement']:.2f}%")

    print("\nMetric Comparison:")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        if metric in comparison["deltas"]:
            delta_info = comparison["deltas"][metric]
            print(
                f"  {metric.capitalize():12s}: LR={delta_info['Logistic Regression']:.4f}, RF={delta_info['Random Forest']:.4f}, Δ={delta_info['delta']:+.4f}"
            )

    print("\nTraining Time Comparison:")
    print(f"  Logistic Regression: {lr_results['training_time']:.2f}s")
    print(f"  Random Forest:       {rf_results['training_time']:.2f}s")

    print("=" * 70)

    return {
        "lr_results": lr_results,
        "rf_results": rf_results,
        "comparison": comparison,
    }


def example_4_inference():
    """Example 4: Load trained model and make predictions"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: MODEL INFERENCE")
    print("=" * 70)

    # First, train a model
    print("\nTraining model for inference demo...")
    df_raw = create_sample_dataset()
    cleanser = DataCleanser()
    df_clean = cleanser.full_pipeline(df_raw)
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_clean)

    trainer = ModelTrainer(model_type="random_forest")
    results = trainer.train_and_evaluate(
        df_features, dataset_version="inference_demo_20260210.parquet"
    )

    # Load model for inference
    print("\n🔮 Loading model for inference...")
    from src.models.predictor import ModelPredictor

    predictor = ModelPredictor(results["model_path"])

    # Create new data for prediction
    print("\n📊 Creating new test data...")
    new_data = create_sample_dataset()
    cleanser = DataCleanser()
    new_clean = cleanser.full_pipeline(new_data[:100])  # Just 100 samples
    engineer = FeatureEngineer()
    new_features = engineer.create_all_features(new_clean)

    # Prepare features only (no target)
    from src.features.target_generator import prepare_training_data

    X_new, _ = prepare_training_data(
        new_features, target_type="binary", exclude_arr_delay=True
    )

    # Make predictions
    print(f"\n🎯 Making predictions on {len(X_new)} samples...")
    predictions = predictor.predict(X_new)
    probabilities = predictor.predict_proba(X_new)

    print(f"✓ Predictions generated")
    print(
        f"  Predicted delays: {predictions.sum():,} / {len(predictions):,} ({predictions.mean() * 100:.1f}%)"
    )
    print(f"  Average delay probability: {probabilities[:, 1].mean():.3f}")

    # Custom threshold
    print(f"\n🎚️  Applying custom threshold (0.7)...")
    custom_preds, custom_probs = predictor.predict_with_threshold(X_new, threshold=0.7)
    print(
        f"  High-confidence delays: {custom_preds.sum():,} / {len(custom_preds):,} ({custom_preds.mean() * 100:.1f}%)"
    )

    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("MODEL TRAINING EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates the complete model training pipeline")
    print("with 4 different scenarios:")
    print("\n1. Logistic Regression baseline")
    print("2. Random Forest primary model")
    print("3. Model comparison")
    print("4. Model inference")

    input("\nPress Enter to start examples...")

    try:
        # Run examples
        example_1_logistic_regression()
        input("\nPress Enter to continue to Example 2...")

        example_2_random_forest()
        input("\nPress Enter to continue to Example 3...")

        example_3_compare_models()
        input("\nPress Enter to continue to Example 4...")

        example_4_inference()

        print("\n" + "=" * 70)
        print("✓ ALL EXAMPLES COMPLETE!")
        print("=" * 70)
        print("\nCheck the following directories for outputs:")
        print("  - models/logistic_regression/ - LR models")
        print("  - models/random_forest/ - RF models")
        print("  - logs/experiments/ - Experiment logs")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
