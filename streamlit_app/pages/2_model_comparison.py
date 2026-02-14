"""
Model Comparison Page - Display Phase 3 Results
"""

import streamlit as st
import json
from pathlib import Path

# Page config
st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")

st.title("📊 Model Comparison")
st.caption("Baseline Experiment Results from Phase 3")

# Load baseline report
baseline_dir = Path("experiments/baseline")
report_files = list(baseline_dir.glob("baseline_report_*.json"))

if not report_files:
    st.error("No baseline reports found. Please run Phase 3 experiments first.")
    st.stop()

latest_report = max(report_files, key=lambda p: p.stat().st_mtime)

with open(latest_report, "r") as f:
    report = json.load(f)

# Extract metrics
lr_metrics = report.get("lr_run1", {}).get("test_metrics", {})
rf_metrics = report.get("rf_run1", {}).get("test_metrics", {})

# Metrics table
st.subheader("Performance Metrics")

metrics_data = {
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [
        f"{lr_metrics.get('accuracy', 0):.4f}",
        f"{rf_metrics.get('accuracy', 0):.4f}",
    ],
    "Precision": [
        f"{lr_metrics.get('precision', 0):.4f}",
        f"{rf_metrics.get('precision', 0):.4f}",
    ],
    "Recall": [
        f"{lr_metrics.get('recall', 0):.4f}",
        f"{rf_metrics.get('recall', 0):.4f}",
    ],
    "F1 Score": [f"{lr_metrics.get('f1', 0):.4f}", f"{rf_metrics.get('f1', 0):.4f}"],
}

st.table(metrics_data)

# Winner
lr_f1 = lr_metrics.get("f1", 0)
rf_f1 = rf_metrics.get("f1", 0)
winner = "Random Forest" if rf_f1 > lr_f1 else "Logistic Regression"
improvement = (
    abs(rf_f1 - lr_f1) / min(lr_f1, rf_f1) * 100 if min(lr_f1, rf_f1) > 0 else 0
)

st.success(
    f"**Best Model**: {winner} (F1: {max(lr_f1, rf_f1):.4f}, +{improvement:.2f}% improvement)"
)

# Reproducibility
st.subheader("Reproducibility Verification")

lr_repro = report.get("lr_reproducibility", {})
rf_repro = report.get("rf_reproducibility", {})

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Logistic Regression",
        "✅ VERIFIED" if lr_repro.get("reproducible") else "❌ FAILED",
        f"F1 diff: {lr_repro.get('checks', {}).get('metrics', {}).get('max_difference', 0):.4f}",
    )

with col2:
    st.metric(
        "Random Forest",
        "✅ VERIFIED" if rf_repro.get("reproducible") else "❌ FAILED",
        f"F1 diff: {rf_repro.get('checks', {}).get('metrics', {}).get('max_difference', 0):.4f}",
    )

# Training info
st.subheader("Training Information")

st.info(f"""
- **Report**: `{latest_report.name}`
- **Training Time (LR)**: {report.get("lr_run1", {}).get("training_time", 0):.2f} seconds
- **Training Time (RF)**: {report.get("rf_run1", {}).get("training_time", 0):.2f} seconds
- **Feature Count**: 48 features (29 engineered + 8 weather + 1 traffic)
""")

# Footer
st.markdown("---")
st.caption("Phase 3: Model Development | Phase 4: Streamlit UI")
