# ✈️ Flight Delay Prediction System

A production-grade machine learning system to predict flight delays using **35 months of data (2023-2025)** from the Bureau of Transportation Statistics (BTS).

![Streamlit UI](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🌟 Key Features

- **Unified Training Pipeline**: Processes 24 months of training data + 11 months of test data in a single run without memory errors (using chunked processing & subsampling).
- **Novel Human Factors**: Integrates **crew fatigue**, **aircraft utilization**, and **turnaround stress** features (addressing IEEE limitation #3).
- **Leakage-Free**: Rigorous feature selection excludes 60+ potential leakage columns (e.g., `ArrivalDelayGroups`, `Div*` columns).
- **Class Balancing**: Uses SMOTE to handle class imbalance (delayed flights are rare).
- **Interactive UI**: Streamlit app for real-time predictions with confidence scores.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/flight-delay-prediction.git
cd flight-delay-prediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the App (Inference)

If models are already trained (artifacts in `models/`), launch the UI:

```bash
streamlit run streamlit_app/app.py
```

Visit `http://localhost:8501` to use the prediction interface.

### 3. Train Models (Pipeline)

To retrain models from scratch using raw data in `data/raw/`:

```bash
python train_pipeline.py
```
*Approx. runtime: 1 hour (for full 35-month dataset)*

This will:
1. Load & process data chunk-by-month to fit in 8GB RAM.
2. Engineer features (temporal, weather, human factors).
3. Apply SMOTE balancing.
4. Train **Random Forest**, **Logistic Regression**, and **SGD Classifier**.
5. Save best models and artifacts to `models/`.

## 📂 Project Structure

```
├── config/               # Configuration files
├── data/                 # Raw and processed data
├── models/               # Saved model artifacts (joblib, json)
├── src/
│   ├── data/             # Data loaders & API clients
│   ├── features/         # Feature engineering & target generation
│   ├── models/           # Model definitions
│   └── utils/            # Helper functions
├── streamlit_app/        # Streamlit UI application
│   ├── app.py            # Main entry point
│   └── components/       # UI components
├── train_pipeline.py     # Unified training script
├── Dockerfile            # Docker configuration
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## 🐳 Docker Support

Build and run the containerized application:

```bash
# Build image
docker build -t flight-delay-prediction .

# Run container (UI)
docker run -p 8501:8501 flight-delay-prediction
```

## 📊 Performance

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| **Random Forest** | **81.7%** | **0.5358** |
| SGD Classifier | 76.0% | 0.4772 |
| Logistic Regression | 73.8% | 0.4776 |

*(Metrics based on 2025 test dataset)*
