"""
Flight Delay Prediction - Streamlit UI
Phase 4: Inference & Model Comparison
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, date

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit_app.ui_config import APP_TITLE, APP_ICON, APP_SUBTITLE, MODELS
from streamlit_app.components.inference import ModelPredictor
from streamlit_app.utils import (
    validate_flight_date,
    validate_departure_time,
    validate_airports,
    format_probability,
    get_delay_color,
    get_confidence_level,
)

# Page config
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# Title
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption(APP_SUBTITLE)

# Sidebar - Model Selection
st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose Model",
    list(MODELS.keys()),
    index=1,  # Default to Random Forest
)
model_type = MODELS[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Current Model**: {selected_model_name}

Navigate to other pages using the sidebar menu to view model comparisons and data explorer.
""")

# Main content
st.header("Flight Delay Prediction")

# Input form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        flight_date = st.date_input(
            "Flight Date",
            value=date.today(),
            min_value=date(2020, 1, 1),
            max_value=date(2026, 12, 31),
        )

        carrier = st.text_input(
            "Airline Code",
            value="AA",
            max_chars=2,
            help="2-letter airline code (e.g., AA, UA, DL)",
        )

    with col2:
        dep_time_str = st.text_input(
            "Departure Time (HHMM)",
            value="1430",
            max_chars=4,
            help="24-hour format, e.g., 1430 for 2:30 PM",
        )

        origin = st.text_input(
            "Origin Airport", value="JFK", max_chars=3, help="3-letter airport code"
        )

    with col3:
        arr_time_str = st.text_input(
            "Arrival Time (HHMM)", value="1730", max_chars=4, help="24-hour format"
        )

        dest = st.text_input(
            "Destination Airport",
            value="LAX",
            max_chars=3,
            help="3-letter airport code",
        )

    distance = st.number_input(
        "Distance (miles)",
        value=None,
        min_value=0.0,
        max_value=5000.0,
        help="Optional - median will be used if not provided",
    )

    submitted = st.form_submit_button(
        "Predict Delay", type="primary", use_container_width=True
    )

# Process prediction
if submitted:
    # Validate inputs
    is_valid = True

    # Validate date
    date_valid, date_msg = validate_flight_date(flight_date)
    if not date_valid:
        st.error(f"Invalid date: {date_msg}")
        is_valid = False

    # Validate departure time
    dep_time_valid, dep_time_msg = validate_departure_time(dep_time_str)
    if not dep_time_valid:
        st.error(f"Invalid departure time: {dep_time_msg}")
        is_valid = False

    # Validate arrival time
    arr_time_valid, arr_time_msg = validate_departure_time(arr_time_str)
    if not arr_time_valid:
        st.error(f"Invalid arrival time: {arr_time_msg}")
        is_valid = False

    # Validate airports
    airports_valid, airports_msg = validate_airports(origin.upper(), dest.upper())
    if not airports_valid:
        st.error(f"Invalid airports: {airports_msg}")
        is_valid = False

    if is_valid:
        try:
            # Convert times to integers
            dep_time = int(dep_time_str)
            arr_time = int(arr_time_str)

            # Load model and predict
            with st.spinner("Making prediction..."):
                predictor = ModelPredictor(model_type)
                prediction, probability = predictor.predict(
                    flight_date,
                    dep_time,
                    arr_time,
                    carrier.upper(),
                    origin.upper(),
                    dest.upper(),
                    distance,
                )

            # Display results
            st.success("Prediction complete!")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Prediction", prediction, delta=None)

            with col2:
                st.metric(
                    "Confidence",
                    format_probability(probability),
                    delta=get_confidence_level(probability),
                )

            with col3:
                st.metric("Model", selected_model_name)

            # Visualization
            color = get_delay_color(prediction == "Delayed")
            st.markdown(
                f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {color}20; border-left: 5px solid {color}">
                <h3 style="color: {color}; margin: 0;">Flight Status: {prediction}</h3>
                <p style="margin: 10px 0 0 0;">
                    The model predicts this flight will be <strong>{prediction.lower()}</strong> with 
                    <strong>{format_probability(probability)}</strong> confidence.
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        except FileNotFoundError as e:
            st.error(f"""
            Model artifacts not found: {e}
            
            Please run: `python scripts/export_artifacts.py`
            """)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)

# Footer
st.markdown("---")
st.caption("Phase 4: Streamlit UI | Inference-Only (No Training)")
