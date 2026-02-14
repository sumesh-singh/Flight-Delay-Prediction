"""
Utility functions for Streamlit UI
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import streamlit as st


def validate_flight_date(date: datetime.date) -> Tuple[bool, str]:
    """Validate flight date."""
    if date.year < 2020:
        return False, "Date must be after 2020"

    far_future = datetime.now().date() + timedelta(days=365)
    if date > far_future:
        return False, "Date cannot be more than 1 year in the future"

    return True, ""


def validate_departure_time(time_str: str) -> Tuple[bool, str]:
    """Validate departure time (HHMM format)."""
    try:
        hour = int(time_str[:2])
        minute = int(time_str[2:])

        if not (0 <= hour < 24):
            return False, "Hour must be between 00-23"
        if not (0 <= minute < 60):
            return False, "Minute must be between 00-59"

        return True, ""
    except:
        return False, "Time must be in HHMM format (e.g., 1430 for 2:30 PM)"


def validate_airports(origin: str, dest: str) -> Tuple[bool, str]:
    """Validate origin and destination airports."""
    if origin == dest:
        return False, "Origin and destination must be different"

    return True, ""


def validate_distance(distance: float) -> Tuple[bool, str]:
    """Validate distance."""
    if distance is None:
        return True, ""  # Optional

    if distance <= 0:
        return False, "Distance must be greater than 0"

    if distance > 5000:
        return False, "Distance seems unrealistic (>5000 miles)"

    return True, ""


def format_probability(prob: float) -> str:
    """Format probability as percentage."""
    return f"{prob * 100:.1f}%"


def get_delay_color(is_delayed: bool) -> str:
    """Get color for delay status."""
    return "#ff4b4b" if is_delayed else "#00cc00"


def get_confidence_level(prob: float) -> str:
    """Get confidence level description."""
    if prob > 0.8:
        return "Very High"
    elif prob > 0.6:
        return "High"
    elif prob > 0.4:
        return "Moderate"
    else:
        return "Low"


@st.cache_data
def load_airport_list() -> List[str]:
    """Load available airports from encoders."""
    # This will be populated from label_encoders
    return []


@st.cache_data
def load_carrier_list() -> List[str]:
    """Load available carriers from encoders."""
    # This will be populated from label_encoders
    return []
