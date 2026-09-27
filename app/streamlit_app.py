"""
streamlit_app.py
-----------------
Lightweight Streamlit prototype for the Irrigation Requirement Classification
capstone. Loads the trained model + shared preprocessing pipeline and lets a
user enter a new observation to get a prediction with a probability score.

Run:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import (  # noqa: E402
    ALL_FEATURES,
    CROP_STAGE_OPTIONS,
    CROP_TYPE_OPTIONS,
    SOIL_TYPE_OPTIONS,
    VALID_RANGES,
)

MODEL_PATH = ROOT / "models" / "model.pkl"
MODEL_NAME_PATH = ROOT / "models" / "model_name.txt"

st.set_page_config(page_title="Irrigation Requirement Classifier", page_icon="🌱", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    name = MODEL_NAME_PATH.read_text().strip() if MODEL_NAME_PATH.exists() else "model"
    return model, name


def validate_inputs(values: dict) -> list[str]:
    """Same range checks used to clean the training data (src/preprocessing.py
    VALID_RANGES), applied here to user input before prediction."""
    errors = []
    for col, (lo, hi) in VALID_RANGES.items():
        v = values[col]
        if v is None:
            errors.append(f"'{col}' is required.")
        elif not (lo <= v <= hi):
            errors.append(f"'{col}' should be between {lo} and {hi} (got {v}).")
    return errors


def main():
    st.title("🌱 Irrigation Requirement Classifier")
    st.caption(
        "Educational prototype — predicts whether a field observation is "
        "likely to require irrigation, based on soil moisture, weather and "
        "crop-stage inputs. Not for real-world irrigation decisions."
    )

    model, model_name = load_model()
    if model is None:
        st.error(
            "No trained model found at `models/model.pkl`. "
            "Run `python src/train.py` first, then restart this app."
        )
        st.stop()

    st.sidebar.header("About")
    st.sidebar.write(
        f"**Model in use:** {model_name.replace('_', ' ').title()}\n\n"
        "Trained on a domain-realistic synthetic dataset "
        "(see README for dataset details and how to swap in real data)."
    )

    st.subheader("Enter today's observation")

    col1, col2 = st.columns(2)
    with col1:
        soil_moisture_pct = st.number_input(
            "Soil moisture (%)", min_value=0.0, max_value=60.0, value=25.0, step=0.5,
            help="Volumetric soil moisture reading from a capacitive soil-moisture sensor."
        )
        temperature_c = st.number_input(
            "Temperature (°C)", min_value=-5.0, max_value=50.0, value=28.0, step=0.5
        )
        humidity_pct = st.number_input(
            "Relative humidity (%)", min_value=0.0, max_value=100.0, value=55.0, step=1.0
        )
    with col2:
        rainfall_mm_last_7days = st.number_input(
            "Rainfall in last 7 days (mm)", min_value=0.0, max_value=300.0, value=5.0, step=1.0
        )
        days_since_last_irrigation = st.number_input(
            "Days since last irrigation", min_value=0, max_value=60, value=3, step=1
        )
        crop_stage = st.selectbox("Crop growth stage", CROP_STAGE_OPTIONS, index=1)

    col3, col4 = st.columns(2)
    with col3:
        soil_type = st.selectbox("Soil type", SOIL_TYPE_OPTIONS, index=1)
    with col4:
        crop_type = st.selectbox("Crop type", CROP_TYPE_OPTIONS, index=0)

    if st.button("Predict irrigation requirement", type="primary"):
        values = {
            "soil_moisture_pct": soil_moisture_pct,
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct,
            "rainfall_mm_last_7days": rainfall_mm_last_7days,
            "days_since_last_irrigation": days_since_last_irrigation,
        }
        errors = validate_inputs(values)
        if errors:
            for e in errors:
                st.error(e)
            st.stop()

        row = {
            **values,
            "crop_stage": crop_stage,
            "soil_type": soil_type,
            "crop_type": crop_type,
        }
        X_new = pd.DataFrame([row])[ALL_FEATURES]

        pred = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0, 1] if hasattr(model, "predict_proba") else None

        st.divider()
        if pred == 1:
            st.error("💧 **Irrigation likely required**")
        else:
            st.success("✅ **Irrigation likely NOT required**")

        if proba is not None:
            st.metric("Predicted probability of irrigation need", f"{proba:.1%}")
            st.progress(min(max(proba, 0.0), 1.0))

        with st.expander("Why this matters / how to read this"):
            st.write(
                "This is a single-point estimate from a model trained on "
                "historical patterns. Always cross-check with direct soil "
                "observation and local agronomic guidance before making a "
                "real irrigation decision."
            )


if __name__ == "__main__":
    main()
