# ==============================
# Streamlit Electrical Load Forecast App
# Batch CSV • Auto Features • Forecast Plot • DateTime Picker
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Electrical Load Forecast", layout="wide")

st.title("⚡ Electrical Load Forecasting App")
st.markdown("Single & batch predictions using a trained **XGBoost time-series model**")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgb_tuned_load_forecast_model_.pkl")

model = load_model()
features = model.feature_names_in_

# -----------------------------
# Helper: Feature Engineering
# -----------------------------
def build_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df.index.weekday >= 5).astype(int)

    df['demand_lag_24hr'] = df['demand'].shift(24)
    df['demand_lag_168hr'] = df['demand'].shift(168)
    df['demand_rolling_mean_24hr'] = df['demand'].rolling(24).mean()
    df['demand_rolling_std_24hr'] = df['demand'].rolling(24).std()

    return df.dropna()

# -----------------------------
# Sidebar Mode Selection
# -----------------------------
mode = st.sidebar.radio("Select Mode", ["Single Prediction", "Batch CSV Prediction"])

# =====================================================
# SINGLE PREDICTION MODE
# =====================================================
if mode == "Single Prediction":
    st.subheader("🔮 Single Forecast")

    forecast_time = st.date_input("Select Date", datetime.now().date())
    forecast_hour = st.slider("Hour", 0, 23, datetime.now().hour)

    st.markdown("### Recent Load Data (last 7 days)")
    recent_csv = st.file_uploader("Upload recent demand CSV (must include datetime, demand)", type="csv")

    if recent_csv:
        recent_df = pd.read_csv(recent_csv, parse_dates=['datetime'])
        recent_df = recent_df.set_index('datetime')

        engineered = build_features(recent_df)
        X_latest = engineered[features].iloc[-1:]

        if st.button("Predict"):
            pred = model.predict(X_latest)[0]
            st.success(f"Predicted Load: **{pred:.2f} MW**")

# =====================================================
# BATCH CSV MODE
# =====================================================
if mode == "Batch CSV Prediction":
    st.subheader("📂 Batch CSV Forecast")

    csv_file = st.file_uploader("Upload CSV with datetime & demand columns", type="csv")

    if csv_file:
        df = pd.read_csv(csv_file, parse_dates=['datetime'])
        df = df.set_index('datetime')

        engineered = build_features(df)
        X = engineered[features]
        engineered['prediction'] = model.predict(X)

        st.success("Batch prediction completed")
        st.dataframe(engineered[['demand', 'prediction']].tail(100))

        # -----------------------------
        # Forecast Plot
        # -----------------------------
        st.subheader("📈 Forecast Plot")
        fig, ax = plt.subplots()
        ax.plot(engineered.index, engineered['demand'], label='Actual')
        ax.plot(engineered.index, engineered['prediction'], label='Predicted')
        ax.legend()
        st.pyplot(fig)

        # Download
        st.download_button(
            "Download Predictions",
            engineered.to_csv().encode('utf-8'),
            "load_predictions.csv",
            "text/csv"
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Streamlit • XGBoost • Time-Series Load Forecasting")