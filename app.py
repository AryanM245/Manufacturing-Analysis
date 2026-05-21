"""Streamlit dashboard for Predictive Maintenance & Manufacturing KPI analytics."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Industry 4.0 Smart Factory", layout="wide", initial_sidebar_state="expanded")


def apply_dark_theme() -> None:
    st.markdown(
        """
        <style>
            .stApp { background-color: #0E1117; color: #FAFAFA; }
            section[data-testid="stSidebar"] { background-color: #161A23; }
            .stMetric { background-color: #1F2937; padding: 0.8rem; border-radius: 0.5rem; }
            .stAlert { border-radius: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = Path("data/ai4i2020_clean.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    return pd.DataFrame()


def build_health_score(tool_wear: float, torque: float, process_temp: float) -> float:
    wear_norm = min(max(tool_wear / 250.0, 0), 1)
    torque_norm = min(max((torque - 3.5) / (80.0 - 3.5), 0), 1)
    temp_norm = min(max((process_temp - 300.0) / (315.0 - 300.0), 0), 1)
    return (0.45 * (1 - wear_norm) + 0.30 * (1 - torque_norm) + 0.25 * (1 - temp_norm)) * 100


def infer_root_cause(torque: float, tool_wear: float, process_temp: float, speed: float) -> str:
    if torque > 60 and tool_wear > 180:
        return "Tool Replacement Needed"
    if process_temp > 312 and speed > 2200:
        return "Likely Cooling or Lubrication Issue"
    if tool_wear > 220:
        return "Severe Tool Degradation"
    if torque > 65:
        return "Overload Condition - Check Mechanical Resistance"
    return "No critical root-cause rule triggered"


def load_artifacts():
    model_path = Path("models/xgb_failure_model.pkl")
    scaler_path = Path("models/feature_scaler.pkl")

    if not model_path.exists() or not scaler_path.exists():
        return None, None, None

    model_bundle = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model_bundle["model"], scaler, model_bundle["feature_names"]


def main() -> None:
    apply_dark_theme()
    st.title("🏭 Predictive Maintenance & Manufacturing KPI Analytics")

    df = load_data()
    model, scaler, feature_names = load_artifacts()

    page = st.sidebar.radio(
        "Navigate",
        [
            "Factory Overview",
            "KPI Analytics",
            "Predictive Maintenance",
            "Alerting & Root Cause",
        ],
    )

    if page == "Factory Overview":
        st.subheader("Factory Overview")
        if df.empty:
            st.warning("No dataset found at data/ai4i2020_clean.csv")
            return

        machine_status = "Operational" if df["machine_failure"].mean() < 0.05 else "At Risk"
        uptime = 100 - (df["machine_failure"].mean() * 100)
        overall_oee = 82.4

        c1, c2, c3 = st.columns(3)
        c1.metric("Machine Status", machine_status)
        c2.metric("Overall OEE", f"{overall_oee:.1f}%")
        c3.metric("Uptime", f"{uptime:.2f}%")

    elif page == "KPI Analytics":
        st.subheader("KPI Analytics")
        if df.empty:
            st.warning("No dataset found at data/ai4i2020_clean.csv")
            return

        if "udi" not in df.columns:
            df["udi"] = np.arange(len(df))

        downtime_df = df.copy()
        downtime_df["downtime_event"] = downtime_df["machine_failure"]
        downtime_df["rolling_downtime"] = downtime_df["downtime_event"].rolling(50, min_periods=1).sum()

        fig_line = px.line(
            downtime_df,
            x="udi",
            y="rolling_downtime",
            title="Downtime Trend (Rolling 50 Records)",
            template="plotly_dark",
        )
        st.plotly_chart(fig_line, use_container_width=True)

        fig_scatter = px.scatter(
            df,
            x="process_temperature_k",
            y="rotational_speed_rpm",
            color="machine_failure",
            title="Process Temperature vs Rotational Speed",
            template="plotly_dark",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    elif page == "Predictive Maintenance":
        st.subheader("Live Failure Prediction")
        if model is None:
            st.error("Model artifacts missing. Train model first with models/train_model.py")
            return

        c1, c2, c3 = st.columns(3)
        air_temp = c1.slider("Air Temperature (K)", 295.0, 315.0, 300.0)
        process_temp = c2.slider("Process Temperature (K)", 300.0, 325.0, 310.0)
        speed = c3.slider("Rotational Speed (RPM)", 1100, 3000, 1500)

        torque = c1.slider("Torque (Nm)", 3.0, 80.0, 40.0)
        tool_wear = c2.slider("Tool Wear (min)", 0, 260, 100)
        udi = c3.number_input("Record Index (UDI)", min_value=1, max_value=20000, value=1)

        input_row = {
            "udi": udi,
            "air_temperature_k": air_temp,
            "process_temperature_k": process_temp,
            "rotational_speed_rpm": speed,
            "torque_nm": torque,
            "tool_wear_min": tool_wear,
            "process_temp_ma_5": process_temp,
            "air_temp_ma_5": air_temp,
            "rot_speed_ma_5": speed,
            "temp_gradient": process_temp - air_temp,
            "temp_gradient_delta": 0.0,
            "wear_norm": min(max(tool_wear / 250.0, 0), 1),
            "torque_norm": min(max((torque - 3.5) / (80.0 - 3.5), 0), 1),
            "proc_temp_norm": min(max((process_temp - 300.0) / (315.0 - 300.0), 0), 1),
            "machine_health_score": build_health_score(tool_wear, torque, process_temp),
        }

        input_df = pd.DataFrame([input_row])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        input_scaled = scaler.transform(input_df)
        failure_proba = float(model.predict_proba(input_scaled)[0, 1])
        health_score = input_row["machine_health_score"]

        st.metric("Failure Probability", f"{failure_proba * 100:.2f}%")
        st.metric("Machine Health Score", f"{health_score:.2f}/100")

    elif page == "Alerting & Root Cause":
        st.subheader("Alerting & Root Cause Analysis")

        torque = st.slider("Torque (Nm)", 3.0, 80.0, 55.0)
        tool_wear = st.slider("Tool Wear (min)", 0, 260, 150)
        process_temp = st.slider("Process Temperature (K)", 300.0, 325.0, 312.0)
        speed = st.slider("Rotational Speed (RPM)", 1100, 3000, 2200)
        estimated_risk = st.slider("Estimated Failure Probability (%)", 0, 100, 45)

        if estimated_risk > 80:
            st.warning("🚨 HIGH FAILURE RISK")
        else:
            st.success("✅ Risk level currently acceptable")

        cause = infer_root_cause(torque, tool_wear, process_temp, speed)
        st.info(f"Likely root cause: **{cause}**")


if __name__ == "__main__":
    main()
