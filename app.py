import streamlit as st
import pandas as pd
import joblib
import numpy as np
import urllib.request
import os
import gc
import requests
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="EV AI Smart Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CONSTANTS
# -----------------------------
API_URL = "https://ev-predictive-maintenance-ai.onrender.com/latest"
REPO_URL = "https://huggingface.co/mahez/EV-Predictive-Maintenance-AI/resolve/main/"
MODEL_FILES = [
    "soc_model.pkl", "health_model.pkl", "low_battery_model.pkl",
    "range_model.pkl", "abnormal_discharge_model.pkl"
]

# -----------------------------
# MODEL DOWNLOADER
# -----------------------------
def download_models():
    for f in MODEL_FILES:
        if not os.path.exists(f):
            urllib.request.urlretrieve(REPO_URL + f, f)

# -----------------------------
# UI STYLE
# -----------------------------
st.markdown("""
<style>
.metric-card {
    background:#f8fafc;
    padding:15px;
    border-radius:12px;
    border:1px solid #e5e7eb;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("🚗 EV AI Smart Mission & Health Dashboard")
st.caption("Real-time Predictive Maintenance using AI + IoT")

# -----------------------------
# MODE SELECTOR
# -----------------------------
st.sidebar.header("⚡ Data Source")
mode = st.sidebar.radio(
    "Select Input Mode",
    ["Simulation Mode", "Hardware (Live API)"]
)

# -----------------------------
# HARDWARE MODE FETCH
# -----------------------------
if mode == "Hardware (Live API)":
    try:
        r = requests.get(API_URL, timeout=3)
        if r.status_code == 200:
            hw = r.json()
            v_in = hw["voltage"]
            c_in = hw["current"]
            t_in = hw["battery_temp"]
            speed = hw["speed"]
            roughness = 0.5
            st.success("🟢 Live hardware data connected")
        else:
            st.warning("Waiting for hardware...")
            st.stop()
    except:
        st.error("❌ API not reachable")
        st.stop()

# -----------------------------
# SIMULATION MODE INPUTS
# -----------------------------
else:
    st.sidebar.header("🔋 Battery Sensors")
    v_in = st.sidebar.slider("Voltage (V)", 200, 400, 350)
    c_in = st.sidebar.slider("Current (A)", -200, 200, 20)
    t_in = st.sidebar.slider("Battery Temp (°C)", 10, 75, 35)

    st.sidebar.header("⚙️ Vehicle Dynamics")
    speed = st.sidebar.slider("Speed (km/h)", 0, 120, 60)
    roughness = st.sidebar.select_slider(
        "Road Roughness", options=[0.1, 0.5, 1.0], value=0.5
    )

# -----------------------------
# COMMON INPUTS
# -----------------------------
st.sidebar.header("📍 Trip Info")
target_dist = st.sidebar.number_input("Target Distance (km)", 1, 500, 120)
weight = st.sidebar.number_input("Load Weight (kg)", 0, 1000, 450)
cycles = st.sidebar.number_input("Charge Cycles", 0, 2000, 200)

# -----------------------------
# LIVE TELEMETRY
# -----------------------------
st.subheader("📡 Live Telemetry")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Voltage (V)", f"{v_in:.1f}")
c2.metric("Current (A)", f"{c_in:.1f}")
c3.metric("Battery Temp (°C)", f"{t_in:.1f}")
c4.metric("Speed (km/h)", f"{speed}")

st.divider()

# -----------------------------
# EXECUTE AI
# -----------------------------
if st.button("🚀 EXECUTE FULL SYSTEM ANALYSIS"):

    download_models()

    # ---- STEP A: SOC
    m = joblib.load("soc_model.pkl")
    df = pd.DataFrame([[v_in, c_in, t_in]],
        columns=["Battery_Voltage","Battery_Current","Battery_Temperature"])
    soc_p = m.predict(df)[0]
    del m

    # ---- LOW BATTERY
    m = joblib.load("low_battery_model.pkl")
    low_p = m.predict(df)[0]
    del m

    # ---- RANGE
    m = joblib.load("range_model.pkl")
    df_r = pd.DataFrame([[soc_p, weight, 25]],
        columns=["SoC","Load_Weight","Ambient_Temperature"])
    range_p = m.predict(df_r)[0]
    del m

    # ---- DISCHARGE
    m = joblib.load("abnormal_discharge_model.pkl")
    df_d = pd.DataFrame([[abs(c_in), speed, 100, t_in]],
        columns=["Current_Mag","Driving_Speed","Motor_Torque","Battery_Temperature"])
    fault_p = m.predict(df_d)[0]
    del m

    # ---- HEALTH
    m = joblib.load("health_model.pkl")
    df_h = pd.DataFrame([[cycles, t_in, 0.8]],
        columns=["Charge_Cycles","Battery_Temperature","Component_Health_Score"])
    health_p = m.predict(df_h)[0]
    del m

    gc.collect()

    # -----------------------------
    # RESULTS
    # -----------------------------
    st.header("🎛️ AI Results")

    r1, r2, r3 = st.columns(3)
    r1.metric("SoC", f"{soc_p*100:.1f}%")
    r2.metric("Estimated Range", f"{int(range_p)} km")
    r3.metric("State of Health", f"{health_p*100:.1f}%")

    st.divider()

    # -----------------------------
    # AI ADVISORY
    # -----------------------------
    st.subheader("🤖 AI Advisory")

    if target_dist > range_p:
        st.error("❌ Destination unreachable with current battery")
    else:
        st.success("✅ Trip feasible under current conditions")

    if fault_p == 1:
        st.warning("⚠️ Abnormal discharge detected")
    else:
        st.info("🟢 Power consumption normal")

    if health_p < 0.75:
        st.error("🔴 Immediate maintenance required")
    elif health_p < 0.88:
        st.warning("🟡 Preventive maintenance advised")
    else:
        st.success("🟢 System operating optimally")

# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.caption("EV Predictive Maintenance • Hardware + AI • Final Year Ready")
