import streamlit as st
import pandas as pd
import joblib
import numpy as np
import urllib.request
import os
import gc  # Garbage collector to free RAM

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="EV AI Smart Monitor", layout="wide", initial_sidebar_state="expanded")

# --- 2. MODEL DOWNLOADER LOGIC ---
REPO_URL = "https://huggingface.co/mahez/EV-Predictive-Maintenance-AI/resolve/main/"
MODEL_FILES = [
    "soc_model.pkl", "health_model.pkl", "low_battery_model.pkl",
    "range_model.pkl", "abnormal_discharge_model.pkl", "speed_recommendation_model.pkl"
]

def download_models():
    """Downloads missing models. Does NOT load them into RAM yet."""
    for file in MODEL_FILES:
        if not os.path.exists(file):
            with st.spinner(f'Fetching {file} from cloud...'):
                urllib.request.urlretrieve(REPO_URL + file, file)

# --- 3. CUSTOM CSS ---
st.markdown("""
    <style>
    .stButton>button { background-color: #007bff; color: white; border-radius: 10px; font-weight: bold; width: 100%; }
    [data-testid="stSidebar"] [data-testid="stImage"] img {
        width: 150px !important; height: 150px !important;
        border-radius: 50% !important; object-fit: contain !important;
        margin: auto; display: block; border: 2px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚗 EV AI Smart Mission & Health Dashboard")
st.write("Real-time Predictive Analytics for Electric Vehicle Performance.")

# --- 4. SIDEBAR ---
logo_path = "logo.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path)

st.sidebar.header("📍 Trip & Battery Sensors")
target_dist = st.sidebar.number_input("Target Distance (km)", 1, 500, 100)
v_in = st.sidebar.slider("Voltage (V)", 200, 400, 350)
c_in = st.sidebar.slider("Current (A)", -200, 200, 20)
t_in = st.sidebar.slider("Battery Temp (°C)", 10, 75, 35)
cycles = st.sidebar.number_input("Charge Cycles", 0, 2000, 100)
weight = st.sidebar.number_input("Load Weight (kg)", 0, 1000, 500)
speed = st.sidebar.slider("Current Speed (km/h)", 0, 120, 60)

# --- 5. MEMORY-EFFICIENT ANALYSIS ---
st.divider()
if st.button("🚀 EXECUTE FULL SYSTEM ANALYSIS"):
    try:
        download_models() # Ensure files exist on disk
        
        # Step A: Electrical Status
        with st.status("Analyzing Battery Telemetry...") as s:
            m = joblib.load('soc_model.pkl')
            df = pd.DataFrame([[v_in, c_in, t_in]], columns=['Battery_Voltage', 'Battery_Current', 'Battery_Temperature'])
            soc_p = m.predict(df)[0]
            del m # Free RAM immediately
            
            m = joblib.load('low_battery_model.pkl')
            low_p = m.predict(df)[0]
            del m
            s.update(label="Battery Check Complete!", state="complete")

        # Step B: Trip & Range
        with st.status("Calculating Range...") as s:
            m = joblib.load('range_model.pkl')
            df_trip = pd.DataFrame([[soc_p, weight, 25]], columns=['SoC', 'Load_Weight', 'Ambient_Temperature'])
            range_p = m.predict(df_trip)[0]
            del m
            s.update(label="Range Calculation Complete!", state="complete")

        # Step C: Faults & Health
        with st.status("Diagnosing System Health...") as s:
            m = joblib.load('abnormal_discharge_model.pkl')
            df_fault = pd.DataFrame([[abs(c_in), speed, 100, t_in]], columns=['Current_Mag', 'Driving_Speed', 'Motor_Torque', 'Battery_Temperature'])
            fault_p = m.predict(df_fault)[0]
            del m
            
            m = joblib.load('health_model.pkl')
            df_health = pd.DataFrame([[cycles, t_in, 0.8]], columns=['Charge_Cycles', 'Battery_Temperature', 'Component_Health_Score'])
            health_p = m.predict(df_health)[0]
            del m
            s.update(label="Diagnosis Complete!", state="complete")

        gc.collect() # Force final memory cleanup

        # --- 6. DISPLAY RESULTS ---
        st.header("Step 1: Vehicle Telemetry")
        c1, c2, c3 = st.columns(3)
        c1.metric("Current SoC", f"{soc_p*100:.1f} %")
        c2.metric("Available Range", f"{int(range_p)} km")
        c3.metric("State of Health", f"{health_p*100:.1f} %")

        st.divider()
        st.header("Step 2: AI Advisory Reports")
        if target_dist > range_p:
            st.error(f"❌ Mission Impossible: Destination is {target_dist}km, max range is {int(range_p)}km.")
        else:
            st.success(f"✅ Mission Possible: Destination within range.")

        if fault_p == 1:
            st.warning("🚨 ABNORMAL DISCHARGE: Check for motor leaks!")
        
        if health_p < 0.75:
            st.error(f"🔴 URGENT: Health critical ({health_p*100:.1f}%).")
        elif health_p < 0.88:
            st.warning(f"🟡 MAINTENANCE: Aging detected ({health_p*100:.1f}%).")
        else:
            st.success("🟢 SYSTEM HEALTHY.")

    except Exception as e:
        st.error(f"⚠️ System Error: {e}")