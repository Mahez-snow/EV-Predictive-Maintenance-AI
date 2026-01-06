import streamlit as st
import pandas as pd
import joblib
import numpy as np
import urllib.request
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="EV AI Smart Monitor", layout="wide", initial_sidebar_state="expanded")

# --- 2. AUTOMATIC MODEL DOWNLOADER ---
REPO_URL = "https://huggingface.co/mahez/EV-Predictive-Maintenance-AI/resolve/main/"

MODEL_FILES = [
    "soc_model.pkl",
    "health_model.pkl",
    "range_model.pkl",
    "abnormal_discharge_model.pkl",
    "speed_recommendation_model.pkl",
    "low_battery_model.pkl"
]
@st.cache_resource
def load_model(name):
    if not os.path.exists(name):
        urllib.request.urlretrieve(REPO_URL + name, name)
    return joblib.load(name)

'''@st.cache_resource(show_spinner=False)
def load_all_models():
    loaded_models = {}
    for file in MODEL_FILES:
        if not os.path.exists(file):
            urllib.request.urlretrieve(REPO_URL + file, file)
        loaded_models[file] = joblib.load(file)
    return loaded_models'''


# --- 3. CUSTOM CSS ---
st.markdown("""
    <style>
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    [data-testid="stSidebar"] [data-testid="stImage"] img {
        width: 150px !important;
        height: 150px !important;
        border-radius: 50% !important;
        background-color: transparent;
        padding: 10px;
        object-fit: contain !important;
        display: block;
        margin-left: auto;
        margin-right: auto;
        border: 2px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚗 EV AI Smart Mission & Health Dashboard")
st.info("✅ App loaded. Click the button to start analysis.")
st.write("Real-time Predictive Analytics for Electric Vehicle Performance and Maintenance.")

# --- 4. SIDEBAR: TRIP & SENSOR INPUTS ---
logo_path = "logo.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=80)
st.sidebar.header("📍 Trip Planning")
target_dist = st.sidebar.number_input("Target Distance (km)", min_value=1, max_value=500, value=100)

st.sidebar.header("🔋 Battery Sensors")
v_in = st.sidebar.slider("Voltage (V)", 200, 400, 350)
c_in = st.sidebar.slider("Current (A)", -200, 200, 20)
t_in = st.sidebar.slider("Battery Temp (°C)", 10, 75, 35)
cycles = st.sidebar.number_input("Charge Cycles", 0, 2000, 100)

st.sidebar.header("⚙️ Vehicle Dynamics")
weight = st.sidebar.number_input("Load Weight (kg)", 0, 1000, 500)
speed = st.sidebar.slider("Current Speed (km/h)", 0, 120, 60)
roughness = st.sidebar.select_slider("Road Condition (Roughness)", options=[0.1, 0.5, 1.0], value=0.1)

# --- 5. SYSTEM ANALYSIS LOGIC ---
st.divider()
if st.button("🚀 EXECUTE FULL SYSTEM ANALYSIS"):
    try:
        # ✅ ONLY CHANGE: loading from Hugging Face instead of local
        models = load_all_models()
        
        m_soc = models['soc_model.pkl']
        m_low = models['low_battery_model.pkl']
        m_range = models['range_model.pkl']
        m_fault = models['abnormal_discharge_model.pkl']
        m_health = models['health_model.pkl']

        # Prepare DataFrames for prediction (UNCHANGED)
        df_elec = pd.DataFrame([[v_in, c_in, t_in]],
                               columns=['Battery_Voltage', 'Battery_Current', 'Battery_Temperature'])
        soc_p = m_soc.predict(df_elec)[0]

        df_trip = pd.DataFrame([[soc_p, weight, 25]],
                               columns=['SoC', 'Load_Weight', 'Ambient_Temperature'])
        range_p = m_range.predict(df_trip)[0]

        df_fault = pd.DataFrame([[abs(c_in), speed, 100, t_in]],
                                columns=['Current_Mag', 'Driving_Speed', 'Motor_Torque', 'Battery_Temperature'])
        fault_p = m_fault.predict(df_fault)[0]

        df_health = pd.DataFrame([[cycles, t_in, 0.8]],
                                 columns=['Charge_Cycles', 'Battery_Temperature', 'Component_Health_Score'])
        health_p = m_health.predict(df_health)[0]

        # --- DISPLAY RESULTS (100% UNCHANGED TEXT) ---
        st.header("Step 1: Vehicle Telemetry")
        r1_col1, r1_col2, r1_col3 = st.columns(3)
        r1_col1.metric("Current SoC (Charge)", f"{soc_p*100:.1f} %")
        r1_col2.metric("Available Range", f"{int(range_p)} km")
        r1_col3.metric("State of Health (SoH)", f"{health_p*100:.1f} %")

        st.divider()
        st.header("Step 2: AI Advisory Reports")

        st.subheader("🏁 Trip & Mission Advisory")
        if target_dist > range_p:
            st.error(f"❌ **Mission Impossible:** Destination is {target_dist}km away, but your max range is {int(range_p)}km.")
        else:
            safety_buffer = range_p / target_dist
            rec_v = 80 if safety_buffer > 1.5 else 60 if safety_buffer > 1.1 else 40
            st.success(f"✅ **Mission Possible:** Maintain a speed of **{rec_v} km/h**.")

        st.subheader("📡 Energy Discharge Analysis")
        if fault_p == 1:
            st.warning("🚨 **ABNORMAL DISCHARGE:** System detects excessive drain. Check for motor leaks!")
        else:
            st.info("🟢 **NORMAL DISCHARGE:** Power consumption is stable.")

        st.subheader("🔧 Maintenance & Service Advisory")
        if health_p < 0.75:
            st.error(f"🔴 **URGENT:** Health critical ({health_p*100:.1f}%). Service mandatory.")
        elif health_p < 0.88:
            st.warning(f"🟡 **MAINTENANCE:** Aging detected ({health_p*100:.1f}%). Schedule check-up.")
        else:
            st.success("🟢 **SYSTEM HEALTHY:** Components in excellent condition.")

    except Exception as e:
        st.error(f"⚠️ System Error: {e}")

# --- 6. FOOTER GUIDE ---
st.divider()
with st.expander("📝 How to interpret the dashboard?"):
    st.write("""
    - **SoC:** Current battery energy.
    - **SoH:** Permanent health of the battery.
    - **Abnormal Discharge:** Detects if energy is being wasted.
    """)
