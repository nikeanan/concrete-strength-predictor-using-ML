import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --- 1. SETUP & PAGE CONFIG ---
st.set_page_config(page_title="Domain-Robust Concrete Predictor", page_icon="🏗️", layout="wide")

# Resolve paths dynamically 
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "concrete_rf_model.pkl"

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None

model = load_model()

# --- 3. FRONTEND UI ---
st.title("🏗️ Physics-Informed Concrete Strength Predictor")
st.markdown("""
**Domain Adaptation Research:** This model was trained across multiple institutional datasets (UCI, IIT BBS, Figshare) to resist **domain shift**. 
Unlike standard black-box models, it calculates physical and chemical ratios under the hood before predicting the 28-day compressive strength.
""")

if model is None:
    st.error(f"⚠️ Model not found at `{MODEL_PATH}`. Please ensure the `.pkl` file is in the `models/` folder.")
    st.stop()

st.sidebar.header("🧪 Mix Design Parameters")
st.sidebar.markdown("Enter the raw ingredient quantities in **kg/m³**:")

# Input Sliders
cement = st.sidebar.slider("Cement", min_value=100.0, max_value=550.0, value=300.0, step=1.0)
slag = st.sidebar.slider("Blast Furnace Slag", min_value=0.0, max_value=400.0, value=0.0, step=1.0)
ash = st.sidebar.slider("Fly Ash", min_value=0.0, max_value=300.0, value=0.0, step=1.0)
water = st.sidebar.slider("Water", min_value=120.0, max_value=250.0, value=160.0, step=1.0)
superplastic = st.sidebar.slider("Superplasticizer", min_value=0.0, max_value=35.0, value=5.0, step=0.1)
coarseagg = st.sidebar.slider("Coarse Aggregate", min_value=800.0, max_value=1200.0, value=1000.0, step=1.0)
fineagg = st.sidebar.slider("Fine Aggregate", min_value=500.0, max_value=1000.0, value=750.0, step=1.0)
age = st.sidebar.slider("Age (Days)", min_value=1, max_value=365, value=28, step=1)

# --- 4. BACKEND ENGINEERING ---
# Calculate the physics-informed features under the hood
total_binder = cement + slag + ash
w_c_ratio = water / cement if cement > 0 else 0
w_b_ratio = water / total_binder if total_binder > 0 else 0
agg_ratio = fineagg / coarseagg if coarseagg > 0 else 0

# Create the exact dataframe expected by your new model
input_data = pd.DataFrame({
    'cement': [cement],
    'slag': [slag],
    'ash': [ash],
    'water': [water],
    'superplastic': [superplastic],
    'coarseagg': [coarseagg],
    'fineagg': [fineagg],
    'age': [age],
    'w_c_ratio': [w_c_ratio],
    'total_binder': [total_binder],
    'w_b_ratio': [w_b_ratio],
    'agg_ratio': [agg_ratio]
})

# --- 5. PREDICTION & DISPLAY ---
st.write("### 📊 Live Mix Analysis (Engineered Features)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("W/C Ratio", f"{w_c_ratio:.2f}")
col2.metric("W/B Ratio", f"{w_b_ratio:.2f}")
col3.metric("Total Binder", f"{total_binder:.1f} kg/m³")
col4.metric("F/C Agg Ratio", f"{agg_ratio:.2f}")

st.markdown("---")

if st.button("Predict Compressive Strength", type="primary", use_container_width=True):
    with st.spinner('Analyzing chemical kinetics and physical ratios...'):
        prediction = model.predict(input_data)[0]
    
    st.success("Prediction Complete!")
    st.markdown(f"<h1 style='text-align: center; color: #2e7bcf;'>{prediction:.2f} MPa</h1>", unsafe_allow_html=True)
    
    st.caption("🔍 **Engineering Note:** A standard residential foundation requires ~20-25 MPa, while high-rise commercial structures require 40+ MPa.")