import streamlit as st
import pandas as pd
import joblib

# 1. Setup the Page Interface
st.set_page_config(page_title="Concrete Predictor", page_icon="🏗️")
st.title("🏗️ Structural Concrete ML Predictor")
st.write("Powered by XGBoost Architecture")

# 2. Load the Model Dictionary
@st.cache_resource
def load_models():
    return joblib.load('models/multi_model_sandbox.pkl')

models = load_models()

# --- NEW: Add a Dropdown to the Sidebar ---
st.sidebar.header("AI Engine Selection")
selected_engine = st.sidebar.selectbox("Choose the ML Algorithm", list(models.keys()))

# ... (Keep your slider code and physics math exactly the same) ...

# 3. Create the Interactive Sidebar Sliders
st.sidebar.header("Mix Design Parameters")
cement = st.sidebar.slider("Cement (kg/m³)", 100.0, 500.0, 320.0)
slag = st.sidebar.slider("Blast Furnace Slag (kg/m³)", 0.0, 300.0, 0.0)
ash = st.sidebar.slider("Fly Ash (kg/m³)", 0.0, 200.0, 0.0)
water = st.sidebar.slider("Water (kg/m³)", 120.0, 250.0, 190.0)
superplastic = st.sidebar.slider("Superplasticizer (kg/m³)", 0.0, 30.0, 0.0)
coarseagg = st.sidebar.slider("Coarse Aggregate (kg/m³)", 800.0, 1200.0, 1050.0)
fineagg = st.sidebar.slider("Fine Aggregate (kg/m³)", 500.0, 1000.0, 800.0)
age = st.sidebar.slider("Curing Age (Days)", 1, 365, 28)

# 4. Format the Data and Engineer the Physics Features for the AI
input_data = pd.DataFrame({
    'cement': [cement],
    'slag': [slag],
    'ash': [ash],
    'water': [water],
    'superplastic': [superplastic],
    'coarseagg': [coarseagg],
    'fineagg': [fineagg],
    'age': [age]
})

# Calculate the new features in the background so the model doesn't crash!
input_data['w_c_ratio'] = input_data['water'] / input_data['cement']
input_data['total_binder'] = input_data['cement'] + input_data['slag'] + input_data['ash']
input_data['w_b_ratio'] = input_data['water'] / input_data['total_binder']
input_data['agg_ratio'] = input_data['fineagg'] / input_data['coarseagg']

# 5. Make the Prediction using the SELECTED engine
st.subheader("Simulation Results")
prediction = models[selected_engine].predict(input_data)[0]

# Display the result
st.metric(label=f"Predicted Strength ({selected_engine})", value=f"{prediction:.2f} MPa")

# Add a quick physics check
wc_ratio = water / cement
st.write(f"*Current Water-to-Cement Ratio:* **{wc_ratio:.2f}**")