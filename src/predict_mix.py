import pandas as pd
import joblib

# 1. Load the new multi-model dictionary
print("Loading AI engines...")
models = joblib.load('multi_model_sandbox.pkl')

# Select the winning engine (You can swap this for 'XGBoost (Strong Baseline)')
active_engine = models['LightGBM (Highest Accuracy)']

# 2. Define a brand new, experimental concrete mix
new_mix = pd.DataFrame({
    'cement': [320.0],       # kg in a cubic meter
    'slag': [0.0],           
    'ash': [0.0],            
    'water': [190.0],        
    'superplastic': [0.0],   
    'coarseagg': [1050.0],   
    'fineagg': [800.0],      
    'age': [28]              # days curing
})

# --- THE CRITICAL FIX: Calculate the Structural Physics ---
new_mix['w_c_ratio'] = new_mix['water'] / new_mix['cement']
new_mix['total_binder'] = new_mix['cement'] + new_mix['slag'] + new_mix['ash']
new_mix['w_b_ratio'] = new_mix['water'] / new_mix['total_binder']
new_mix['agg_ratio'] = new_mix['fineagg'] / new_mix['coarseagg']

# 3. Ask the AI for the structural strength
predicted_strength = active_engine.predict(new_mix)

print("\n--- Prediction Results ---")
print(f"Engine Used: LightGBM (Highest Accuracy)")
print(f"Predicted Compressive Strength: {predicted_strength[0]:.2f} MPa")