import pandas as pd
import joblib

# 1. Load the frozen model
print("Loading model...")
model = joblib.load('concrete_rf_model.pkl')

# 2. Define a brand new, experimental concrete mix
# Let's test a standard M30-ish mix at 28 days
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

# 3. Ask the AI for the structural strength
predicted_strength = model.predict(new_mix)

print("\n--- Prediction Results ---")
print(f"Predicted Compressive Strength: {predicted_strength[0]:.2f} MPa")