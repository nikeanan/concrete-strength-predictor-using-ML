import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# 1. The translator for HPC Figshare (DS3i)
ds3_mapping = {
    'Fine\nAggregate': 'fineagg',
    'Coarse\nAggregate': 'coarseagg',
    'Fly ash': 'ash',
    'Cement': 'cement',
    'Slag': 'slag',
    'Water': 'water',
    'Superplasticizer': 'superplastic',
    'Age': 'age',
    'Compressive \nStrength': 'strength'
}

print("Loading frozen UCI models...")
models = joblib.load('multi_model_sandbox.pkl')

# Extract the Random Forest and its data scaler
rf_pipeline = models['Random Forest (Classic Tree)']
scaler = rf_pipeline.named_steps['scaler']
rf_model = rf_pipeline.named_steps['model']

# 2. Load and purify the DS3i data
df_external = pd.read_csv('Data Compressive Strength.csv').rename(columns=ds3_mapping)
expected_columns = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age', 'strength']
df_clean = df_external[expected_columns].copy()

for col in expected_columns:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

df_clean['w_c_ratio'] = df_clean['water'] / df_clean['cement']
df_clean['total_binder'] = df_clean['cement'] + df_clean['slag'] + df_clean['ash']
df_clean['w_b_ratio'] = df_clean['water'] / df_clean['total_binder']
df_clean['agg_ratio'] = df_clean['fineagg'] / df_clean['coarseagg']

df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
df_clean.dropna(inplace=True)

X_external = df_clean.drop('strength', axis=1)

print("\n--- Running SHAP Analysis on DS3i (HPC Figshare) ---")
# 3. Scale the data (Because the RF was trained on scaled data)
X_scaled = scaler.transform(X_external)

# 4. Extract SHAP values
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_scaled)

# Calculate mean absolute SHAP values for feature importance
mean_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    'Feature': X_external.columns,
    'SHAP Importance (Zero-Shot)': mean_shap
}).sort_values(by='SHAP Importance (Zero-Shot)', ascending=False)

print("\n🔍 External Dataset Feature Importance (How the AI sees DS3i):")
print(shap_df.to_string(index=False))

# 5. Generate the visual plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_external, show=False)
plt.title("SHAP Domain Shift Analysis (DS3i)")
plt.savefig('shap_ds3i.png', bbox_inches='tight', dpi=300)
print("\nSaved visual SHAP plot as shap_ds3i.png")