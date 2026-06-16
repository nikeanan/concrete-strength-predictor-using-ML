import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

print("--- Starting Domain Adaptation Experiment ---")

# ==========================================
# 1. LOAD AND PREP THE SOURCE DATA (UCI)
# ==========================================
df_uci = pd.read_csv('concrete.csv')
df_uci['w_c_ratio'] = df_uci['water'] / df_uci['cement']
df_uci['total_binder'] = df_uci['cement'] + df_uci['slag'] + df_uci['ash']
df_uci['w_b_ratio'] = df_uci['water'] / df_uci['total_binder']
df_uci['agg_ratio'] = df_uci['fineagg'] / df_uci['coarseagg']

X_source = df_uci.drop('strength', axis=1)
y_source = df_uci['strength']

# ==========================================
# 2. LOAD AND PREP THE TARGET DATA (Figshare DS3i)
# ==========================================
ds3_mapping = {
    'Fine\nAggregate': 'fineagg', 'Coarse\nAggregate': 'coarseagg',
    'Fly ash': 'ash', 'Cement': 'cement', 'Slag': 'slag',
    'Water': 'water', 'Superplasticizer': 'superplastic',
    'Age': 'age', 'Compressive \nStrength': 'strength'
}

df_target = pd.read_csv('Data Compressive Strength.csv').rename(columns=ds3_mapping)

# FIX: Filter ONLY the base columns first, before calculating the physics
expected_base_columns = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age', 'strength']
df_target = df_target[expected_base_columns].copy()

# Purify the data (turn hidden text into NaNs)
for col in expected_base_columns:
    df_target[col] = pd.to_numeric(df_target[col], errors='coerce')

# NOW we engineer the physical features
df_target['w_c_ratio'] = df_target['water'] / df_target['cement']
df_target['total_binder'] = df_target['cement'] + df_target['slag'] + df_target['ash']
df_target['w_b_ratio'] = df_target['water'] / df_target['total_binder']
df_target['agg_ratio'] = df_target['fineagg'] / df_target['coarseagg']

# Drop the math errors and the purified NaNs
df_target.replace([np.inf, -np.inf], np.nan, inplace=True)
df_target.dropna(inplace=True)

# Lock away 20% for the final exam (Test Set)
X_target_full = df_target.drop('strength', axis=1)
y_target_full = df_target['strength']

X_transfer_pool, X_test, y_transfer_pool, y_test = train_test_split(
    X_target_full, y_target_full, test_size=0.20, random_state=42
)

# ==========================================
# 3. THE AUGMENTATION LOOP
# ==========================================
# We will test injecting 0%, 5%, 10%, 25%, 50%, 75%, and 100% of the Transfer Pool
percentages = [0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
r2_scores = []
mae_scores = []

print("\nRunning Drip-Feed Injections...")

for p in percentages:
    if p == 0:
        # Pure Zero-Shot (Just the UCI data)
        X_train = X_source
        y_train = y_source
    else:
        # Sample p% of the Transfer Pool
        X_sample = X_transfer_pool.sample(frac=p, random_state=42)
        y_sample = y_transfer_pool.loc[X_sample.index]
        
        # Combine the Source Data with the new Target Data
        X_train = pd.concat([X_source, X_sample])
        y_train = pd.concat([y_source, y_sample])
    
    # Train the LightGBM Engine on the combined data
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1))
    ])
    
    model.fit(X_train, y_train)
    
    # Grade it on the locked 20% Test Set
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    r2_scores.append(r2)
    mae_scores.append(mae)
    
    # Calculate how many actual target cylinders the AI saw
    num_samples = 0 if p == 0 else len(X_sample)
    print(f"Injection: {int(p*100):>3}% (Added {num_samples:>3} mixes) -> R²: {r2:.3f} | MAE: {mae:.2f} MPa")

# ==========================================
# 4. PLOT THE LEARNING CURVE
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:blue'
ax1.set_xlabel('Percentage of Target Domain Data Added (%)')
ax1.set_ylabel('R-Squared Accuracy', color=color1)
ax1.plot([p * 100 for p in percentages], r2_scores, color=color1, marker='o', linewidth=2, label='R-Squared')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim([0, 1.0])
ax1.grid(True, linestyle='--', alpha=0.7)

# Create a second y-axis for the MAE
ax2 = ax1.twinx()  
color2 = 'tab:red'
ax2.set_ylabel('Mean Absolute Error (MPa)', color=color2)  
ax2.plot([p * 100 for p in percentages], mae_scores, color=color2, marker='s', linewidth=2, linestyle='dashed', label='MAE')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Domain Adaptation: How Much Local Data is Needed to Calibrate the AI?')
fig.tight_layout()  
plt.savefig('augmentation_curve.png', dpi=300)
print("\n✅ Saved visual learning curve as augmentation_curve.png")