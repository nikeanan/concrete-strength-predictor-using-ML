import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
print("🚀 Starting the Final Comprehensive Analysis Suite...")

# 1. SETUP PATHS
ROOT_DIR = Path(__file__).resolve().parent.parent 
DATA_FOLDER = ROOT_DIR / "data" / "raw"

# 2. DEFINE FUNCTION FIRST (This fixes your NameError)
def load_and_clean(filepath, rename_map=None):
    df = pd.read_csv(filepath)
    if rename_map: df.rename(columns=rename_map, inplace=True)
    expected_base = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']
    if 'strength' in df.columns: expected_base.append('strength')
    df = df[expected_base].copy()
    for col in expected_base: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['w_c_ratio'] = df['water'] / df['cement']
    df['total_binder'] = df['cement'] + df['slag'] + df['ash']
    df['w_b_ratio'] = df['water'] / df['total_binder']
    df['agg_ratio'] = df['fineagg'] / df['coarseagg']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df.drop('strength', axis=1), df['strength'], df

# --- 3. EXECUTE DATA LOADING (Updated) ---
X_uci, y_uci, df_uci = load_and_clean(DATA_FOLDER / 'concrete.csv')
X_ds2, y_ds2, df_ds2 = load_and_clean(DATA_FOLDER / 'Database_IIT_BBS_LAB_Concrete.csv', 
    rename_map={'flyash': 'ash', 'slag (GGBS)': 'slag', 'SP': 'superplastic', 'Coarse Agg.': 'coarseagg', 'SAND (fine agg.)': 'fineagg', 'AGE': 'age', 'Compresive Strength': 'strength'})
X_ds3, y_ds3, df_ds3 = load_and_clean(DATA_FOLDER / 'Data Compressive Strength.csv', 
    rename_map={'Fine\nAggregate': 'fineagg', 'Coarse\nAggregate': 'coarseagg', 'Fly ash': 'ash', 'Cement': 'cement', 'Slag': 'slag', 'Water': 'water', 'Superplasticizer': 'superplastic', 'Age': 'age', 'Compressive \nStrength': 'strength'})

df_uci['Dataset'], df_ds2['Dataset'], df_ds3['Dataset'] = 'UCI (Training)', 'DS2 (IIT BBS)', 'DS3i (Figshare)'
df_all = pd.concat([df_uci, df_ds2, df_ds3])

# --- FIX: Ensure 'models' is a dictionary ---
# If your pickle file is a single model, this dictionary wrapper fixes the loop error
loaded_model = joblib.load(ROOT_DIR / 'models' / 'concrete_rf_model.pkl')
models = {'Random Forest (Classic Tree)': loaded_model}
# PART A: TABLE 2
table2_data = []
for name, pipeline in models.items():
    table2_data.append({
        'Model': name,
        'UCI R²': round(r2_score(y_uci, pipeline.predict(X_uci)), 3),
        'DS2 R²': round(r2_score(y_ds2, pipeline.predict(X_ds2)), 3),
        'DS3i R²': round(r2_score(y_ds3, pipeline.predict(X_ds3)), 3)
    })
df_table2 = pd.DataFrame(table2_data)
df_table2.to_csv(ROOT_DIR / 'results' / 'table2_zero_shot.csv', index=False)
print("✅ Table 2 Saved.")

# PART B: AUGMENTATION
X_tr, X_te, y_tr, y_te = train_test_split(X_ds3, y_ds3, test_size=0.20, random_state=42)
percentages = [0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
sample_counts = [int(len(X_tr) * p) for p in percentages]
means = []
for p in percentages:
    scores = []
    for s in [10, 42, 99]:
        X_t = X_uci if p == 0 else pd.concat([X_uci, X_tr.sample(frac=p, random_state=s)])
        y_t = y_uci if p == 0 else pd.concat([y_uci, y_tr.loc[X_t.index[len(X_uci):]]])
        model = Pipeline([('scaler', StandardScaler()), ('model', LGBMRegressor(verbose=-1))])
        model.fit(X_t, y_t)
        scores.append(r2_score(y_te, model.predict(X_te)))
    means.append(np.mean(scores))
plt.figure(figsize=(10, 6))
plt.plot(sample_counts, means, marker='o'); plt.axvline(x=sample_counts[3], color='red', linestyle='--')
plt.savefig(ROOT_DIR / 'figures' / 'fig_B_augmentation_curve.png', dpi=300)

# PART C: KDE
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, f in enumerate(['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']):
    sns.kdeplot(data=df_all, x=f, hue='Dataset', ax=axes.flatten()[i])
    if f == 'ash': axes.flatten()[i].set_xlim(0, 420)
    if f == 'superplastic': axes.flatten()[i].set_ylim(0, 0.3)
plt.savefig(ROOT_DIR / 'figures' / 'fig_C_kde_all_features.png', dpi=300)

# PART D: IMPORTANCE
rf = models['Random Forest (Classic Tree)']
df_imp = pd.DataFrame({'UCI': permutation_importance(rf, X_uci, y_uci).importances_mean, 
                       'DS2': permutation_importance(rf, X_ds2, y_ds2).importances_mean, 
                       'DS3i': permutation_importance(rf, X_ds3, y_ds3).importances_mean}, index=X_uci.columns)
df_imp.plot(kind='barh', color=['#003f5c', '#bc5090', '#ffa600'])
plt.xlim(0, df_imp.values.max() * 1.15)
plt.savefig(ROOT_DIR / 'figures' / 'fig_D_permutation_importance.png', dpi=300, bbox_inches='tight')

# PART E: BENCHMARK
df_table2['Model'] = df_table2['Model'].str.replace(r'\(.*\)', '', regex=True).str.strip()
df_melted = df_table2.melt(id_vars='Model', value_vars=['UCI R²', 'DS2 R²', 'DS3i R²'], var_name='Domain', value_name='R²')
plt.figure(figsize=(10,6))
sns.barplot(data=df_melted, x='Model', y='R²', hue='Domain', palette=['#003f5c', '#bc5090', '#ffa600'])
plt.savefig(ROOT_DIR / 'figures' / 'fig_E_model_benchmark.png', dpi=300, bbox_inches='tight')

print("🎉 ALL ANALYSIS COMPLETE!")