import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

def evaluate_domain_shift(new_dataset_path, rename_map=None):
    print(f"\n--- Running Zero-Shot Evaluation on {new_dataset_path} ---")
    
    try:
        df_external = pd.read_csv(new_dataset_path)
    except FileNotFoundError:
        print(f"Error: Could not find {new_dataset_path}. Check the file name!")
        return

    # Apply the specific translation dictionary if one was provided
    if rename_map:
        df_external.rename(columns=rename_map, inplace=True)

    expected_columns = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age', 'strength']
    
    try:
        df_clean = df_external[expected_columns].copy()
    except KeyError as e:
        print(f"Error: Missing expected columns after translation: {e}")
        return

    # --- THE DATA PURIFIER FIX ---
    # Force every column to be a number. If there is text (like "N/A"), turn it to NaN.
    for col in expected_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Engineer the exact same physical features
    df_clean['w_c_ratio'] = df_clean['water'] / df_clean['cement']
    df_clean['total_binder'] = df_clean['cement'] + df_clean['slag'] + df_clean['ash']
    df_clean['w_b_ratio'] = df_clean['water'] / df_clean['total_binder']
    df_clean['agg_ratio'] = df_clean['fineagg'] / df_clean['coarseagg']

    # Filter out math errors (division by zero) AND the text we just turned into NaNs
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean.dropna(inplace=True)

    X_external = df_clean.drop('strength', axis=1)
    y_external = df_clean['strength']

    # Check if we have any data left after dropping NaNs!
    if len(df_clean) == 0:
        print("Error: No valid numeric data left after cleaning!")
        return

    print("Loading frozen UCI models...")
    models = joblib.load('multi_model_sandbox.pkl')

    results = []
    for name, pipeline in models.items():
        # Predict ZERO-SHOT
        predictions = pipeline.predict(X_external)
        
        r2 = r2_score(y_external, predictions)
        mae = mean_absolute_error(y_external, predictions)
        rmse = root_mean_squared_error(y_external, predictions)
        results.append({'Model': name, 'Zero-Shot R-Squared': r2, 'MAE (MPa)': mae, 'RMSE': rmse})

    results_df = pd.DataFrame(results).sort_values(by='Zero-Shot R-Squared', ascending=False)
    print("\n📉 Domain Shift Results (Zero-Shot):")
    print(results_df.to_string(index=False))