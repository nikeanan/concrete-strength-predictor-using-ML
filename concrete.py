### Reusability: If you get a new dataset from your IIT Hyderabad lab tomorrow called rfa_concrete_mixes.csv, you don't need to rewrite any code. You just type predictor2 = ConcreteStrengthPredictor('rfa_concrete_mixes.csv') and run it.###

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class ConcreteStrengthPredictor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.pipeline = None # We will store the trained AI model here

    def load_dataset(self):
        self.df = pd.read_csv(self.filepath)
        
        # --- FEATURE ENGINEERING (Adding Physics to the AI) ---
        # 1. Water-to-Cement Ratio (Abrams' Law)
        self.df['w_c_ratio'] = self.df['water'] / self.df['cement']
        
        # 2. Total Binder Content
        self.df['total_binder'] = self.df['cement'] + self.df['slag'] + self.df['ash']
        
        # 3. Water-to-Binder Ratio (Crucial for modern mixes with SCMs)
        self.df['w_b_ratio'] = self.df['water'] / self.df['total_binder']

        # 4. Aggregate Packing Density
        self.df['agg_ratio'] = self.df['fineagg'] / self.df['coarseagg']
        print(f"\n--- Dataset '{self.filepath}' Loaded Successfully ---")
        print(f"Dataset Shape: {self.df.shape}")
        print("--- Dataset Loaded & Physics Features Engineered ---")

    def plot_correlation_heatmap(self):
        corr_matrix = self.df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap of Concrete Mix Features")
        plt.savefig('correlation_heatmap.png', dpi=300)
        print("Saved correlation heatmap as image.")

    def train_model(self):
        print("\n--- Training Predictive Model ---")
        X = self.df.drop('strength', axis=1)
        y = self.df['strength']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create the automated pipeline (Upgraded to Tuned LightGBM)
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LGBMRegressor(
                n_estimators=300,        # Increased from 200
                learning_rate=0.05,      # Slowed down from 0.1 for finer learning
                num_leaves=31,           # Allows slightly more complex tree branching
                max_depth=7,             # Slightly deeper trees
                random_state=42,
                verbose=-1               # Hides annoying LightGBM warnings
            ))
        ])

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Model Training Complete!")
        print(f"R-squared Score: {r2:.3f} (Values closer to 1.0 are better)")
        print(f"Mean Absolute Error: {mae:.2f} MPa")

    def get_feature_importance(self):
        print("\n--- Extracting Feature Importance ---")
        rf_model = self.pipeline.named_steps['model']
        features = self.df.drop('strength', axis=1).columns
        importances = rf_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print(importance_df.to_string(index=False))
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, hue='Feature', palette='viridis', legend=False)
        plt.title('What Drives Concrete Strength? (Feature Importance)')
        plt.xlabel('Relative Importance (0 to 1.0)')
        plt.ylabel('Mix Component')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        print("Saved feature importance chart as image.")
        
    def save_model(self, filename='concrete_rf_model.pkl'):
        import joblib
        if self.pipeline is not None:
            joblib.dump(self.pipeline, filename)
            print(f"\n--- Model successfully saved as {filename} ---")
        else:
            print("Error: Train the model first before saving!")

    # ---------------------------------------------------------
    # THE NEW BENCHMARKING METHOD
    # ---------------------------------------------------------
    def run_full_benchmark(self):
        from lightgbm import LGBMRegressor
        from sklearn.linear_model import Ridge
        from sklearn.svm import SVR
        from sklearn.neural_network import MLPRegressor
        
        print("\n--- Running 6-Model Academic Benchmark ---")
        
        # 1. Setup the data for this specific test
        X = self.df.drop('strength', axis=1)
        y = self.df['strength']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

       # 2. Define the 3 Tiers of Models (Synchronized!)
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "XGBoost (Tuned)": XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
            "LightGBM": LGBMRegressor(n_estimators=200, random_state=42, verbose=-1),
            "Ridge Regression": Ridge(alpha=1.0),
            "SVR (RBF Kernel)": SVR(kernel='rbf', C=100, gamma=0.1),
            "MLP Neural Net": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
        }

        results = []

        # 3. Loop through and test each one
        for name, model in models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            results.append({'Model': name, 'R-Squared': r2, 'MAE (MPa)': mae})

        # 4. Display the leaderboard
        results_df = pd.DataFrame(results).sort_values(by='R-Squared', ascending=False)
        print("\n🏆 Benchmark Results (Current Dataset):")
        print(results_df.to_string(index=False))


# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    predictor = ConcreteStrengthPredictor('concrete.csv')
    predictor.load_dataset()
    
    # We run the benchmark first to see the leaderboard
    predictor.run_full_benchmark()
    
    # Then we train your main XGBoost model, save the charts, and save the .pkl file for your website
    predictor.train_model()
    predictor.get_feature_importance() 
    predictor.plot_correlation_heatmap()
    predictor.save_model()