import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
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
        self.pipeline = None # We will store the active AI model here

    def load_dataset(self):
        self.df = pd.read_csv(self.filepath)
        
        # --- FEATURE ENGINEERING (Adding Physics to the AI) ---
        self.df['w_c_ratio'] = self.df['water'] / self.df['cement']
        self.df['total_binder'] = self.df['cement'] + self.df['slag'] + self.df['ash']
        self.df['w_b_ratio'] = self.df['water'] / self.df['total_binder']
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

    def run_full_benchmark(self):
        from sklearn.linear_model import Ridge
        from sklearn.svm import SVR
        from sklearn.neural_network import MLPRegressor
        
        print("\n--- Running 6-Model Academic Benchmark ---")
        
        X = self.df.drop('strength', axis=1)
        y = self.df['strength']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "XGBoost (Tuned)": XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
            "LightGBM": LGBMRegressor(n_estimators=200, random_state=42, verbose=-1),
            "Ridge Regression": Ridge(alpha=1.0),
            "SVR (RBF Kernel)": SVR(kernel='rbf', C=100, gamma=0.1),
            "MLP Neural Net": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
        }

        results = []

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

        results_df = pd.DataFrame(results).sort_values(by='R-Squared', ascending=False)
        print("\n🏆 Benchmark Results (Current Dataset):")
        print(results_df.to_string(index=False))

    def train_and_save_all_models(self):
        print("\n--- Training Multiple Models for Web Sandbox ---")
        
        # We use the FULL dataset here to give the web models maximum learning data
        X = self.df.drop('strength', axis=1)
        y = self.df['strength']

        # 1. Define the top 3 models
        models_to_train = {
            "LightGBM (Highest Accuracy)": LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=7, random_state=42, verbose=-1),
            "XGBoost (Strong Baseline)": XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
            "Random Forest (Classic Tree)": RandomForestRegressor(n_estimators=200, random_state=42)
        }

        # 2. Train them all
        trained_pipelines = {}
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            pipe.fit(X, y)
            trained_pipelines[name] = pipe

        # 3. Assign the winning model to self.pipeline so the charting tool works
        self.pipeline = trained_pipelines["LightGBM (Highest Accuracy)"]

        # 4. Save the dictionary
        joblib.dump(trained_pipelines, 'multi_model_sandbox.pkl')
        print("\n--- All models successfully packed into multi_model_sandbox.pkl ---")

    def get_feature_importance(self):
        print("\n--- Extracting Feature Importance ---")
        engine_model = self.pipeline.named_steps['model']
        features = self.df.drop('strength', axis=1).columns
        importances = engine_model.feature_importances_
        
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

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    predictor = ConcreteStrengthPredictor('concrete.csv')
    predictor.load_dataset()
    
    predictor.run_full_benchmark()
    
    # Run the NEW multi-model training tool
    predictor.train_and_save_all_models()
    
    # Now the charts will work perfectly again!
    predictor.get_feature_importance() 
    predictor.plot_correlation_heatmap()