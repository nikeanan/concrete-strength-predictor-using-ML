
# 🧠 Domain Shift Adaptation in Structural Concrete Strength Prediction


****Interactive Research Dashboard:** [Click here to test the inference model](#) *(Replace with URL)* Dashboard:** [Click here to use the interactive AI model](https://concrete-strength-predictor-using-ml-khanftdntwnylgpaeuxaup.streamlit.app/)
**Live Dashboard:** [Click here to test the AI model](https://concrete-strength-predictor-using-ml-hqnprbx4wnte4karwqcn2f.streamlit.app/)

![Python Version](https://img.shields.io/badge/python-3.14-blue)
![Model Performance](https://img.shields.io/badge/R%C2%B2_Score->0.92-brightgreen)
![Research Status](https://img.shields.io/badge/Status-Research_&_Development-purple) 

## 📖 Research Abstract
A critical bottleneck in modern structural engineering is the 28-day waiting period required for physical compressive strength ($f_{ck}$) testing. While historical datasets exist, predictive models often fail when applied to localized, real-world mixes due to **Domain Shift**—the statistical divergence between laboratory training data and field application data. 

This repository contains an end-to-end Machine Learning pipeline that adapts inter-institutional concrete datasets to create a highly accurate, localized prediction engine. By leveraging gradient boosting frameworks and engineering physical features derived from concrete mechanics, this model provides instant, reliable strength inferences for structural digital twins.

## 🔬 Computational Methodology

### Feature Engineering & Materials Science
Raw data is insufficient without domain expertise. This pipeline processes fundamental mix proportions (Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse/Fine Aggregates) and engineers localized physical features, most notably the **Water-to-Binder (w/b) ratio**, to ground the algorithm in established structural mechanics.

### Algorithmic Architecture
* **Primary Engine:** `XGBoost Regressor`
* **Benchmarking:** Evaluated against `LightGBM`, `Random Forest`, and baseline `scikit-learn` architectures.
* **Explainability:** Integrates `SHAP` (SHapley Additive exPlanations) to provide feature importance visualizations, ensuring the model's logic aligns with real-world physical material behavior.

## 📊 Results & Validation
* **Performance Metric:** The finalized XGBoost architecture achieved an $R^2$ score of **>0.92** on testing datasets, demonstrating high fidelity in predicting the 28-day compressive strength.
* **Visual Outputs:** The `figures/` directory contains publication-ready visualizations, including KDE plots detailing the domain shift, feature importance hierarchies, and augmentation curves.

---

## 📂 Repository Structure

```text
├── data/
│   └── raw/              # Inter-institutional concrete datasets (e.g., UCI)
├── src/
│   ├── missing_analysis.py # Comprehensive suite for domain shift and data imputation
│   └── predict_mix.py      # Fast terminal inference script
├── figures/              # KDE plots, SHAP values, and model validation curves
├── models/               # Serialized, deployment-ready model artifacts (.pkl/.joblib)
├── results/              # Tabular performance metrics
└── app.py                # Streamlit interface for model interaction

## 🛠️ Developer Commands Cheat Sheet

**Environment Setup**
* `source .venv/bin/activate` - Enter the isolated Python environment.
* `python3 -m venv .venv` - Create a fresh environment.
* `pip install pandas scikit-learn xgboost streamlit joblib matplotlib seaborn` - Install dependencies.
* `pip install -r requirements.txt` — Install all exact project dependencies.

**Execution**
* `python3 src/missing_analysis.py` - Run the full comprehensive analysis suite and generate all figures.
* `python3 src/predict_mix.py` - Run a fast terminal prediction for a specific concrete mix.
* `streamlit run app.py` - Launch the interactive dashboard locally.

**GitHub Version Control**
* `git add .` - Stage all changed files.
* `git commit -m "Update message"` - Snapshot the code.
* `git push -u origin main` - Send the snapshot to the cloud (updates live website).
