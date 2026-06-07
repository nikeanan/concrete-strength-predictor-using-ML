# 🏗️ Domain Shift Adaptation in Structural Concrete Strength Prediction

**Live Dashboard:** [Click here to use the interactive AI model](https://concrete-strength-predictor-using-ml-duvwxdpdfzu6u38wehc8gz.streamlit.app/)
**Live Dashboard:** [Click here to test the AI model](https://concrete-strength-predictor-using-ml-hqnprbx4wnte4karwqcn2f.streamlit.app/)

### About This Project
This repository contains a localized, end-to-end Machine Learning pipeline built for structural engineering research. It utilizes an XGBoost Regressor to predict the 28-day compressive strength of concrete based on mix design parameters.

**Architecture:**
* **Engine:** (`ightGBM',`xgboost`, `scikit-learn`)
* **Data Processing:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`, `shap`
* **Interface:** Streamlit Community Cloud
* **Performance:** Achieved an R-squared score of >0.92 on the testing dataset.

**📂 Repository Structure**
* **data/raw/: Contains raw inter-institutional datasets.**
* **src/: Core Python modules for data loading, feature engineering, and model training.**

**
figures/: Publication-ready visualizations (KDE plots, feature importance shifts, Augmentation curves).**

**results/: Tabular outputs and benchmark metrics.**

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
