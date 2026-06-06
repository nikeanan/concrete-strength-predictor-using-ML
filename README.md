# 🏗️ Structural Concrete Strength Predictor

**Live Dashboard:** [Click here to use the interactive AI model](https://concrete-strength-predictor-using-ml-duvwxdpdfzu6u38wehc8gz.streamlit.app/)

### About This Project
This repository contains a localized, end-to-end Machine Learning pipeline built for structural engineering research. It utilizes an XGBoost Regressor to predict the 28-day compressive strength of concrete based on mix design parameters.

**Architecture:**
* **Engine:** XGBoost (`xgboost`, `scikit-learn`)
* **Data Processing:** `pandas`, `numpy`
* **Interface:** Streamlit Community Cloud
* **Performance:** Achieved an R-squared score of >0.92 on the testing dataset.

## 🛠️ Developer Commands Cheat Sheet

**Environment Setup**
* `source .venv/bin/activate` - Enter the isolated Python environment.
* `python3 -m venv .venv` - Create a fresh environment.
* `pip install pandas scikit-learn xgboost streamlit joblib matplotlib seaborn` - Install dependencies.

**Execution**
* `python3 concrete.py` - Train the XGBoost model and save the `.pkl` file.
* `python3 predict_mix.py` - Run a fast terminal prediction.
* `streamlit run app.py` - Launch the interactive dashboard locally.

**GitHub Version Control**
* `git add .` - Stage all changed files.
* `git commit -m "Update message"` - Snapshot the code.
* `git push -u origin main` - Send the snapshot to the cloud (updates live website).
