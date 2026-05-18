# Financial-Inclusion
MLPR Course Final
Aim
This code moves beyond the existing framework for evaluating financial inclusivity which heavily emphasizes whether an individual has a bank  account(structural factor).  It goes deeper to interpret the behavioral factors that determine inclusivity. 
The code trains two models -Logistic Regression and XGBoost  to predict whether an adult with a bank or mobile-money account actively makes digital payments. The former is more explainable and works under the assumption of linearity and the latter is free from this assumption and is supplemented by SHAP analysis for explainability. The code explains what drives the prediction using SHAP, Random Forest permutation importance, and a small neural network.
Data
CSV File : the Global Findex 2025 individual-level microdata (~144k rows). The path is set at the top of the script as INDIVIDUAL_CSV.
Optional: a country-level CSV (COUNTRY_CSV), used only if you set USE_COUNTRY_LEVEL_FEATURES = True.
What it predicts
A person is counted as digitally active if they did any one of these in the past year:
•	Paid in-store with a card or phone
•	Paid a bill via phone or computer
•	Bought something online
Sample is limited to account holders.
How to run
1. Install packages:
pip install numpy pandas scikit-learn xgboost shap torch matplotlib
2. Edit the three paths at the top: INDIVIDUAL_CSV, COUNTRY_CSV, OUTPUT_DIR.
3. Run:
python finalcode_v3.py
Expect 30–60 minutes on a normal laptop.
Chronology
•	Loads the data and renames cryptic Findex codes (fin25e2, con1a, …) into readable names.
•	Keeps only account holders and builds the target.
•	Drops columns that would leak the answer (the target's own components, account flags, IDs).
•	Cleans up missing values and drops features that are >60% empty.
•	Drops one variable from each pair that's >85% correlated.
•	Runs leave-one-country-out cross-validation for both models: train on every country except one, test on the one held out, rotate.
•	Refits both models with the full data and saves their feature importances.
•	Runs SHAP -once globally, then once per country to see which drivers are universal vs local.
•	Trains a 4-layer neural network and runs DeepSHAP on it.
•	Saves a side-by-side India vs Zambia SHAP comparison (change the country names in Step 13 if you want a different pair).
Output
Note: Everything saves to OUTPUT_DIR.
The main files:
•	model_summary.csv: headline metrics (ROC-AUC, F1, etc.) for both models.
•	fold_metrics.csv: per-country performance.
•	xgb_feature_importance.csv, logistic_coefficients.csv: what drives the prediction.
•	shap_importance.csv, shap_by_country.csv, shap_stability.csv : SHAP results.
•	nn_train_val_loss.png: neural network loss curve.
•	run_metadata.json : record of the run (row count, features kept, dropped columns).

•	nn_train_val_loss.png: neural network loss curve.
•	run_metadata.json : record of the run (row count, features kept, dropped columns).
