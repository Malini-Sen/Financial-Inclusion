# =============================================================================
# FINDEX 2025 — DIGITAL FINANCIAL INCLUSION MODEL
# =============================================================================
# Pipeline overview:
#   1.  Configuration & data load
#   2.  Target definition (active digital payments — Path 2)
#   3.  Leakage column removal
#   4.  Redundancy drops
#   5.  Missing-code recoding & high-missing column removal
#   6.  Optional country-level feature merge
#   7.  High-correlation feature removal (|r| > 0.85)
#   8.  Preprocessors (LR + tree variants)
#   9.  Model definitions (Logistic Regression, XGBoost)
#   10. LOCO cross-validation
#   11. Final fits on full data
#   12. Feature importance exports (LR coefficients, XGB gain, RF permutation)
#   13. Global SHAP
#   14. SHAP stability across LOCO folds
#   15. Country-pair SHAP comparison (India vs Zambia)
#   16. Run metadata export
#   17. MLP benchmark
#   18. Class imbalance diagnostic
#   19. Country fixed-effects logistic regression
#   20. LOCO performance by country
#   21. Country vs individual factor AUC decomposition
#   22. Country heterogeneity analysis
#   23. Fairness audit (equalized odds)
#   24. ALE plots
#   25. SHAP interaction heatmap
#   26. Segmented barrier analysis (inactive users by subgroup)
#   27. Trajectory analysis (2021/22 vs 2025 SHAP shift)
# =============================================================================


# ── IMPORTS ───────────────────────────────────────────────────────────────────

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scipy.stats import pearsonr
import statsmodels.api as sm

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

import shap
from PyALE import ale

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

INDIVIDUAL_CSV = "C:/Users/Malini/Downloads/WLD_2024_FINDEX_v02_M_CSV (4)/findex_microdata_2025_labelled_update112425.csv"
COUNTRY_CSV    = "C:/Users/Malini/Downloads/GlobalFindexDatabase2025 (1).csv"
WAVE_2021_CSV  = r"C:/Users/Malini/Downloads/WLD_2021_FINDEX_v03_M_csv/micro_world_139countries.csv"
OUTPUT_DIR     = Path("mlpr_for_git")

USE_COUNTRY_LEVEL_FEATURES = False  # merge macro indicators onto individual rows
RUN_SHAP                   = True   # SHAP is slow on large datasets
SHAP_SAMPLE_N              = 3000
RANDOM_STATE               = 42

# Column name constants (pre-rename)
TARGET_COL           = "makes_digital_payment"
ACCOUNT_FILTER_COL   = "account"
ACCOUNT_FILTER_VALUE = 1
COUNTRY_COL          = "economy"
COUNTRY_CODE_COL     = "economycode"
WEIGHT_COL           = "wgt"

RENAME_MAP = {
    # Identifiers & design variables
    "economy":         "country_name",
    "economycode":     "country_code",
    "regionwb":        "world_bank_region",
    "pop_adult":       "adult_population_2023",
    "wpid_random":     "gallup_respondent_id",
    "wgt":             "survey_weight",

    # Demographics
    "female":          "gender",            # 1=female, 2=male
    "age":             "age_years",
    "educ":            "education_level",   # 1=primary … 3=tertiary
    "inc_q":           "income_quintile",   # 1 (lowest) … 5 (highest)
    "emp_in":          "in_workforce",      # 1=yes, 2=no
    "urbanicity":      "urban_rural",       # 1=rural, 2=urban

    # Account ownership (constructed)
    "account":         "has_any_account",
    "account_fin":     "has_financial_institution_account",
    "account_mob":     "has_mobile_money_account",
    "dig_account":     "has_digitally_enabled_account",

    # Target & key payment variables (constructed)
    "anydigpayment":   "made_or_received_digital_payment",
    "merchantpay_dig": "made_digital_merchant_payment",
    "pay_utilities":   "utility_payment_method",

    # Receipts (constructed)
    "receive_wages":       "wage_receipt_method",
    "receive_transfers":   "govt_transfer_receipt_method",
    "receive_pensions":    "govt_pension_receipt_method",
    "receive_agriculture": "agri_payment_receipt_method",
    "borrowed":            "borrowed_any",
    "saved":               "saved_any",
    "domestic_remittances":"domestic_remittance_method",

    # Debit / card / account usage
    "fin2":  "has_debit_card",
    "fin3":  "used_card_or_phone_for_account",
    "fin4":  "opened_account_for_wage_or_govt",
    "fin5":  "freq_deposits_into_account",
    "fin6":  "freq_withdrawals_from_account",
    "fin7":  "made_any_deposit_or_withdrawal",
    "fin8":  "stores_money_in_account",
    "fin9a": "receives_balance_alerts_by_phone",
    "fin9b": "checks_balance_by_phone_or_internet",
    "fin10": "has_credit_card",

    # Unbanked card / prior account
    "fin11_0": "has_prepaid_or_stored_value_card",
    "fin11_1": "ever_had_bank_account",
    "fin11_2": "could_use_account_without_help",

    # Barriers to bank account
    "fin11a": "no_account_too_far",
    "fin11b": "no_account_fees_too_high",
    "fin11c": "no_account_lack_documents",
    "fin11d": "no_account_not_enough_money",
    "fin11e": "no_account_family_member_has_one",
    "fin11f": "no_account_distrust",

    # Mobile money account usage
    "fin13_1": "mobile_money_meets_all_needs",
    "fin13a":  "freq_deposits_mobile_money",
    "fin13b":  "freq_sends_from_mobile_money",
    "fin13c":  "freq_withdrawals_mobile_money",
    "fin13d":  "stores_money_in_mobile_money",
    "fin13e":  "received_phishing_request_mobile_money",
    "fin13f":  "sent_money_to_wrong_number",
    "fin13f_1":"recovered_money_sent_to_wrong_number",

    # Barriers to mobile money
    "fin14a": "no_mobile_money_agents_too_far",
    "fin14b": "no_mobile_money_too_expensive",
    "fin14c": "no_mobile_money_lack_documents",
    "fin14d": "no_mobile_money_not_enough_money",
    "fin14e": "no_mobile_money_worried_about_security",

    # Mobile money agent use (unbanked)
    "fin15": "used_agent_for_mobile_money_payment",
    "fin16": "ever_had_mobile_money_account",

    # Saving behaviour
    "fin17a": "saved_at_bank",
    "fin17b": "saved_via_mobile_money",
    "fin17c": "saved_via_savings_club",
    "fin17d": "freq_formal_saving",
    "fin17e": "earned_interest_on_savings",
    "fin17f": "saved_for_old_age",
    "fin18":  "saved_for_any_reason_informal",

    # Insurance & credit
    "fin19": "makes_insurance_payments",
    "fin20": "applied_for_mobile_loan",
    "fin21": "received_mobile_loan",

    # Borrowing sources
    "fin22a":   "borrowed_from_bank",
    "fin22a_1": "borrowed_from_mobile_money_provider",
    "fin22b":   "borrowed_from_friends_or_family",
    "fin22c":   "borrowed_from_savings_club",
    "fin22d":   "borrowed_for_health",
    "fin22e":   "borrowed_for_business",
    "fin22f":   "purchased_food_on_credit",
    "fin22g":   "used_credit_card",
    "fin22h":   "pays_credit_card_in_full",
    "fin23":    "borrowed_for_any_reason_informal",

    # Financial resilience
    "fin24":   "main_source_of_emergency_funds",
    "fin24a":  "difficulty_raising_emergency_funds",
    "fin24b":  "weeks_household_can_cover_if_income_lost",
    "fin24c":  "experienced_natural_disaster",
    "fin24d1": "disaster_caused_income_loss",
    "fin24d2": "disaster_caused_property_damage",
    "fin24d3": "disaster_blocked_account_access",

    # Digital payments (questionnaire)
    "fin25e1": "used_card_or_phone_for_cleaning_supplies",
    "fin25e2": "used_card_or_phone_for_instore_purchase",
    "fin25e3": "freq_digital_instore_payments",
    "fin25e4": "main_reason_cash_only_instore",
    "fin26a":  "used_phone_or_computer_for_bill_payment",
    "fin26b":  "bought_something_online",
    "fin27":   "online_purchase_payment_mode",

    # Remittances
    "fh1":   "sent_domestic_remittance",
    "fin28": "sent_domestic_remittance_digitally",
    "fh2":   "received_domestic_remittance",
    "fin29": "received_domestic_remittance_digitally",
    "fh2a":  "received_international_remittance",

    # Utility payments (questionnaire)
    "fin30":  "makes_utility_payments",
    "fin31a": "utility_paid_via_bank_account",
    "fin31b": "utility_paid_via_mobile_phone",
    "fin31c": "utility_paid_cash_to_bank_agent",
    "fin31d": "utility_paid_exclusively_in_cash",

    # Wage receipts (questionnaire)
    "fin32":  "received_wage_payment",
    "fin33":  "employed_by_government",
    "fin34a": "wage_received_into_bank_account",
    "fin34b": "wage_received_via_mobile_phone",
    "fin34c": "wage_received_in_cash",
    "fin34d": "wage_received_into_card",
    "fin35":  "paid_unexpected_fee_to_withdraw_wage",
    "fin36":  "mode_of_wage_withdrawal",
    "fin36a": "who_withdrew_wage_from_account",

    # Government transfers (questionnaire)
    "fin37":  "received_govt_financial_support",
    "fin38":  "received_govt_pension",
    "fin39a": "govt_money_received_into_bank",
    "fin39b": "govt_money_received_via_mobile",
    "fin39c": "govt_money_received_in_cash_only",
    "fin39d": "govt_money_received_into_card",
    "fin40":  "mode_of_govt_money_withdrawal",
    "fin41":  "paid_unexpected_fee_to_withdraw_govt_money",
    "fin41a": "who_withdrew_govt_money_from_account",

    # Agricultural payments (questionnaire)
    "fin42":  "received_agricultural_payment",
    "fin43a": "agri_payment_into_bank_account",
    "fin43b": "agri_payment_via_mobile_phone",
    "fin43c": "agri_payment_in_cash_only",
    "fin43d": "agri_payment_into_card",
    "fin44":  "uses_phone_for_farming_info",

    # Financial worry
    "fin45": "greatest_financial_worry",

    # Internet access (constructed)
    "internet": "used_internet_past_3_months",

    # Phone ownership & usage
    "con1":  "owns_mobile_phone",
    "con2a": "no_phone_cant_afford_handset",
    "con2b": "no_phone_minutes_too_expensive",
    "con2c": "no_phone_no_coverage",
    "con2d": "no_phone_reading_difficulties",
    "con2e": "no_phone_family_disapproval",
    "con2f": "no_phone_safety_concerns",
    "con2g": "no_phone_uses_someone_elses",
    "con3":  "main_reason_no_phone",
    "con4":  "used_someone_elses_phone",
    "con5":  "type_of_borrowed_phone",
    "con6":  "uses_own_sim_in_borrowed_phone",
    "con7":  "has_rules_on_phone_use",
    "con8":  "household_member_owns_phone",
    "con9":  "phone_type",                   # 1=smartphone, 2=basic
    "con10": "basic_phone_can_run_whatsapp",
    "con11": "sim_registered_in_own_name",
    "con12": "freq_phone_use",
    "con13": "used_phone_past_3_months",
    "con14": "can_read_text_message",
    "con15": "can_understand_latin_script_sms",
    "con16": "has_sent_text_message",
    "con17": "preferred_govt_communication_channel",
    "con18": "has_phone_pin_or_password",
    "con19": "can_change_phone_pin",
    "con20": "others_set_rules_on_phone_use",
    "con21": "received_scam_call_or_text",
    "con22": "sent_money_to_scammer",
    "con23": "received_offensive_messages",
    "con24": "used_internet_past_7_days",
    "con25": "used_internet_past_3_months_survey",
    "con26": "freq_internet_use",
    "con27": "purchases_data_package",
    "con28": "freq_data_package_purchase",
    "con29": "connects_only_via_free_wifi",
    "con30a": "sends_voice_messages",
    "con30b": "sends_photos_from_phone",
    "con30c": "uses_social_media_on_phone",
    "con30d": "reads_news_online",
    "con30e": "accesses_educational_content_online",
    "con30f": "earns_money_online",
    "con30g": "accesses_govt_services_online",
    "con30h": "searched_or_applied_for_job_online",

    # Barriers to smartphone
    "con31a": "no_smartphone_cant_afford",
    "con31b": "no_smartphone_data_plan_expensive",
    "con31c": "no_smartphone_no_coverage",
    "con31d": "no_smartphone_reading_difficulties",
    "con31e": "no_smartphone_family_disapproval",
    "con31f": "no_smartphone_safety_concerns",
    "con31g": "no_smartphone_uses_someone_elses",
    "con31h": "no_smartphone_no_need",
    "con32":  "main_reason_no_smartphone",

    # ID ownership (ID4D)
    "fin46":  "owns_national_id",
    "fin47":  "id_used_without_permission",
    "fin48a": "no_id_has_other_id",
    "fin48b": "no_id_no_need",
    "fin48c": "no_id_too_expensive",
    "fin48d": "no_id_lack_documents",
    "fin48e": "no_id_travel_too_far",
    "fin48f": "no_id_privacy_concerns",
    "fin49a": "id_barrier_govt_support",
    "fin49b": "id_barrier_financial_services",
    "fin49c": "id_barrier_sim_card",
    "fin49d": "id_barrier_elections",
    "fin49e": "id_barrier_job_application",
    "fin49f": "id_barrier_medical_care",
    "fin50":  "owns_online_digital_id",
    "fin51":  "used_digital_id_to_verify_identity_online",
}

df = pd.read_csv(INDIVIDUAL_CSV)
df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

# Update constants to match renamed columns
ACCOUNT_FILTER_COL = "has_any_account"
COUNTRY_COL        = "country_name"
COUNTRY_CODE_COL   = "country_code"
WEIGHT_COL         = "survey_weight"

print(df.shape)


# =============================================================================
# 2. TARGET DEFINITION — PATH 2: ACTIVE DIGITAL PAYMENTS ONLY
# =============================================================================
# Target = 1 if the person INITIATED at least one digital transaction:
#   - paid in-store with card/phone  (fin25e2)
#   - paid a bill digitally          (fin26a)
#   - bought something online        (fin26b)
# Passive receipts (wages, transfers) are excluded. This makes the
# classification harder but produces a cleaner behavioural signal.

df = df[df[ACCOUNT_FILTER_COL] == ACCOUNT_FILTER_VALUE].copy()


def to_binary_yes(series):
    """Map survey response 1 → 1, anything else (including NaN) → 0."""
    return (series == 1).astype(float)


paid_instore_digitally = to_binary_yes(df["used_card_or_phone_for_instore_purchase"])
paid_bill_digitally    = to_binary_yes(df["used_phone_or_computer_for_bill_payment"])
bought_online          = to_binary_yes(df["bought_something_online"])

df[TARGET_COL] = (
    (paid_instore_digitally + paid_bill_digitally + bought_online) >= 1
).astype(int)

# Drop rows where ALL three source variables are missing (target is unanswerable)
all_missing = (
    df["used_card_or_phone_for_instore_purchase"].isna()
    & df["used_phone_or_computer_for_bill_payment"].isna()
    & df["bought_something_online"].isna()
)
df = df[~all_missing].copy()

print(f"Target rate (Path 2 — active digital payments): {df[TARGET_COL].mean():.1%}")
print(f"Class balance: {df[TARGET_COL].value_counts(normalize=True).to_dict()}")
print(f"  └─ paid in-store digitally : {paid_instore_digitally[df.index].mean():.1%}")
print(f"  └─ paid bill digitally     : {paid_bill_digitally[df.index].mean():.1%}")
print(f"  └─ bought something online : {bought_online[df.index].mean():.1%}")

# Stash groups and weights before column drops
groups      = df[COUNTRY_COL].astype(str).copy()
group_codes = df[COUNTRY_CODE_COL].astype(str).copy()
weights     = (
    df[WEIGHT_COL].copy()
    if WEIGHT_COL in df.columns
    else pd.Series(np.ones(len(df)), index=df.index)
)


# =============================================================================
# 3. DROP LEAKAGE COLUMNS
# =============================================================================
# Three categories:
#   A. Identifiers / design variables — not features by definition
#   B. Account ownership flags — used as the sample filter, not predictors
#   C. Direct digital payment encodings — either IS the target, a component
#      of the target, or a variable that exists only because target == 1

leakage_drop = {
    # A. Identifiers / design variables
    "year",
    "country_name",
    "country_code",
    "adult_population_2023",
    "gallup_respondent_id",
    "survey_weight",

    # B. Account ownership (sample filter, not a feature)
    "has_any_account",
    "has_financial_institution_account",
    "has_mobile_money_account",
    # Barriers to account — only asked of non-account holders; all NA after filter
    "no_account_too_far",
    "no_account_fees_too_high",
    "no_account_lack_documents",
    "no_account_not_enough_money",
    "no_account_family_member_has_one",
    "no_account_distrust",

    # C. Direct digital payment encodings
    "made_or_received_digital_payment",         # Path 1 target
    "used_card_or_phone_for_instore_purchase",  # Path 2 target component
    "used_phone_or_computer_for_bill_payment",  # Path 2 target component
    "bought_something_online",                  # Path 2 target component
    "made_digital_merchant_payment",
    "has_digitally_enabled_account",
    "used_card_or_phone_for_account",           # near-identical to target
    "used_card_or_phone_for_cleaning_supplies", # subset of in-store digital
    "freq_digital_instore_payments",            # only asked when target == 1
    "main_reason_cash_only_instore",            # only asked when target == 0
    "online_purchase_payment_mode",             # only asked when fin26b == 1
    "sent_domestic_remittance_digitally",
    "received_domestic_remittance_digitally",
    "utility_paid_via_mobile_phone",
    "wage_received_via_mobile_phone",
    "govt_money_received_via_mobile",
    "agri_payment_via_mobile_phone",
    "utility_payment_method",                   # value 1 = paid from account → digital
    "domestic_remittance_method",               # value 1 = via account → digital
    "freq_sends_from_mobile_money",             # sending money IS a digital payment
    "sent_money_to_scammer",                    # downstream outcome of being a digital user
}

df_model = df.drop(columns=[c for c in leakage_drop if c in df.columns], errors="ignore")


# =============================================================================
# 4. DROP STRUCTURALLY REDUNDANT COLUMNS
# =============================================================================
# Only strict sub-cases or direct re-encodings of retained variables.
# Correlation-based pruning is deferred to Step 7.

redundant_drop = [
    "saved_any",                         # fully captured by fin17a/b/c (all retained)
    "borrowed_for_any_reason_informal",  # fin23: no signal beyond fin22a-f
    "main_reason_no_phone",              # con3: redundant with con2a-g barrier flags
    "used_internet_past_7_days",         # con24: component of `used_internet_past_3_months`
    "used_internet_past_3_months_survey",# con25: component of `used_internet_past_3_months`
    "purchases_data_package",            # con27: subsumed by con28 (frequency)
]

df_model = df_model.drop(
    columns=[c for c in redundant_drop if c in df_model.columns], errors="ignore"
)


# =============================================================================
# 5. MISSING CODE RECODING & HIGH-MISSING COLUMN REMOVAL
# =============================================================================
# Survey codes -1 to -4 mean "don't know / refused / not asked" → recode to NaN.
# Columns missing > 60 % of values are dropped, but first check whether the
# missingness is driven by small-n countries; if so, drop those countries instead.

df_model = df_model.replace([-1, -2, -3, -4], pd.NA)

country_counts    = df["country_name"].value_counts()
low_imp_countries = country_counts[country_counts < 500].index

missing_pct     = df_model.isna().mean() * 100
candidate_drops = missing_pct[missing_pct > 60].index.tolist()

high_missing_cols = []
countries_to_drop = set()

for col in candidate_drops:
    mask            = ~df["country_name"].isin(low_imp_countries)
    pct_without_low = df_model.loc[mask, col].isna().mean() * 100
    if pct_without_low < 60:
        countries_to_drop.update(low_imp_countries)
        print(f"Saved feature '{col}' by dropping low-importance countries.")
    else:
        high_missing_cols.append(col)

# Apply country-level filter consistently across all aligned series
valid_rows  = ~df["country_name"].isin(countries_to_drop)
df_model    = df_model[valid_rows]
df          = df[valid_rows]
groups      = groups[valid_rows]
group_codes = group_codes[valid_rows]
weights     = weights[valid_rows]

df_model = df_model.drop(columns=high_missing_cols, errors="ignore")

# Drop world_bank_region — high-level geographic grouping causes region leakage
if "world_bank_region" in df_model.columns:
    df_model = df_model.drop(columns=["world_bank_region"])
    print("Dropped 'world_bank_region' to prevent region leakage.")

y = df_model[TARGET_COL].astype(int)
X = df_model.drop(columns=[TARGET_COL])


# =============================================================================
# 6. OPTIONAL COUNTRY-LEVEL FEATURE MERGE
# =============================================================================
# Pulls macro/infrastructure indicators from the country-level file and
# left-joins onto individual rows. Disabled by default.

if USE_COUNTRY_LEVEL_FEATURES:
    df_country = pd.read_csv(COUNTRY_CSV)
    c = df_country[df_country["year"] == 2024].copy()
    candidate_country_cols = [
        "account_t_d", "fiaccount_t_d", "mobileaccount_t_d",
        "borrow_any_t_d", "dig_acc", "regionwb24_hi", "incomegroupwb24",
    ]
    candidate_country_cols = [col for col in candidate_country_cols if col in c.columns]
    c = c[["codewb"] + candidate_country_cols].drop_duplicates("codewb")
    X = X.merge(c, left_on=group_codes.values, right_on="codewb", how="left")
    X = X.drop(columns=["codewb"], errors="ignore")


# =============================================================================
# 7. HIGH-CORRELATION FEATURE REMOVAL (|r| > 0.85)
# =============================================================================
# For each correlated pair, drop the column with more missing values
# (keeps the cleaner signal). This prevents tautological prediction
# and ensures epistemic integrity.

num_cols_pre = X.select_dtypes(include=["number"]).columns.tolist()
corr_matrix  = X[num_cols_pre].corr().abs()
to_drop_corr = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.85:
            c1, c2   = corr_matrix.columns[i], corr_matrix.columns[j]
            drop_col = c2 if X[c1].isna().mean() <= X[c2].isna().mean() else c1
            to_drop_corr.add(drop_col)

X = X.drop(columns=list(to_drop_corr), errors="ignore")


# =============================================================================
# 8. PREPROCESSORS
# =============================================================================
# preprocessor_lr   — for linear models: median impute + StandardScaler + OHE
# preprocessor_tree — for tree models:   median impute only + OHE (no scaling)

numeric_cols     = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

preprocessor_lr = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ],
    remainder="drop",
)

preprocessor_tree = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ],
    remainder="drop",
)


# =============================================================================
# 9. MODEL DEFINITIONS
# =============================================================================
# Logistic Regression: balanced class weights, liblinear solver.
# XGBoost: constant fill (-999) imputation for speed; scale_pos_weight
#          compensates for class imbalance (neg:pos ratio).

log_reg = Pipeline([
    ("preprocess", preprocessor_lr),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )),
])

pos = int(y.sum())
neg = int((1 - y).sum())
scale_pos_weight = neg / max(pos, 1)

num_cols_xgb = X.select_dtypes(include=np.number).columns.tolist()
cat_cols_xgb = [c for c in X.columns if c not in num_cols_xgb]

preprocess_xgb = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="constant", fill_value=-999), num_cols_xgb),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), cat_cols_xgb),
    ],
    sparse_threshold=1.0,
    n_jobs=1,
)

xgb = Pipeline([
    ("preprocess", preprocess_xgb),
    ("model", XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=2,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )),
])

print(X.shape)


# =============================================================================
# 10. LEAVE-ONE-COUNTRY-OUT (LOCO) CROSS-VALIDATION
# =============================================================================
# For each fold: train on all countries except one, evaluate on the held-out
# country. Rotates through every country (69 folds).
# Threshold is tuned per fold on training data to maximise F1.

logo = LeaveOneGroupOut()


def evaluate_model_cv(model, model_name, X, y, groups):
    fold_rows, pred_rows = [], []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
        test_country    = groups.iloc[test_idx].iloc[0]

        model.fit(X_train, y_train)
        proba_train = model.predict_proba(X_train)[:, 1]

        # Tune decision threshold on training fold to maximise F1
        best_f1, best_thresh = 0, 0.5
        for t in np.arange(0.3, 0.7, 0.05):
            f1 = f1_score(y_train, (proba_train >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t

        proba = model.predict_proba(X_test)[:, 1]
        pred  = (proba >= best_thresh).astype(int)

        roc_auc_train = (
            np.nan if y_train.nunique() < 2
            else roc_auc_score(y_train, proba_train)
        )

        fold_metrics = {
            "model":              model_name,
            "fold":               fold,
            "test_country":       test_country,
            "n_test":             len(test_idx),
            "inactive_rate_test": float(y_test.mean()),
            "roc_auc_train":      roc_auc_train,
            "roc_auc":   np.nan if y_test.nunique() < 2 else roc_auc_score(y_test, proba),
            "pr_auc":    average_precision_score(y_test, proba),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall":    recall_score(y_test, pred, zero_division=0),
            "f1":        f1_score(y_test, pred, zero_division=0),
            "best_thresh": best_thresh,
        }
        fold_rows.append(fold_metrics)
        pred_rows.append(pd.DataFrame({
            "model":   model_name,
            "fold":    fold,
            "country": groups.iloc[test_idx].values,
            "y_true":  y_test.values,
            "y_prob":  proba,
            "y_pred":  pred,
        }, index=X_test.index))

        auc_str = f"{fold_metrics['roc_auc']:.3f}" if not np.isnan(fold_metrics["roc_auc"]) else "NA"
        print(f"[{model_name}] Fold {fold:03d} | {test_country:<30} | n={len(test_idx):>5} | ROC-AUC={auc_str}")

    return pd.DataFrame(fold_rows), pd.concat(pred_rows).sort_index()


lr_fold_df,  lr_preds_df  = evaluate_model_cv(log_reg, "logistic_regression", X, y, groups)
xgb_fold_df, xgb_preds_df = evaluate_model_cv(xgb,     "xgboost",            X, y, groups)

all_fold_metrics = pd.concat([lr_fold_df, xgb_fold_df], ignore_index=True)
all_preds        = pd.concat([lr_preds_df, xgb_preds_df])

summary = (
    all_fold_metrics
    .groupby("model")[["roc_auc", "pr_auc", "precision", "recall", "f1"]]
    .agg(["mean", "std"])
)
summary.columns = [f"{a}_{b}" for a, b in summary.columns]
summary = summary.reset_index()

all_fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv",     index=False)
all_preds.to_csv(        OUTPUT_DIR / "oof_predictions.csv", index=False)
summary.to_csv(          OUTPUT_DIR / "model_summary.csv",   index=False)


# =============================================================================
# 11. FINAL FITS ON FULL DATA
# =============================================================================

log_reg.fit(X, y)
xgb.fit(X, y)
print("Fitted both models on full dataset.")


# =============================================================================
# 12. FEATURE IMPORTANCE EXPORTS
# =============================================================================

# (a) Logistic regression coefficients
lr_feature_names = log_reg.named_steps["preprocess"].get_feature_names_out()
lr_coef_df = pd.DataFrame({
    "feature":         lr_feature_names,
    "coefficient":     log_reg.named_steps["model"].coef_[0],
    "abs_coefficient": np.abs(log_reg.named_steps["model"].coef_[0]),
}).sort_values("abs_coefficient", ascending=False)
lr_coef_df.to_csv(OUTPUT_DIR / "logistic_coefficients.csv", index=False)
print("Saved: logistic_coefficients.csv")

# (b) XGBoost gain importance
xgb_feature_names = xgb.named_steps["preprocess"].get_feature_names_out()
xgb_imp_df = pd.DataFrame({
    "feature":    xgb_feature_names,
    "importance": xgb.named_steps["model"].feature_importances_,
}).sort_values("importance", ascending=False)
xgb_imp_df.to_csv(OUTPUT_DIR / "xgb_feature_importance.csv", index=False)
print("Saved: xgb_feature_importance.csv")

# (c) Random Forest permutation importance (memory-safe: 20 k-row sample)
SAMPLE_N    = min(20_000, len(X))
X_sample    = X.sample(SAMPLE_N, random_state=RANDOM_STATE)
y_sample    = y.loc[X_sample.index]
X_tree_sample = preprocessor_tree.fit_transform(X_sample)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=1,
)
rf.fit(X_tree_sample, y_sample)
print("RF fitted.")

perm = permutation_importance(
    rf, X_tree_sample, y_sample,
    n_repeats=2,
    random_state=RANDOM_STATE,
    n_jobs=1,
)
rf_perm_df = pd.DataFrame({
    "feature":              preprocessor_tree.get_feature_names_out(),
    "perm_importance_mean": perm.importances_mean,
}).sort_values("perm_importance_mean", ascending=False)
rf_perm_df.to_csv(OUTPUT_DIR / "rf_permutation_importance.csv", index=False)
print("Saved: rf_permutation_importance.csv")


# =============================================================================
# 13. GLOBAL SHAP
# =============================================================================
# TreeExplainer on a random subsample of the preprocessed feature matrix.

if RUN_SHAP:
    try:
        X_tree_df = pd.DataFrame(
            preprocessor_tree.transform(X),
            columns=preprocessor_tree.get_feature_names_out(),
            index=X.index,
        )
        X_shap_global = X_tree_df.sample(
            min(SHAP_SAMPLE_N, len(X_tree_df)), random_state=RANDOM_STATE
        )
        explainer_global   = shap.TreeExplainer(xgb.named_steps["model"])
        shap_values_global = explainer_global.shap_values(X_shap_global)

        pd.DataFrame({
            "feature":       X_shap_global.columns,
            "mean_abs_shap": np.abs(shap_values_global).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).to_csv(
            OUTPUT_DIR / "shap_importance.csv", index=False
        )
        pd.DataFrame(
            shap_values_global,
            columns=X_shap_global.columns,
            index=X_shap_global.index,
        ).to_csv(OUTPUT_DIR / "shap_values_sample.csv", index=True)

        # Global summary plot
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values_global, X_shap_global, max_display=10, show=True)

    except Exception as e:
        (OUTPUT_DIR / "shap_error.txt").write_text(str(e))


# =============================================================================
# 14. SHAP STABILITY ACROSS LOCO FOLDS
# =============================================================================
# For each fold, fit a fresh XGBoost, compute per-country mean |SHAP|,
# then measure cross-country variance to identify universal vs
# context-dependent features.

SHAP_FOLD_SAMPLE  = 500  # rows sampled per country for speed
shap_fold_records = []

for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
    X_train      = X.iloc[train_idx].copy()
    y_train      = y.iloc[train_idx].copy()
    X_test       = X.iloc[test_idx].copy()
    test_country = groups.iloc[test_idx].iloc[0]

    fold_xgb = Pipeline([
        ("preprocess", preprocessor_tree),
        ("model", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss",
            reg_lambda=1.0, min_child_weight=2,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, n_jobs=4,
        )),
    ])
    fold_xgb.fit(X_train, y_train)

    X_test_transformed = pd.DataFrame(
        fold_xgb.named_steps["preprocess"].transform(X_test),
        columns=fold_xgb.named_steps["preprocess"].get_feature_names_out(),
    )
    X_test_sample = X_test_transformed.sample(
        min(SHAP_FOLD_SAMPLE, len(X_test_transformed)),
        random_state=RANDOM_STATE,
    )

    explainer        = shap.TreeExplainer(fold_xgb.named_steps["model"])
    fold_shap_values = explainer.shap_values(X_test_sample)

    mean_abs = np.abs(fold_shap_values).mean(axis=0)
    record   = {"country": test_country, "fold": fold}
    record.update(dict(zip(X_test_sample.columns, mean_abs)))
    shap_fold_records.append(record)

    print(f"SHAP fold {fold:03d} | {test_country}")

# Stability summary
shap_fold_df = pd.DataFrame(shap_fold_records)
shap_fold_df.to_csv(OUTPUT_DIR / "shap_by_country.csv", index=False)

feature_cols = [c for c in shap_fold_df.columns if c not in ["country", "fold"]]

stability_df = pd.DataFrame({
    "feature":   feature_cols,
    "mean_shap": shap_fold_df[feature_cols].mean(),
    "std_shap":  shap_fold_df[feature_cols].std(),
    "cv_shap":   (
        shap_fold_df[feature_cols].std()
        / shap_fold_df[feature_cols].mean().replace(0, np.nan)
    ),
}).sort_values("mean_shap", ascending=False).reset_index(drop=True)

stability_df["stability"] = pd.cut(
    stability_df["cv_shap"],
    bins=[0, 0.5, 1.0, np.inf],
    labels=["stable", "moderate", "unstable"],
)
stability_df.to_csv(OUTPUT_DIR / "shap_stability.csv", index=False)
print(stability_df.head(20).to_string())


# =============================================================================
# 15. COUNTRY-PAIR SHAP COMPARISON (INDIA vs ZAMBIA)
# =============================================================================

shap_by_country = pd.read_csv(OUTPUT_DIR / "shap_by_country.csv")

COUNTRY_A = "India"
COUNTRY_B = "Zambia"
TOP_N     = 20

feature_cols = [c for c in shap_by_country.columns if c not in ["country", "fold"]]


def country_mean_shap(df, country_name):
    rows = df[df["country"].str.strip() == country_name]
    if rows.empty:
        raise ValueError(
            f"'{country_name}' not found. "
            f"Available: {sorted(df['country'].unique())[:10]} ..."
        )
    return rows[feature_cols].mean().rename(country_name)


shap_a = country_mean_shap(shap_by_country, COUNTRY_A)
shap_b = country_mean_shap(shap_by_country, COUNTRY_B)

compare_df = (
    pd.DataFrame({COUNTRY_A: shap_a, COUNTRY_B: shap_b})
    .assign(
        max_shap=lambda d: d[[COUNTRY_A, COUNTRY_B]].max(axis=1),
        diff    =lambda d: d[COUNTRY_A] - d[COUNTRY_B],
        ratio   =lambda d: (d[COUNTRY_A] + 1e-9) / (d[COUNTRY_B] + 1e-9),
    )
    .sort_values("max_shap", ascending=False)
)
compare_df.to_csv(OUTPUT_DIR / f"shap_compare_{COUNTRY_A}_{COUNTRY_B}.csv")
print(f"\nTop {TOP_N} features by max SHAP across both countries:\n")
print(compare_df.head(TOP_N).to_string())

# Side-by-side bar chart
top_features = compare_df.head(TOP_N).index.tolist()
plot_df      = compare_df.loc[top_features, [COUNTRY_A, COUNTRY_B]].iloc[::-1]

fig, ax = plt.subplots(figsize=(11, 7))
y_pos   = np.arange(len(plot_df))
bar_h   = 0.35

ax.barh(y_pos + bar_h / 2, plot_df[COUNTRY_A], bar_h, label=COUNTRY_A, color="#2196F3", alpha=0.85)
ax.barh(y_pos - bar_h / 2, plot_df[COUNTRY_B], bar_h, label=COUNTRY_B, color="#FF9800", alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(
    [f.replace("num__", "").replace("cat__", "") for f in plot_df.index], fontsize=9
)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title(f"Top {TOP_N} Features — {COUNTRY_A} vs {COUNTRY_B}", fontsize=13, fontweight="bold")
ax.legend()
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"shap_compare_bar_{COUNTRY_A}_{COUNTRY_B}.png", dpi=150)
plt.show()

# Scatter: all features, annotate top 15
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(compare_df[COUNTRY_B], compare_df[COUNTRY_A], s=18, alpha=0.5, color="#555")
for feat, row in compare_df.head(15).iterrows():
    label = feat.replace("num__", "").replace("cat__", "")
    ax.annotate(label, (row[COUNTRY_B], row[COUNTRY_A]),
                fontsize=7, alpha=0.85, xytext=(4, 2), textcoords="offset points")
lim = compare_df[[COUNTRY_A, COUNTRY_B]].max().max() * 1.05
ax.plot([0, lim], [0, lim], "r--", lw=1, label="Equal importance")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel(f"{COUNTRY_B} mean |SHAP|", fontsize=11)
ax.set_ylabel(f"{COUNTRY_A} mean |SHAP|", fontsize=11)
ax.set_title(f"Feature Importance Scatter — {COUNTRY_A} vs {COUNTRY_B}", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"shap_compare_scatter_{COUNTRY_A}_{COUNTRY_B}.png", dpi=150)
plt.show()

# Most divergent features table
divergence = (
    compare_df
    .assign(abs_diff=compare_df["diff"].abs())
    .sort_values("abs_diff", ascending=False)
    .head(15)[[  "abs_diff", COUNTRY_A, COUNTRY_B, "diff"]]
)
divergence.index = [i.replace("num__", "").replace("cat__", "") for i in divergence.index]
print(f"\nTop 15 most divergent features ({COUNTRY_A} − {COUNTRY_B}):\n")
print(divergence.to_string())
divergence.to_csv(OUTPUT_DIR / f"shap_divergence_{COUNTRY_A}_{COUNTRY_B}.csv")


# =============================================================================
# 16. RUN METADATA EXPORT
# =============================================================================

y = df_model[TARGET_COL].astype(int)

weights_aligned = (
    pd.Series(weights, index=y.index)
    if not hasattr(weights, "reindex")
    else weights.reindex(y.index)
)

metadata = {
    "n_rows_final":                  int(len(X)),
    "n_features_final_pre_encoding": int(X.shape[1]),
    "n_countries":                   int(groups.nunique()),
    "inactive_rate":                 float(y.mean()),
    "weighted_inactive_rate":        float(
        (y * weights_aligned).sum() / weights_aligned.sum()
    ),
    "numeric_cols":               numeric_cols,
    "categorical_cols":           categorical_cols,
    "dropped_high_missing_cols":  high_missing_cols,
    "dropped_high_corr_cols":     sorted(list(to_drop_corr)),
    "used_country_level_features": USE_COUNTRY_LEVEL_FEATURES,
}

with open(OUTPUT_DIR / "run_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved outputs to:", OUTPUT_DIR)
print(summary.to_string(index=False))

shap_df = pd.read_csv(OUTPUT_DIR / "shap_importance.csv")
print(shap_df.head(10).to_string(index=False))


# =============================================================================
# 17. MLP BENCHMARK (4 hidden layers, global split)
# =============================================================================
# Note: trained on the full dataset (no LOCO), so AUC is not comparable
# to the LOCO-validated scores above. Use as an indicative upper bound only.

print("\nSTEP 17: 4-LAYER MLP BENCHMARK")

try:
    X_nn = preprocessor_lr.fit_transform(X)
    if hasattr(X_nn, "toarray"):
        X_nn = X_nn.toarray()

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32, 16),
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        alpha=0.0005,
        early_stopping=True,
        max_iter=200,
        random_state=RANDOM_STATE,
    )
    print("Training MLP...")
    mlp.fit(X_nn, y)

    probs = mlp.predict_proba(X_nn)[:, 1]
    preds = (probs >= 0.5).astype(int)

    mlp_metrics = pd.DataFrame([{
        "Model":     "MLP_4layer",
        "AUC":       roc_auc_score(y, probs),
        "PR-AUC":    average_precision_score(y, probs),
        "Accuracy":  accuracy_score(y, preds),
        "Precision": precision_score(y, preds),
        "Recall":    recall_score(y, preds),
        "F1":        f1_score(y, preds),
    }])
    print(mlp_metrics.round(3).to_string(index=False))
    mlp_metrics.to_csv(OUTPUT_DIR / "mlp_metrics.csv", index=False)
    print("Saved: mlp_metrics.csv")

except Exception as e:
    print("MLP error:", e)


# =============================================================================
# 18. CLASS IMBALANCE DIAGNOSTIC
# =============================================================================
# Checks global and per-country balance.
# Rule of thumb:
#   Global ratio < 3:1  → existing scale_pos_weight handling is sufficient
#   Country-level ratio > 9:1 in many countries → consider dropping those folds

print("\nSTEP 18: CLASS IMBALANCE DIAGNOSTIC")

pos_count = int(y.sum())
neg_count = int((1 - y).sum())
total     = len(y)
ratio     = neg_count / max(pos_count, 1)

print(f"\nGlobal class distribution:")
print(f"  Positive (made digital payment): {pos_count:>7,}  ({pos_count/total:.1%})")
print(f"  Negative (did not)             : {neg_count:>7,}  ({neg_count/total:.1%})")
print(f"  Neg:Pos ratio                  : {ratio:.2f}:1")

if ratio < 1.5:
    print("  → Well balanced.")
elif ratio < 3:
    print("  → Mildly imbalanced. Existing class_weight/scale_pos_weight are sufficient.")
elif ratio < 9:
    print("  → Moderately imbalanced. Monitor per-country F1.")
else:
    print("  → Severely imbalanced. Consider dropping countries with <10% positive rate.")

country_balance = (
    pd.DataFrame({"country": groups, "y": y})
    .groupby("country")["y"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "pos_rate", "count": "n"})
    .assign(neg_pos_ratio=lambda d: (1 - d["pos_rate"]) / d["pos_rate"].clip(lower=0.01))
    .sort_values("pos_rate")
)
print("\nPer-country positive rates (bottom 10):")
print(country_balance.head(10).to_string())
print("\nPer-country positive rates (top 10):")
print(country_balance.tail(10).to_string())
country_balance.to_csv(OUTPUT_DIR / "class_balance_by_country.csv")


# =============================================================================
# 19. COUNTRY FIXED-EFFECTS LOGISTIC REGRESSION
# =============================================================================
# Models A / B / C decompose individual vs country effects on AUC.
#   A: individual socioeconomic features only
#   B: country dummy only
#   C: individual + country (combined)

print("\nSTEP 19: COUNTRY vs INDIVIDUAL FACTOR AUC DECOMPOSITION")

INDIV_FEATS = [
    f for f in [
        "age_years", "gender", "education_level", "income_quintile",
        "in_workforce", "urban_rural", "used_internet_past_3_months",
    ]
    if f in X.columns
]

df_compare = X[INDIV_FEATS].copy()
df_compare["country"] = groups.values
df_compare["target"]  = y.values
df_compare = df_compare.dropna()


def fit_auc(X_model, y_model):
    """Fit a penalised logistic regression and return ROC-AUC on training data."""
    num = X_model.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X_model.columns if c not in num]

    model = Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first")),
            ]), cat),
        ])),
        ("logit", LogisticRegression(
            penalty="l2", max_iter=2000,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    model.fit(X_model, y_model)
    pred = model.predict_proba(X_model)[:, 1]
    return roc_auc_score(y_model, pred)


auc_individual = fit_auc(df_compare[INDIV_FEATS],             df_compare["target"])
auc_country    = fit_auc(df_compare[["country"]],             df_compare["target"])
auc_combined   = fit_auc(df_compare[INDIV_FEATS + ["country"]], df_compare["target"])

comparison = pd.DataFrame({
    "model": ["Individual only", "Country only", "Individual + Country"],
    "AUC":   [auc_individual, auc_country, auc_combined],
}).sort_values("AUC", ascending=False)

print("\nAUC comparison:")
print(comparison.round(3).to_string(index=False))
comparison.to_csv(OUTPUT_DIR / "country_vs_individual_auc.csv", index=False)


# =============================================================================
# 20. LOCO PERFORMANCE BY COUNTRY
# =============================================================================

print("\nSTEP 20: LOCO COUNTRY PERFORMANCE")

country_perf = (
    xgb_preds_df
    .assign(country=groups.loc[xgb_preds_df.index])
    .groupby("country")
    .apply(lambda d: pd.Series({
        "n":   len(d),
        "AUC": roc_auc_score(d.y_true, d.y_prob) if d.y_true.nunique() > 1 else np.nan,
        "F1":  f1_score(d.y_true, d.y_pred),
    }))
    .reset_index()
    .sort_values("F1")
)
country_perf.to_csv(OUTPUT_DIR / "loco_country_metrics.csv", index=False)
print(country_perf.head(10))


# =============================================================================
# 21. COUNTRY HETEROGENEITY ANALYSIS
# =============================================================================
# Tests whether model transferability (LOCO AUC/F1) is associated with
# a country's baseline inactivity level, then fits an OLS regression
# controlling for sample size.

print("\nSTEP 21: COUNTRY HETEROGENEITY")

country_rates = (
    pd.DataFrame({"country": groups, "target": y})
    .groupby("country")
    .agg(n=("target", "size"), inactivity_rate=("target", "mean"))
    .reset_index()
)

country_loco = (
    all_fold_metrics
    .groupby("test_country")
    .agg(auc=("roc_auc", "mean"), f1=("f1", "mean"))
    .reset_index()
    .rename(columns={"test_country": "country"})
)

heterogeneity = country_rates.merge(country_loco, on="country", how="inner")

corr_auc, p_auc = pearsonr(heterogeneity["inactivity_rate"], heterogeneity["auc"])
corr_f1,  p_f1  = pearsonr(heterogeneity["inactivity_rate"], heterogeneity["f1"])

print(f"\nInactivity vs AUC: r={corr_auc:.3f}, p={p_auc:.4f}")
print(f"Inactivity vs F1 : r={corr_f1:.3f}, p={p_f1:.4f}")

print("\nLowest-performing countries (AUC):")
print(
    heterogeneity.sort_values("auc").head(10)
    [["country", "auc", "f1", "inactivity_rate", "n"]]
    .round(3).to_string(index=False)
)

heterogeneity.to_csv(OUTPUT_DIR / "country_heterogeneity_metrics.csv", index=False)
heterogeneity.sort_values("auc").head(10).to_csv(OUTPUT_DIR / "lowest_auc_countries.csv", index=False)

for metric, ylabel in [("auc", "LOCO AUC"), ("f1", "LOCO F1")]:
    plt.figure(figsize=(6, 5))
    plt.scatter(heterogeneity["inactivity_rate"], heterogeneity[metric])
    plt.xlabel("Country inactivity rate")
    plt.ylabel(ylabel)
    plt.title(f"Country inactivity vs model {metric.upper()}")
    plt.show()

# OLS regression: AUC ~ inactivity_rate + n
reg_df = heterogeneity[["auc", "inactivity_rate", "n"]].dropna()
X_reg  = sm.add_constant(reg_df[["inactivity_rate", "n"]])
print(sm.OLS(reg_df["auc"], X_reg).fit().summary())


# =============================================================================
# 22. FAIRNESS AUDIT — EQUALIZED ODDS
# =============================================================================
# Uses XGBoost out-of-fold predictions. Flags subgroups (gender, urban/rural,
# income) where TPR deviates more than 10 pp from the overall rate.

print("\nSTEP 22: FAIRNESS AUDIT")

try:
    xgb_oof = xgb_preds_df[["y_true", "y_pred", "y_prob"]].copy()

    demo_cols = {}
    if "gender" in df.columns:
        demo_cols["gender"]     = df["gender"].map({1: "female", 2: "male"})
    if "urban_rural" in df.columns:
        demo_cols["urban_rural"] = df["urban_rural"].map({1: "rural", 2: "urban"})
    if "income_quintile" in df.columns:
        demo_cols["income_group"] = df["income_quintile"].apply(
            lambda x: "low_income"  if x in [1, 2]
            else      "high_income" if x in [4, 5]
            else      "middle_income"
        )

    audit_df = xgb_oof.join(pd.DataFrame(demo_cols)).dropna()

    def fairness_metrics(df_sub, label):
        tn, fp, fn, tp = confusion_matrix(
            df_sub["y_true"], df_sub["y_pred"], labels=[0, 1]
        ).ravel()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        f1  = 2 * tp / max(2 * tp + fp + fn, 1)
        return {"subgroup": label, "n": len(df_sub),
                "TPR": tpr, "FPR": fpr, "F1": f1,
                "pos_rate": df_sub["y_true"].mean()}

    audit_rows = [fairness_metrics(audit_df, "overall")]

    for col, vals in [
        ("gender",      audit_df["gender"].dropna().unique()),
        ("urban_rural", audit_df["urban_rural"].dropna().unique()),
    ]:
        for val in vals:
            sub = audit_df[audit_df[col] == val]
            if len(sub) > 50:
                audit_rows.append(fairness_metrics(sub, f"{col}={val}"))

    for val in ["low_income", "high_income"]:
        sub = audit_df[audit_df["income_group"] == val]
        if len(sub) > 50:
            audit_rows.append(fairness_metrics(sub, f"income={val}"))

    # Intersectional: gender × urban_rural
    for g in audit_df["gender"].dropna().unique():
        for u in audit_df["urban_rural"].dropna().unique():
            sub = audit_df[(audit_df["gender"] == g) & (audit_df["urban_rural"] == u)]
            if len(sub) > 50:
                audit_rows.append(fairness_metrics(sub, f"gender={g} × urban_rural={u}"))

    fairness_df = pd.DataFrame(audit_rows)
    fairness_df.to_csv(OUTPUT_DIR / "fairness_audit.csv", index=False)
    print("\nFairness audit results:\n", fairness_df.to_string(index=False))

    overall_tpr = fairness_df.loc[fairness_df["subgroup"] == "overall", "TPR"].iloc[0]
    flagged     = fairness_df[
        (fairness_df["subgroup"] != "overall")
        & ((fairness_df["TPR"] - overall_tpr).abs() > 0.10)
    ]
    if not flagged.empty:
        print(f"\n⚠  Subgroups with TPR deviation > 10pp from overall ({overall_tpr:.3f}):")
        print(flagged[["subgroup", "TPR", "F1", "n"]].to_string(index=False))
    else:
        print(f"\n✓ No subgroup deviates more than 10pp in TPR from overall ({overall_tpr:.3f})")

except Exception as e:
    print("Fairness audit error:", e)


# =============================================================================
# 23. ALE PLOTS (Accumulated Local Effects)
# =============================================================================
# Preferred over PDPs because income, education, and urban/rural are
# correlated — ALE conditions on local neighbourhoods and avoids
# impossible feature combinations.

print("\nSTEP 23: ALE PLOTS")

try:
    X_ale = pd.DataFrame(
        xgb.named_steps["preprocess"].transform(X),
        columns=xgb.named_steps["preprocess"].get_feature_names_out(),
        index=X.index,
    )

    top_ale_features = (
        xgb_imp_df[xgb_imp_df["feature"].str.startswith("num__")]
        .head(6)["feature"]
        .tolist()
    )

    if top_ale_features:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for i, feat in enumerate(top_ale_features):
            if feat in X_ale.columns:
                ale_eff = ale(
                    X=X_ale, model=xgb.named_steps["model"],
                    feature=[feat], grid_size=20,
                    include_CI=True, plot=False,
                )
                ax = axes.flatten()[i]
                ax.plot(ale_eff.index, ale_eff["eff"], color="#2196F3", lw=2)
                ax.fill_between(
                    ale_eff.index, ale_eff["lowerCI_95%"], ale_eff["upperCI_95%"],
                    alpha=0.2, color="#2196F3",
                )
                ax.axhline(0, color="black", lw=0.8, linestyle="--")
                ax.set_title(feat.replace("num__", ""), fontsize=9)
                ax.set_xlabel("Feature value")
                ax.set_ylabel("ALE effect")

        plt.suptitle(
            "Accumulated Local Effects — Top 6 Numeric Features (XGBoost)",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "ale_plots.png", dpi=150)
        plt.show()
        print("Saved: ale_plots.png")

except ImportError:
    print("PyALE not installed. Run: pip install PyALE")
except Exception as e:
    print("ALE plot error:", e)


# =============================================================================
# 24. SHAP INTERACTION HEATMAP
# =============================================================================
# Computes pairwise SHAP interactions on a small sample, then subsets
# to the top-10 features for readability.

print("\nSTEP 24: SHAP INTERACTION VALUES")

try:
    INTERACTION_SAMPLE_N = 200

    X_tree_full = pd.DataFrame(
        xgb.named_steps["preprocess"].transform(X),
        columns=xgb.named_steps["preprocess"].get_feature_names_out(),
        index=X.index,
    )
    X_int_sample = X_tree_full.sample(
        min(INTERACTION_SAMPLE_N, len(X_tree_full)), random_state=RANDOM_STATE
    )

    explainer_int    = shap.TreeExplainer(xgb.named_steps["model"])
    shap_interactions = explainer_int.shap_interaction_values(X_int_sample)

    interaction_matrix = np.abs(shap_interactions).mean(axis=0)
    interaction_df     = pd.DataFrame(
        interaction_matrix,
        index=X_tree_full.columns,
        columns=X_tree_full.columns,
    )

    # Subset to top-10 features for readability
    top_feats      = [f for f in xgb_imp_df.head(10)["feature"] if f in interaction_df.columns]
    interaction_df = interaction_df.loc[top_feats, top_feats]
    interaction_df.index   = interaction_df.index.str.replace("num__", "").str.replace("cat__", "")
    interaction_df.columns = interaction_df.columns.str.replace("num__", "").str.replace("cat__", "")

    interaction_df.to_csv(OUTPUT_DIR / "shap_interaction_matrix.csv")

    plt.figure(figsize=(9, 7))
    plt.imshow(interaction_df, aspect="auto")
    plt.xticks(range(len(interaction_df.columns)), interaction_df.columns, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(interaction_df.index)),   interaction_df.index,   fontsize=8)
    plt.colorbar(label="Mean |interaction SHAP|")
    plt.title("Top Feature Interactions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_interaction_heatmap.png", dpi=150)
    plt.show()

    # Strongest off-diagonal pairs
    tmp = interaction_df.copy()
    np.fill_diagonal(tmp.values, 0)
    strongest = (
        tmp.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_A", "level_1": "feature_B", 0: "interaction"})
        .query("feature_A < feature_B")
        .sort_values("interaction", ascending=False)
    )
    print("\nTop interactions:")
    print(strongest.head(10).to_string(index=False))

except Exception as e:
    print("SHAP interaction error:", e)


# =============================================================================
# 25. SEGMENTED BARRIER ANALYSIS
# =============================================================================
# Among inactive users only, compute mean SHAP per demographic subgroup.
# Most-negative mean SHAP → strongest contributors toward predicted inactivity.

print("\nSTEP 25: SEGMENTED BARRIER ANALYSIS")

inactive_mask = (y == 0)
X_inactive    = X.loc[inactive_mask].copy()

segments = [
    ["gender"],
    ["urban_rural"],
    ["income_quintile"],
    ["gender", "urban_rural"],
    ["gender", "income_quintile"],
]

rows             = []
explainer_seg    = shap.TreeExplainer(xgb.named_steps["model"])

for seg in segments:
    for grp, subset in X_inactive.groupby(seg):
        if len(subset) < 200:
            continue

        X_proc = pd.DataFrame(
            xgb.named_steps["preprocess"].transform(subset),
            columns=xgb.named_steps["preprocess"].get_feature_names_out(),
            index=subset.index,
        )
        shap_vals = explainer_seg.shap_values(X_proc)
        mean_shap = pd.Series(shap_vals.mean(axis=0), index=X_proc.columns)

        for feature, value in mean_shap.nsmallest(5).items():
            rows.append({
                "segment":  "_".join(seg),
                "group":    str(grp),
                "feature":  feature.replace("num__", "").replace("cat__", ""),
                "mean_shap": value,
                "n":         len(subset),
            })

barriers_df = pd.DataFrame(rows)
barriers_df.to_csv(OUTPUT_DIR / "segmented_barriers.csv", index=False)
print("\nTop subgroup barriers:")
print(barriers_df.sort_values("mean_shap").head(20).to_string(index=False))


# =============================================================================
# 26. TRAJECTORY ANALYSIS — 2021/22 vs 2025 SHAP SHIFT
# =============================================================================
# Trains an XGBoost on each wave using a common feature set and compares
# mean |SHAP| importances to surface features that have grown or shrunk
# in explanatory power between surveys.

print("\nSTEP 26: TRAJECTORY ANALYSIS (2021/22 → 2025)")

TRAJ_TARGET   = "made_or_received_digital_payment"
TRAJ_FEATURES = [
    "gender", "age_years", "education_level", "income_quintile",
    "in_workforce", "urban_rural",
    "has_financial_institution_account", "has_mobile_money_account",
    "saved_any", "borrowed_any",
    "wage_receipt_method", "govt_transfer_receipt_method",
    "has_debit_card", "owns_mobile_phone", "used_internet_past_3_months",
]

RENAME_2021 = {
    "economy":            "country_name",
    "female":             "gender",
    "age":                "age_years",
    "educ":               "education_level",
    "inc_q":              "income_quintile",
    "emp_in":             "in_workforce",
    "urbanicity_f2f":     "urban_rural",
    "account":            "has_any_account",
    "account_fin":        "has_financial_institution_account",
    "account_mob":        "has_mobile_money_account",
    "saved":              "saved_any",
    "borrowed":           "borrowed_any",
    "receive_wages":      "wage_receipt_method",
    "receive_transfers":  "govt_transfer_receipt_method",
    "fin2":               "has_debit_card",
    "mobileowner":        "owns_mobile_phone",
    "internetaccess":     "used_internet_past_3_months",
    "anydigpayment":      TRAJ_TARGET,
}


def load_csv_safe(path):
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    raise Exception(f"Cannot load {path}")


def prepare_wave(df_raw, rename_map):
    df_raw = df_raw.rename(columns=rename_map)
    if "has_any_account" in df_raw.columns:
        df_raw = df_raw[df_raw["has_any_account"] == 1]
    df_raw = df_raw.replace([-1, -2, -3, -4], np.nan)
    df_raw = df_raw[df_raw[TRAJ_TARGET].notna()]
    df_raw[TRAJ_TARGET] = (df_raw[TRAJ_TARGET] == 1).astype(int)

    cols = [c for c in TRAJ_FEATURES if c in df_raw.columns]
    X_w  = df_raw[cols].copy()
    for c in X_w.columns:
        X_w[c] = pd.to_numeric(X_w[c].astype(object), errors="ignore")
    return X_w, df_raw[TRAJ_TARGET]


def fit_shap_wave(X_w, y_w, label):
    num = X_w.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X_w.columns if c not in num]

    model = Pipeline([
        ("pre", ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat),
        ])),
        ("xgb", XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=42, eval_metric="logloss",
        )),
    ])
    model.fit(X_w, y_w)

    Xt     = pd.DataFrame(
        model.named_steps["pre"].transform(X_w),
        columns=model.named_steps["pre"].get_feature_names_out(),
    )
    sample = Xt.sample(min(1000, len(Xt)), random_state=42)
    sv     = shap.TreeExplainer(model.named_steps["xgb"]).shap_values(sample)

    return pd.Series(np.abs(sv).mean(0), index=sample.columns, name=label)


print("Loading 2021 wave...")
df21       = load_csv_safe(WAVE_2021_CSV)
X21, y21   = prepare_wave(df21, RENAME_2021)

print("Loading 2025 wave...")
df25_traj  = load_csv_safe(INDIVIDUAL_CSV)
X25, y25   = prepare_wave(df25_traj, {**RENAME_2021, "anydigpayment": TRAJ_TARGET})

common_cols = list(set(X21.columns) & set(X25.columns))
X21, X25    = X21[common_cols], X25[common_cols]

print("Fitting 2021 XGBoost...")
shap21 = fit_shap_wave(X21, y21, "2021")
print("Fitting 2025 XGBoost...")
shap25 = fit_shap_wave(X25, y25, "2025")

traj_compare = pd.DataFrame({"2021": shap21, "2025": shap25}).dropna()
traj_compare["change"] = traj_compare["2025"] - traj_compare["2021"]
traj_compare = traj_compare.sort_values("2025", ascending=False)
print(traj_compare.head(20))
traj_compare.to_csv(OUTPUT_DIR / "trajectory_compare.csv")

# Bar chart of importance shift
top_traj = traj_compare.head(15).iloc[::-1]
plt.figure(figsize=(9, 6))
plt.barh(top_traj.index, top_traj["2021"], alpha=0.7, label="2021")
plt.barh(top_traj.index, top_traj["2025"], alpha=0.7, label="2025")
plt.legend()
plt.xlabel("Mean |SHAP|")
plt.title("2021/22 → 2025 Feature Importance Shift")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "trajectory_shift.png", dpi=150)
plt.show()
print("Saved: trajectory_shift.png, trajectory_compare.csv")
