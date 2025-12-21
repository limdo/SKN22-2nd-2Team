
import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, roc_auc_score
import json
import os
from pathlib import Path

# Configuration
RANDOM_STATE = 719
N_TRIALS = 20
param_storage = "sqlite:///optuna_v5_2.db"
study_name = "catboost_v5_2_optimization"

# 1. Load Data (Using V4 Baseline Data)
print("Loading V4 Baseline Data...")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
train_path = PROJECT_ROOT / "data/processed/kkbox_train_feature_v4.parquet"

if not train_path.exists():
    raise FileNotFoundError(f"Data not found at {train_path}")

df = pd.read_parquet(train_path)

# 2. Define Feature Sets
# Full Candidate List from V4
CATEGORICAL_COLS = [
    "city", "gender", "registered_via", "last_payment_method",
    "has_ever_paid", "has_ever_cancelled",
    "is_auto_renew_last",
    "is_free_user",
]

# Features to DROP (UNSAFE Transaction/Status Keys)
# Strategy: Keep "Context/History" (Safe), Drop "Status/Leakage" (Unsafe)
# Safe (Kept): last_payment_method, avg_amount_per_payment, total_payment_count, 
#              total_amount_paid, unique_plan_count, subscription_months_est, 
#              has_ever_cancelled, has_ever_paid
DROP_UNSAFE_COLS = [
    "days_since_last_payment",  # Direct leakage of churn status
    "days_since_last_cancel",   # Direct leakage of churn intent
    "is_auto_renew_last",       # Status cheat key
    "last_plan_days",           # Often mirrors payment timing
    "payment_count_last_30d",   # Recent status indicator (0 vs 1)
    "payment_count_last_90d",   # Recent status indicator
]

# Filter Features
ALL_FEATURES = [c for c in df.columns if c not in ["msno", "is_churn", "date", "last_active_date"]]
SELECTED_FEATURES = [c for c in ALL_FEATURES if c not in DROP_UNSAFE_COLS]

print(f"Total Features available: {len(ALL_FEATURES)}")
print(f"Unsafe Features excluded: {len(DROP_UNSAFE_COLS)}")
print(f"Final Features for V5.2 Model: {len(SELECTED_FEATURES)}")
print("-" * 30)
print(SELECTED_FEATURES)
print("-" * 30)

cat_cols = [c for c in CATEGORICAL_COLS if c in SELECTED_FEATURES]
FEATURE_COLS = SELECTED_FEATURES
TARGET_COL = "is_churn"

# Preprocessing
for col in cat_cols:
    df[col] = df[col].astype(str).astype("category")

X = df[FEATURE_COLS]
y = df[TARGET_COL].astype(int)

# Split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_full
)

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 1000, 2500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10, log=True),
        "auto_class_weights": "Balanced",
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "early_stopping_rounds": 50,
        "task_type": "CPU"
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_valid, y_valid), verbose=False)
    
    y_pred = model.predict(X_valid)
    return f1_score(y_valid, y_pred)

print(f"Starting Optuna Optimization ({N_TRIALS} trials)...")
study = optuna.create_study(direction="maximize", study_name=study_name, storage=None, load_if_exists=False)
study.optimize(objective, n_trials=N_TRIALS)

print("Best Parameters:")
print(study.best_params)

# Train Final Model
print("Training Final V5.2 Model...")
best_params = study.best_params.copy()
best_params.update({
    "loss_function": "Logloss",
    "eval_metric": "F1",
    "auto_class_weights": "Balanced",
    "random_seed": RANDOM_STATE,
    "verbose": 100,
    "early_stopping_rounds": 100,
    "task_type": "CPU"
})

final_model = CatBoostClassifier(**best_params)
final_model.fit(
    X_train_full, y_train_full,
    cat_features=cat_cols,
    eval_set=(X_test, y_test)
)

print("Final Evaluation on Test Set...")
y_pred_test = final_model.predict(X_test)
print(classification_report(y_test, y_pred_test))

# Verify Feature Importance
importance = final_model.get_feature_importance(type="FeatureImportance")
fi_df = pd.DataFrame({'feature': FEATURE_COLS, 'importance': importance}).sort_values(by='importance', ascending=False)
print("Top 10 Features in V5.2 Model:")
print(fi_df.head(10))

# Save
MODEL_DIR = PROJECT_ROOT / "03_trained_model"
if not MODEL_DIR.exists():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

final_model.save_model(os.path.join(MODEL_DIR, "catboost_model_v5.2.cbm"))

metadata = {
    "model_version": "v5.2",
    "description": "Safe Transactions Model (Context retained, Status dropped)",
    "best_params": best_params,
    "best_f1_score": float(study.best_value),
    "feature_names": FEATURE_COLS,
    "categorical_features": cat_cols,
    "dropped_features": DROP_UNSAFE_COLS
}
with open(os.path.join(MODEL_DIR, "model_metadata_v5.2.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("V5.2 Model Saved.")
