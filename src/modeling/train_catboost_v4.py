import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import json
import os
from pathlib import Path

# Configuration
RANDOM_STATE = 719
N_TRIALS = 20  # Increased for PC run
param_storage = "sqlite:///optuna_v4.db" # Persistent storage for long runs
study_name = "catboost_v4_optimization"

# Load Data
print("Loading V4 Data...")
# Load Data
print("Loading V4 Data...")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
train_path = PROJECT_ROOT / "data/processed/kkbox_train_feature_v4.parquet"

if not train_path.exists():
    raise FileNotFoundError(f"kkbox_train_feature_v4.parquet not found at {train_path}. Please ensure data is present.")

df = pd.read_parquet(train_path)
print(f"Data Shape: {df.shape}")

# Features
CATEGORICAL_COLS = [
    "city", "gender", "registered_via", "last_payment_method",
    "has_ever_paid", "has_ever_cancelled",
    "is_auto_renew_last",
    "is_free_user",
]

NUMERICAL_COLS = [
    "reg_days",
    # w7
    "num_days_active_w7", "total_secs_w7", "avg_secs_per_day_w7", "std_secs_w7",
    "num_songs_w7", "avg_songs_per_day_w7", "num_unq_w7", "num_25_w7", "num_100_w7",
    "short_play_w7", "skip_ratio_w7", "completion_ratio_w7", "short_play_ratio_w7", "variety_ratio_w7",
    # w14
    "num_days_active_w14", "total_secs_w14", "avg_secs_per_day_w14", "std_secs_w14",
    "num_songs_w14", "avg_songs_per_day_w14", "num_unq_w14", "num_25_w14", "num_100_w14",
    "short_play_w14", "skip_ratio_w14", "completion_ratio_w14", "short_play_ratio_w14", "variety_ratio_w14",
    # w21
    "num_days_active_w21", "total_secs_w21", "avg_secs_per_day_w21", "std_secs_w21",
    "num_songs_w21", "avg_songs_per_day_w21", "num_unq_w21", "num_25_w21", "num_100_w21",
    "short_play_w21", "skip_ratio_w21", "completion_ratio_w21", "short_play_ratio_w21", "variety_ratio_w21",
    # trends
    "days_trend_w7_w14", "secs_trend_w7_w30", "secs_trend_w14_w30",
    "days_trend_w7_w30", "songs_trend_w7_w30", "songs_trend_w14_w30",
    "skip_trend_w7_w30", "completion_trend_w7_w30",
    # transactions
    "days_since_last_payment", "days_since_last_cancel", "last_plan_days",
    "total_payment_count", "total_amount_paid", "avg_amount_per_payment",
    "unique_plan_count", "subscription_months_est",
    "payment_count_last_30d", "payment_count_last_90d",
    # V4 Derived
    "active_decay_rate", "listening_time_velocity", "discovery_index", 
    "skip_passion_index", "last_active_gap"
]

cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
FEATURE_COLS = cat_cols + num_cols
TARGET_COL = "is_churn"

# Preprocessing (Categorical to string)
for col in cat_cols:
    df[col] = df[col].astype(str).astype("category")

X = df[FEATURE_COLS]
y = df[TARGET_COL].astype(int)

# Split
print("Splitting Data (Train/Valid/Test)...")
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_full
)

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 1000, 3000), 
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "auto_class_weights": "Balanced",
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "early_stopping_rounds": 100,
        "task_type": "CPU"  # Change to GPU if available on PC
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_valid, y_valid), verbose=False)
    
    # Optimization Metric: F1 on Valid
    y_pred = model.predict(X_valid)
    return f1_score(y_valid, y_pred)

print(f"Starting Optuna Optimization ({N_TRIALS} trials)...")
study = optuna.create_study(direction="maximize", study_name=study_name, storage=None, load_if_exists=False)
study.optimize(objective, n_trials=N_TRIALS)

print("Best Parameters:")
print(study.best_params)

# Train Final Model
print("Training Final Model on Train+Valid...")
best_params = study.best_params.copy()
best_params.update({
    "loss_function": "Logloss",
    "eval_metric": "F1",
    "auto_class_weights": "Balanced",
    "random_seed": RANDOM_STATE,
    "verbose": 100,
    "early_stopping_rounds": 100,
    "task_type": "CPU" # User can change this to GPU
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

# Save
# Save
MODEL_DIR = PROJECT_ROOT / "03_trained_model"
if not MODEL_DIR.exists():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

final_model.save_model(os.path.join(MODEL_DIR, "catboost_model_v4.cbm"))

# Save Metadata
metadata = {
    "best_params": best_params,
    "best_f1_score": float(study.best_value),
    "feature_names": FEATURE_COLS,
    "categorical_features": cat_cols,
    "numerical_features": num_cols
}
with open(os.path.join(MODEL_DIR, "model_metadata_v4.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("Model and Metadata Saved.")
