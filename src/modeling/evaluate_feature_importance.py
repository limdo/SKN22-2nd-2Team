
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Config
RANDOM_STATE = 719
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/kkbox_train_feature_v4.parquet" # Using V4 baseline data
MODEL_DIR = PROJECT_ROOT / "03_trained_model"
MODEL_PATH = MODEL_DIR / "catboost_model_v5.2.cbm"
DATA_TUNED_DIR = PROJECT_ROOT / "data/tuned"
IMAGES_TUNED_DIR = PROJECT_ROOT / "images/tuned"

# Ensure directories exist
DATA_TUNED_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_TUNED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATA_TUNED_DIR / "feature_importance_v5.2_permutation.csv"
PLOT_PATH = IMAGES_TUNED_DIR / "feature_importance_v5.2_plot.png"

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    return df

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    return model

def main():
    # 1. Load Model First to get Feature Names
    model = load_model()
    FEATURE_COLS = model.feature_names_
    print(f"Loaded {len(FEATURE_COLS)} features from model.")
    TARGET_COL = "is_churn"
    
    # 2. Load Data
    df = load_data()
    
    # Preprocessing (Categorical to string)
    # We still need to know which are categorical to cast them. 
    # Use the intersection of known categorical cols and model features.
    KNOWN_CATEGORICALS = [
        "city", "gender", "registered_via", "last_payment_method",
        "has_ever_paid", "has_ever_cancelled",
        "is_auto_renew_last",
        "is_free_user",
    ]
    model_cat_features = [c for c in FEATURE_COLS if c in KNOWN_CATEGORICALS]
    
    for col in model_cat_features:
        if col in df.columns:
            df[col] = df[col].astype(str).astype("category")

    # Ensure df has all model features
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Data is missing features required by model: {missing_cols}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int)
    
    # 3. Split (Using same Seed)
    print("Splitting data (recreating Test set)...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )


    
    # 4. Built-in Feature Importance
    print("Calculating Built-in Feature Importance...")
    importance = model.get_feature_importance(type="FeatureImportance")
    fi_df = pd.DataFrame({'feature': FEATURE_COLS, 'importance': importance})
    fi_df = fi_df.sort_values(by='importance', ascending=False)
    
    print("Top 10 Built-in Features:")
    print(fi_df.head(10))
    fi_df.to_csv(DATA_TUNED_DIR / "feature_importance_v5.2_builtin.csv", index=False)

    # 5. Permutation Importance
    print("\nCalculating Permutation Importance (this may take a while)...")
    # Using a subset of test set if it's too large to speed up interactive response
    if len(X_test) > 5000:
        print(f"Subsampling test set from {len(X_test)} to 5000 for permutation importance...")
        X_test_perm = X_test.sample(5000, random_state=RANDOM_STATE)
        y_test_perm = y_test.loc[X_test_perm.index]
    else:
        X_test_perm = X_test
        y_test_perm = y_test
        
    perm_result = permutation_importance(
        model, X_test_perm, y_test_perm, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1, scoring='f1'
    )
    
    perm_df = pd.DataFrame({
        'feature': FEATURE_COLS, 
        'importance_mean': perm_result.importances_mean,
        'importance_std': perm_result.importances_std
    })
    perm_df = perm_df.sort_values(by='importance_mean', ascending=False)
    
    print("Top 10 Permutation Features:")
    print(perm_df.head(10))
    
    perm_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Permutation importance saved to {OUTPUT_PATH}")
    
    # 6. Plotting
    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance_mean", y="feature", data=perm_df.head(20))
    plt.title("Top 20 Features (Permutation Importance)")
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Plot saved to {PLOT_PATH}")

if __name__ == "__main__":
    main()
