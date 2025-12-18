# -*- coding: utf-8 -*-
"""
KKBox Churn Prediction - Ensemble Model Training
작성자: 이도훈 (LDH)
작성일: 2025-12-18

이 모듈은 LightGBM과 CatBoost를 결합한 앙상블 모델을 학습합니다.

앙상블 방법:
1. Soft Voting (확률 평균)
2. Weighted Average (가중 평균)
3. Stacking (Logistic Regression 메타러너)

평가 지표:
- ROC-AUC
- PR-AUC (Average Precision)
- Recall / Precision / F1-Score
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
import warnings
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib

warnings.filterwarnings("ignore")

# ============================================
# 설정
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "docs" / "02_training_report"

RANDOM_STATE = 719


# ============================================
# 평가 함수
# ============================================
def evaluate_model(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """
    모델 성능을 평가합니다.
    """
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "recall": float(recall_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_negative"] = int(tn)
    metrics["false_positive"] = int(fp)
    metrics["false_negative"] = int(fn)
    metrics["true_positive"] = int(tp)
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return metrics


def print_metrics(metrics: Dict[str, float], name: str) -> None:
    """평가 지표를 출력합니다."""
    print(f"\n[{name}] 평가 결과:")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  F1-Score:   {metrics['f1']:.4f}")
    print(f"  Specificity:{metrics['specificity']:.4f}")


# ============================================
# 데이터 로드
# ============================================
def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """전처리된 train/valid/test 데이터를 로드합니다."""
    print("📂 데이터 로드 중...")

    train = pd.read_csv(DATA_DIR / "train_set.csv")
    valid = pd.read_csv(DATA_DIR / "valid_set.csv")
    test = pd.read_csv(DATA_DIR / "test_set.csv")

    print(f"  ✓ Train: {train.shape}")
    print(f"  ✓ Valid: {valid.shape}")
    print(f"  ✓ Test:  {test.shape}")

    return train, valid, test


def prepare_features(
    train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame
) -> Tuple:
    """학습에 사용할 피처와 타겟을 분리합니다."""
    exclude_cols = ["msno", "is_churn"]
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X_train = train[feature_cols]
    y_train = train["is_churn"]

    X_valid = valid[feature_cols]
    y_valid = valid["is_churn"]

    X_test = test[feature_cols]
    y_test = test["is_churn"]

    print(f"\n📊 피처 준비 완료")
    print(f"  피처 수: {len(feature_cols)}")
    print(f"  Train Churn 비율: {y_train.mean()*100:.2f}%")

    return X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols


# ============================================
# 개별 모델 학습
# ============================================
def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> Tuple[Any, Dict[str, Any]]:
    """LightGBM 모델을 학습합니다."""
    print("\n" + "=" * 60)
    print("🌲 LightGBM 학습")
    print("=" * 60)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_pos_weight,
        "min_child_samples": 100,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": RANDOM_STATE,
        "verbose": -1,
        "n_jobs": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    print("  학습 중...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )

    y_valid_prob = model.predict(X_valid, num_iteration=model.best_iteration)
    valid_metrics = evaluate_model(
        y_valid, (y_valid_prob >= 0.5).astype(int), y_valid_prob
    )
    print_metrics(valid_metrics, "LightGBM Valid")

    # Feature Importance
    importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)

    results = {
        "model_name": "LightGBM",
        "valid_metrics": valid_metrics,
        "params": params,
        "best_iteration": model.best_iteration,
        "feature_importance": importance.head(10).to_dict("records"),
    }

    return model, results


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> Tuple[Any, Dict[str, Any]]:
    """CatBoost 모델을 학습합니다."""
    print("\n" + "=" * 60)
    print("🐱 CatBoost 학습")
    print("=" * 60)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": RANDOM_STATE,
        "iterations": 500,
        "early_stopping_rounds": 50,
        "thread_count": -1,
        "scale_pos_weight": float(scale_pos_weight),
        "verbose": 100,
    }

    model = CatBoostClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )

    y_valid_prob = model.predict_proba(X_valid)[:, 1]
    valid_metrics = evaluate_model(
        y_valid, (y_valid_prob >= 0.5).astype(int), y_valid_prob
    )
    print_metrics(valid_metrics, "CatBoost Valid")

    # Feature Importance
    importances = model.get_feature_importance()
    importance_df = pd.DataFrame(
        {"feature": list(X_train.columns), "importance": importances}
    ).sort_values("importance", ascending=False)

    results = {
        "model_name": "CatBoost",
        "valid_metrics": valid_metrics,
        "params": {k: str(v) if not isinstance(v, (int, float, str, bool)) else v 
                   for k, v in params.items()},
        "best_iteration": getattr(model, "best_iteration_", None),
        "feature_importance": importance_df.head(10).to_dict("records"),
    }

    return model, results


# ============================================
# 앙상블 모델
# ============================================
def ensemble_soft_voting(
    probs_list: List[np.ndarray],
) -> np.ndarray:
    """
    Soft Voting: 확률 평균
    """
    return np.mean(probs_list, axis=0)


def ensemble_weighted_average(
    probs_list: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """
    Weighted Average: 가중 평균
    """
    weights = np.array(weights)
    weights = weights / weights.sum()  # 정규화
    return np.average(probs_list, axis=0, weights=weights)


def train_stacking_meta_learner(
    X_meta_train: np.ndarray,
    y_train: np.ndarray,
    X_meta_valid: np.ndarray,
    y_valid: np.ndarray,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Stacking: Logistic Regression 메타러너 학습
    """
    print("\n" + "=" * 60)
    print("🎯 Stacking (Meta Learner: Logistic Regression)")
    print("=" * 60)

    meta_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    meta_model.fit(X_meta_train, y_train)

    y_valid_prob = meta_model.predict_proba(X_meta_valid)[:, 1]
    valid_metrics = evaluate_model(
        y_valid, (y_valid_prob >= 0.5).astype(int), y_valid_prob
    )
    print_metrics(valid_metrics, "Stacking Valid")

    results = {
        "model_name": "Stacking (LR Meta)",
        "valid_metrics": valid_metrics,
        "params": {"C": 1.0, "class_weight": "balanced"},
        "coef": meta_model.coef_.tolist(),
    }

    return meta_model, results


def evaluate_ensemble_methods(
    lgb_model: Any,
    cb_model: Any,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    다양한 앙상블 방법을 평가합니다.
    """
    print("\n" + "=" * 60)
    print("🔗 앙상블 모델 평가")
    print("=" * 60)

    # 개별 모델 예측
    lgb_valid_prob = lgb_model.predict(X_valid, num_iteration=lgb_model.best_iteration)
    cb_valid_prob = cb_model.predict_proba(X_valid)[:, 1]

    lgb_test_prob = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    cb_test_prob = cb_model.predict_proba(X_test)[:, 1]

    ensemble_results = {}

    # 1. Soft Voting (단순 평균)
    print("\n--- Soft Voting (Simple Average) ---")
    soft_valid = ensemble_soft_voting([lgb_valid_prob, cb_valid_prob])
    soft_test = ensemble_soft_voting([lgb_test_prob, cb_test_prob])

    soft_valid_metrics = evaluate_model(
        y_valid, (soft_valid >= 0.5).astype(int), soft_valid
    )
    soft_test_metrics = evaluate_model(
        y_test, (soft_test >= 0.5).astype(int), soft_test
    )
    print_metrics(soft_valid_metrics, "Soft Voting Valid")
    print_metrics(soft_test_metrics, "Soft Voting Test")

    ensemble_results["Soft Voting"] = {
        "model_name": "Soft Voting (Average)",
        "method": "Simple average of LightGBM and CatBoost probabilities",
        "weights": [0.5, 0.5],
        "valid_metrics": soft_valid_metrics,
        "test_metrics": soft_test_metrics,
    }

    # 2. Weighted Average (ROC-AUC 기반 가중치)
    print("\n--- Weighted Average (AUC-based weights) ---")
    lgb_auc = roc_auc_score(y_valid, lgb_valid_prob)
    cb_auc = roc_auc_score(y_valid, cb_valid_prob)
    weights = [lgb_auc, cb_auc]
    print(f"  Weights: LightGBM={lgb_auc:.4f}, CatBoost={cb_auc:.4f}")

    weighted_valid = ensemble_weighted_average([lgb_valid_prob, cb_valid_prob], weights)
    weighted_test = ensemble_weighted_average([lgb_test_prob, cb_test_prob], weights)

    weighted_valid_metrics = evaluate_model(
        y_valid, (weighted_valid >= 0.5).astype(int), weighted_valid
    )
    weighted_test_metrics = evaluate_model(
        y_test, (weighted_test >= 0.5).astype(int), weighted_test
    )
    print_metrics(weighted_valid_metrics, "Weighted Avg Valid")
    print_metrics(weighted_test_metrics, "Weighted Avg Test")

    ensemble_results["Weighted Average"] = {
        "model_name": "Weighted Average",
        "method": "AUC-based weighted average",
        "weights": [float(w) for w in weights],
        "valid_metrics": weighted_valid_metrics,
        "test_metrics": weighted_test_metrics,
    }

    # 3. Stacking (Logistic Regression 메타러너)
    print("\n--- Stacking (Logistic Regression Meta-Learner) ---")
    X_meta_valid = np.column_stack([lgb_valid_prob, cb_valid_prob])
    X_meta_test = np.column_stack([lgb_test_prob, cb_test_prob])

    stacking_model, stacking_results = train_stacking_meta_learner(
        X_meta_valid, y_valid.values, X_meta_test, y_test.values
    )

    # 스태킹은 valid로 학습했으므로 test 평가만 의미 있음
    stacking_test_prob = stacking_model.predict_proba(X_meta_test)[:, 1]
    stacking_test_metrics = evaluate_model(
        y_test, (stacking_test_prob >= 0.5).astype(int), stacking_test_prob
    )
    print_metrics(stacking_test_metrics, "Stacking Test")

    ensemble_results["Stacking"] = {
        "model_name": "Stacking (LR Meta)",
        "method": "Logistic Regression meta-learner on base model predictions",
        "meta_coef": stacking_results["coef"],
        "valid_metrics": stacking_results["valid_metrics"],
        "test_metrics": stacking_test_metrics,
    }

    return ensemble_results


# ============================================
# 마크다운 리포트 생성
# ============================================
def generate_ensemble_report(
    lgb_results: Dict[str, Any],
    cb_results: Dict[str, Any],
    ensemble_results: Dict[str, Any],
    lgb_test_metrics: Dict[str, float],
    cb_test_metrics: Dict[str, float],
    report_dir: Path,
) -> None:
    """앙상블 모델 결과를 마크다운으로 저장합니다."""
    report_dir.mkdir(parents=True, exist_ok=True)

    # 최고 성능 앙상블 찾기
    best_ensemble = max(
        ensemble_results.items(), key=lambda x: x[1]["test_metrics"]["roc_auc"]
    )
    best_name = best_ensemble[0]
    best_metrics = best_ensemble[1]["test_metrics"]

    # 개별 모델 vs 앙상블 비교
    all_test_metrics = {
        "LightGBM": lgb_test_metrics,
        "CatBoost": cb_test_metrics,
        **{k: v["test_metrics"] for k, v in ensemble_results.items()},
    }

    overall_best = max(all_test_metrics.items(), key=lambda x: x[1]["roc_auc"])

    md = f"""# 03. 앙상블 모델 학습 결과 (Ensemble Training Results)

> **작성자**: 이도훈 (LDH)  
> **작성일**: {datetime.now().strftime("%Y-%m-%d")}  
> **버전**: v1.0

---

## 1. 학습 개요

### 1.1 앙상블 구성
| 구성 요소 | 설명 |
|----------|------|
| **Base Model 1** | LightGBM (GBDT) |
| **Base Model 2** | CatBoost (Ordered Boosting) |
| **앙상블 방법** | Soft Voting, Weighted Average, Stacking |

### 1.2 앙상블 방법 설명
| 방법 | 설명 |
|------|------|
| **Soft Voting** | 두 모델의 예측 확률을 단순 평균 |
| **Weighted Average** | Validation AUC 기반 가중 평균 |
| **Stacking** | Logistic Regression 메타러너로 예측 결합 |

---

## 2. 개별 모델 성능 (Base Models)

### 2.1 Validation Set 성능

| 지표 | LightGBM | CatBoost | 우수 모델 |
|------|----------|----------|-----------|
| **ROC-AUC** | {lgb_results['valid_metrics']['roc_auc']:.4f} | {cb_results['valid_metrics']['roc_auc']:.4f} | {'LightGBM ✅' if lgb_results['valid_metrics']['roc_auc'] > cb_results['valid_metrics']['roc_auc'] else 'CatBoost ✅'} |
| **PR-AUC** | {lgb_results['valid_metrics']['pr_auc']:.4f} | {cb_results['valid_metrics']['pr_auc']:.4f} | {'LightGBM ✅' if lgb_results['valid_metrics']['pr_auc'] > cb_results['valid_metrics']['pr_auc'] else 'CatBoost ✅'} |
| **Recall** | {lgb_results['valid_metrics']['recall']:.4f} | {cb_results['valid_metrics']['recall']:.4f} | {'LightGBM ✅' if lgb_results['valid_metrics']['recall'] > cb_results['valid_metrics']['recall'] else 'CatBoost ✅'} |
| **Precision** | {lgb_results['valid_metrics']['precision']:.4f} | {cb_results['valid_metrics']['precision']:.4f} | {'LightGBM ✅' if lgb_results['valid_metrics']['precision'] > cb_results['valid_metrics']['precision'] else 'CatBoost ✅'} |
| **F1-Score** | {lgb_results['valid_metrics']['f1']:.4f} | {cb_results['valid_metrics']['f1']:.4f} | {'LightGBM ✅' if lgb_results['valid_metrics']['f1'] > cb_results['valid_metrics']['f1'] else 'CatBoost ✅'} |

### 2.2 Test Set 성능

| 지표 | LightGBM | CatBoost | 우수 모델 |
|------|----------|----------|-----------|
| **ROC-AUC** | {lgb_test_metrics['roc_auc']:.4f} | {cb_test_metrics['roc_auc']:.4f} | {'LightGBM ✅' if lgb_test_metrics['roc_auc'] > cb_test_metrics['roc_auc'] else 'CatBoost ✅'} |
| **PR-AUC** | {lgb_test_metrics['pr_auc']:.4f} | {cb_test_metrics['pr_auc']:.4f} | {'LightGBM ✅' if lgb_test_metrics['pr_auc'] > cb_test_metrics['pr_auc'] else 'CatBoost ✅'} |
| **Recall** | {lgb_test_metrics['recall']:.4f} | {cb_test_metrics['recall']:.4f} | {'LightGBM ✅' if lgb_test_metrics['recall'] > cb_test_metrics['recall'] else 'CatBoost ✅'} |
| **Precision** | {lgb_test_metrics['precision']:.4f} | {cb_test_metrics['precision']:.4f} | {'LightGBM ✅' if lgb_test_metrics['precision'] > cb_test_metrics['precision'] else 'CatBoost ✅'} |
| **F1-Score** | {lgb_test_metrics['f1']:.4f} | {cb_test_metrics['f1']:.4f} | {'LightGBM ✅' if lgb_test_metrics['f1'] > cb_test_metrics['f1'] else 'CatBoost ✅'} |

---

## 3. 앙상블 모델 성능 (Test Set)

### 3.1 앙상블 방법별 성능 비교

| 지표 | Soft Voting | Weighted Avg | Stacking | 최고 성능 |
|------|-------------|--------------|----------|-----------|
| **ROC-AUC** | {ensemble_results['Soft Voting']['test_metrics']['roc_auc']:.4f} | {ensemble_results['Weighted Average']['test_metrics']['roc_auc']:.4f} | {ensemble_results['Stacking']['test_metrics']['roc_auc']:.4f} | {best_name} ✅ |
| **PR-AUC** | {ensemble_results['Soft Voting']['test_metrics']['pr_auc']:.4f} | {ensemble_results['Weighted Average']['test_metrics']['pr_auc']:.4f} | {ensemble_results['Stacking']['test_metrics']['pr_auc']:.4f} | - |
| **Recall** | {ensemble_results['Soft Voting']['test_metrics']['recall']:.4f} | {ensemble_results['Weighted Average']['test_metrics']['recall']:.4f} | {ensemble_results['Stacking']['test_metrics']['recall']:.4f} | - |
| **Precision** | {ensemble_results['Soft Voting']['test_metrics']['precision']:.4f} | {ensemble_results['Weighted Average']['test_metrics']['precision']:.4f} | {ensemble_results['Stacking']['test_metrics']['precision']:.4f} | - |
| **F1-Score** | {ensemble_results['Soft Voting']['test_metrics']['f1']:.4f} | {ensemble_results['Weighted Average']['test_metrics']['f1']:.4f} | {ensemble_results['Stacking']['test_metrics']['f1']:.4f} | - |

### 3.2 앙상블 가중치 정보

| 앙상블 방법 | LightGBM 가중치 | CatBoost 가중치 |
|------------|-----------------|-----------------|
| Soft Voting | 0.5000 | 0.5000 |
| Weighted Average | {ensemble_results['Weighted Average']['weights'][0]:.4f} | {ensemble_results['Weighted Average']['weights'][1]:.4f} |
| Stacking (Meta Coef) | {ensemble_results['Stacking']['meta_coef'][0][0]:.4f} | {ensemble_results['Stacking']['meta_coef'][0][1]:.4f} |

---

## 4. 전체 모델 성능 비교 (Test Set)

| 순위 | 모델 | ROC-AUC | PR-AUC | Recall | Precision | F1-Score |
|------|------|---------|--------|--------|-----------|----------|
"""

    # 순위별 정렬
    sorted_models = sorted(
        all_test_metrics.items(), key=lambda x: x[1]["roc_auc"], reverse=True
    )
    for rank, (name, metrics) in enumerate(sorted_models, 1):
        best_marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
        md += f"| {rank} {best_marker} | **{name}** | {metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} | {metrics['recall']:.4f} | {metrics['precision']:.4f} | {metrics['f1']:.4f} |\n"

    md += f"""
---

## 5. Confusion Matrix (Test Set)

### 5.1 LightGBM

```
              Predicted
              0        1
Actual  0    {lgb_test_metrics['true_negative']:,}    {lgb_test_metrics['false_positive']:,}
        1    {lgb_test_metrics['false_negative']:,}    {lgb_test_metrics['true_positive']:,}
```

### 5.2 CatBoost

```
              Predicted
              0        1
Actual  0    {cb_test_metrics['true_negative']:,}    {cb_test_metrics['false_positive']:,}
        1    {cb_test_metrics['false_negative']:,}    {cb_test_metrics['true_positive']:,}
```

### 5.3 {best_name} (Best Ensemble)

```
              Predicted
              0        1
Actual  0    {best_metrics['true_negative']:,}    {best_metrics['false_positive']:,}
        1    {best_metrics['false_negative']:,}    {best_metrics['true_positive']:,}
```

---

## 6. 결론

### 6.1 최종 모델 선정

| 항목 | 값 |
|------|-----|
| **최고 성능 모델** | {overall_best[0]} |
| **ROC-AUC** | {overall_best[1]['roc_auc']:.4f} |
| **PR-AUC** | {overall_best[1]['pr_auc']:.4f} |
| **Recall** | {overall_best[1]['recall']:.4f} |

### 6.2 앙상블 효과 분석

| 비교 항목 | 값 |
|----------|-----|
| LightGBM (단독) ROC-AUC | {lgb_test_metrics['roc_auc']:.4f} |
| CatBoost (단독) ROC-AUC | {cb_test_metrics['roc_auc']:.4f} |
| 최고 앙상블 ({best_name}) ROC-AUC | {best_metrics['roc_auc']:.4f} |
| **앙상블 향상도** | {'+' if best_metrics['roc_auc'] > max(lgb_test_metrics['roc_auc'], cb_test_metrics['roc_auc']) else ''}{(best_metrics['roc_auc'] - max(lgb_test_metrics['roc_auc'], cb_test_metrics['roc_auc']))*100:.2f}%p |

### 6.3 권장 사항

"""

    if best_metrics['roc_auc'] > max(lgb_test_metrics['roc_auc'], cb_test_metrics['roc_auc']):
        md += f"""- ✅ **앙상블 모델 사용 권장**: {best_name}이 단일 모델보다 우수한 성능을 보임
- 앙상블을 통해 ROC-AUC가 {(best_metrics['roc_auc'] - max(lgb_test_metrics['roc_auc'], cb_test_metrics['roc_auc']))*100:.2f}%p 향상됨
"""
    else:
        single_best = "LightGBM" if lgb_test_metrics['roc_auc'] > cb_test_metrics['roc_auc'] else "CatBoost"
        md += f"""- ⚠️ **단일 모델 사용 권장**: {single_best}이 앙상블보다 우수하거나 동등한 성능
- 앙상블의 추가적인 복잡성 대비 성능 향상이 미미함
"""

    md += f"""
---

## 7. 저장된 파일

| 파일 | 경로 | 설명 |
|------|------|------|
| LightGBM | `models/ensemble_lightgbm.txt` | Base Model 1 |
| CatBoost | `models/ensemble_catboost.cbm` | Base Model 2 |
| 앙상블 결과 | `models/ensemble_results.json` | 전체 결과 |
| 리포트 | `docs/02_training_report/03_ensemble_training_results.md` | 이 문서 |

---

> **📌 다음 단계**: Risk Score 기반 비즈니스 액션 매핑 또는 SHAP 분석
"""

    report_path = report_dir / "03_ensemble_training_results.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n📄 리포트 저장: {report_path}")


# ============================================
# 메인 파이프라인
# ============================================
def run_ensemble_training() -> Dict[str, Any]:
    """전체 앙상블 학습 파이프라인을 실행합니다."""
    print("=" * 60)
    print("🚀 KKBox Churn Prediction - Ensemble Training Pipeline")
    print("=" * 60)

    # 1. 데이터 로드
    train, valid, test = load_datasets()

    # 2. 피처 준비
    X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols = prepare_features(
        train, valid, test
    )

    # 3. LightGBM 학습
    lgb_model, lgb_results = train_lightgbm(X_train, y_train, X_valid, y_valid)

    # 4. CatBoost 학습
    cb_model, cb_results = train_catboost(X_train, y_train, X_valid, y_valid)

    # 5. 개별 모델 Test 평가
    lgb_test_prob = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    cb_test_prob = cb_model.predict_proba(X_test)[:, 1]

    lgb_test_metrics = evaluate_model(
        y_test, (lgb_test_prob >= 0.5).astype(int), lgb_test_prob
    )
    cb_test_metrics = evaluate_model(
        y_test, (cb_test_prob >= 0.5).astype(int), cb_test_prob
    )

    print("\n" + "=" * 60)
    print("📊 개별 모델 Test Set 성능")
    print("=" * 60)
    print_metrics(lgb_test_metrics, "LightGBM Test")
    print_metrics(cb_test_metrics, "CatBoost Test")

    lgb_results["test_metrics"] = lgb_test_metrics
    cb_results["test_metrics"] = cb_test_metrics

    # 6. 앙상블 모델 평가
    ensemble_results = evaluate_ensemble_methods(
        lgb_model, cb_model, X_valid, y_valid, X_test, y_test
    )

    # 7. 모델 저장
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    lgb_model.save_model(str(MODEL_DIR / "ensemble_lightgbm.txt"))
    cb_model.save_model(str(MODEL_DIR / "ensemble_catboost.cbm"))

    all_results = {
        "LightGBM": lgb_results,
        "CatBoost": cb_results,
        "Ensemble": ensemble_results,
        "meta": {
            "n_samples": len(train),
            "n_features": len(feature_cols),
            "created_at": datetime.now().isoformat(),
        },
    }

    with open(MODEL_DIR / "ensemble_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n✓ 모델 저장 완료: {MODEL_DIR}")

    # 8. 리포트 생성
    generate_ensemble_report(
        lgb_results,
        cb_results,
        ensemble_results,
        lgb_test_metrics,
        cb_test_metrics,
        REPORT_DIR,
    )

    print("\n" + "=" * 60)
    print("✅ 앙상블 학습 파이프라인 완료!")
    print("=" * 60)

    return all_results


# ============================================
# 실행
# ============================================
if __name__ == "__main__":
    results = run_ensemble_training()

