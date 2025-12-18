# -*- coding: utf-8 -*-
"""
KKBox Churn Prediction - Recall 최적화 모델 학습
작성자: 이도훈 (LDH)
작성일: 2025-12-18

목표: Precision보다 Recall이 높은 모델 선택
- CatBoost vs LightGBM 비교
- 중복 피처 제거 (recency_*_ratio = *_trend_w7_w30)
- 피처 수는 유지
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
import warnings
from datetime import datetime

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ============================================
# 설정
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "docs" / "02_training_report"

RANDOM_STATE = 719

# 중복 피처 (동일한 값) - 제거 대상
DUPLICATE_FEATURES = [
    "recency_secs_ratio",   # = secs_trend_w7_w30
    "recency_songs_ratio",  # = songs_trend_w7_w30
]


# ============================================
# 평가 함수
# ============================================
def evaluate_model(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """모델 성능 평가"""
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
    """평가 지표 출력"""
    print(f"\n[{name}]")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  F1-Score:   {metrics['f1']:.4f}")


def find_optimal_threshold_for_recall(
    y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.95
) -> Tuple[float, Dict[str, float]]:
    """
    목표 Recall을 달성하는 최적 Threshold 찾기
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # target_recall 이상인 threshold 중 가장 높은 것 선택
    valid_idx = np.where(recalls >= target_recall)[0]
    if len(valid_idx) == 0:
        # target_recall을 달성할 수 없으면 최대 recall threshold 사용
        best_idx = np.argmax(recalls)
    else:
        # target_recall 이상 중 가장 높은 precision
        best_idx = valid_idx[np.argmax(precisions[valid_idx])]
    
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1
    
    best_threshold = thresholds[best_idx]
    
    # 해당 threshold에서의 metrics
    y_pred = (y_prob >= best_threshold).astype(int)
    metrics = evaluate_model(y_true, y_pred, y_prob)
    
    return best_threshold, metrics


# ============================================
# 데이터 로드
# ============================================
def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """데이터 로드"""
    print("=" * 60)
    print("데이터 로드")
    print("=" * 60)

    train = pd.read_csv(DATA_DIR / "train_set.csv")
    valid = pd.read_csv(DATA_DIR / "valid_set.csv")
    test = pd.read_csv(DATA_DIR / "test_set.csv")

    print(f"  Train: {train.shape}")
    print(f"  Valid: {valid.shape}")
    print(f"  Test:  {test.shape}")

    return train, valid, test


def prepare_features(
    train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame
) -> Tuple:
    """피처 준비 및 중복 피처 제거"""
    exclude_cols = ["msno", "is_churn"]
    
    # 중복 피처 제거
    all_cols = [c for c in train.columns if c not in exclude_cols]
    removed_cols = [c for c in DUPLICATE_FEATURES if c in all_cols]
    feature_cols = [c for c in all_cols if c not in DUPLICATE_FEATURES]

    if removed_cols:
        print(f"\n  제거된 중복 피처: {removed_cols}")
    else:
        print(f"\n  제거할 중복 피처 없음 (해당 피처가 데이터에 없음)")
    
    print(f"  최종 피처 수: {len(feature_cols)}")

    X_train = train[feature_cols]
    y_train = train["is_churn"]

    X_valid = valid[feature_cols]
    y_valid = valid["is_churn"]

    X_test = test[feature_cols]
    y_test = test["is_churn"]

    churn_rate = y_train.mean() * 100
    print(f"  Train Churn 비율: {churn_rate:.2f}%")

    return X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols


# ============================================
# 모델 학습
# ============================================
def train_lightgbm_recall(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    recall_weight_multiplier: float = 1.5,
) -> Tuple[Any, Dict[str, Any], np.ndarray]:
    """LightGBM - Recall 최적화"""
    print("\n" + "=" * 60)
    print("LightGBM (Recall 최적화)")
    print("=" * 60)

    base_scale = (y_train == 0).sum() / (y_train == 1).sum()
    scale_pos_weight = base_scale * recall_weight_multiplier
    print(f"  scale_pos_weight: {scale_pos_weight:.2f} (base: {base_scale:.2f} x {recall_weight_multiplier})")

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
    
    # 기본 threshold (0.5)
    valid_metrics_default = evaluate_model(
        y_valid, (y_valid_prob >= 0.5).astype(int), y_valid_prob
    )
    print_metrics(valid_metrics_default, "LightGBM Valid (threshold=0.5)")

    # Feature Importance
    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    results = {
        "model_name": "LightGBM",
        "valid_metrics": valid_metrics_default,
        "params": params,
        "best_iteration": model.best_iteration,
        "feature_importance": importance.head(20).to_dict("records"),
    }

    return model, results, y_valid_prob


def train_catboost_recall(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    recall_weight_multiplier: float = 1.5,
) -> Tuple[Any, Dict[str, Any], np.ndarray]:
    """CatBoost - Recall 최적화"""
    print("\n" + "=" * 60)
    print("CatBoost (Recall 최적화)")
    print("=" * 60)

    base_scale = (y_train == 0).sum() / (y_train == 1).sum()
    scale_pos_weight = base_scale * recall_weight_multiplier
    print(f"  scale_pos_weight: {scale_pos_weight:.2f} (base: {base_scale:.2f} x {recall_weight_multiplier})")

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
    
    # 기본 threshold (0.5)
    valid_metrics_default = evaluate_model(
        y_valid, (y_valid_prob >= 0.5).astype(int), y_valid_prob
    )
    print_metrics(valid_metrics_default, "CatBoost Valid (threshold=0.5)")

    # Feature Importance
    importances = model.get_feature_importance()
    importance_df = pd.DataFrame({
        "feature": list(X_train.columns),
        "importance": importances,
    }).sort_values("importance", ascending=False)

    results = {
        "model_name": "CatBoost",
        "valid_metrics": valid_metrics_default,
        "params": {k: str(v) if not isinstance(v, (int, float, str, bool)) else v 
                   for k, v in params.items()},
        "best_iteration": getattr(model, "best_iteration_", None),
        "feature_importance": importance_df.head(20).to_dict("records"),
    }

    return model, results, y_valid_prob


# ============================================
# 모델 비교 및 선택
# ============================================
def compare_and_select_model(
    lgb_model, lgb_results, lgb_valid_prob,
    cb_model, cb_results, cb_valid_prob,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """두 모델 비교하여 Recall이 높은 모델 선택"""
    print("\n" + "=" * 60)
    print("모델 비교 및 선택 (Recall 우선)")
    print("=" * 60)

    # Validation에서 Recall 비교 (기본 threshold)
    lgb_recall = lgb_results["valid_metrics"]["recall"]
    cb_recall = cb_results["valid_metrics"]["recall"]
    
    print(f"\n  [기본 Threshold 0.5]")
    print(f"  LightGBM Recall: {lgb_recall:.4f}")
    print(f"  CatBoost Recall: {cb_recall:.4f}")

    # Recall이 높은 모델 선택
    if cb_recall > lgb_recall:
        selected_model = cb_model
        selected_name = "CatBoost"
        selected_results = cb_results
        selected_valid_prob = cb_valid_prob
        print(f"\n  >> CatBoost 선택 (Recall: {cb_recall:.4f} > {lgb_recall:.4f})")
    else:
        selected_model = lgb_model
        selected_name = "LightGBM"
        selected_results = lgb_results
        selected_valid_prob = lgb_valid_prob
        print(f"\n  >> LightGBM 선택 (Recall: {lgb_recall:.4f} >= {cb_recall:.4f})")

    # 최적 Threshold 찾기 (목표 Recall: 95%)
    print("\n  [Threshold 최적화 - 목표 Recall: 95%]")
    optimal_threshold, optimal_valid_metrics = find_optimal_threshold_for_recall(
        y_valid.values, selected_valid_prob, target_recall=0.95
    )
    print(f"  최적 Threshold: {optimal_threshold:.2f}")
    print(f"  Recall: {optimal_valid_metrics['recall']:.4f}")
    print(f"  Precision: {optimal_valid_metrics['precision']:.4f}")

    # Test Set 평가
    print("\n" + "=" * 60)
    print(f"Test Set 평가 ({selected_name})")
    print("=" * 60)

    if selected_name == "CatBoost":
        y_test_prob = cb_model.predict_proba(X_test)[:, 1]
    else:
        y_test_prob = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

    # 기본 threshold
    test_metrics_default = evaluate_model(
        y_test, (y_test_prob >= 0.5).astype(int), y_test_prob
    )
    print_metrics(test_metrics_default, f"{selected_name} Test (threshold=0.5)")

    # 최적 threshold
    test_metrics_optimal = evaluate_model(
        y_test, (y_test_prob >= optimal_threshold).astype(int), y_test_prob
    )
    print_metrics(test_metrics_optimal, f"{selected_name} Test (threshold={optimal_threshold:.2f})")

    # 결과 정리
    final_results = {
        "selected_model": selected_name,
        "comparison": {
            "LightGBM": {
                "recall": lgb_recall,
                "roc_auc": lgb_results["valid_metrics"]["roc_auc"],
            },
            "CatBoost": {
                "recall": cb_recall,
                "roc_auc": cb_results["valid_metrics"]["roc_auc"],
            },
        },
        "optimal_threshold": optimal_threshold,
        "valid_metrics_default": selected_results["valid_metrics"],
        "valid_metrics_optimal": optimal_valid_metrics,
        "test_metrics_default": test_metrics_default,
        "test_metrics_optimal": test_metrics_optimal,
        "feature_importance": selected_results["feature_importance"],
        "params": selected_results["params"],
        "best_iteration": selected_results.get("best_iteration"),
    }

    return final_results, selected_model, selected_name


# ============================================
# 리포트 생성
# ============================================
def generate_report(
    final_results: Dict[str, Any],
    feature_cols: List[str],
    report_dir: Path,
) -> None:
    """마크다운 리포트 생성"""
    report_dir.mkdir(parents=True, exist_ok=True)

    selected = final_results["selected_model"]
    lgb_comp = final_results["comparison"]["LightGBM"]
    cb_comp = final_results["comparison"]["CatBoost"]
    
    opt_thresh = final_results["optimal_threshold"]
    test_default = final_results["test_metrics_default"]
    test_optimal = final_results["test_metrics_optimal"]
    feat_imp = final_results["feature_importance"]

    md = f"""# 05. Recall 최적화 모델 선택 결과

> **작성자**: 이도훈 (LDH)  
> **작성일**: {datetime.now().strftime("%Y-%m-%d")}  
> **목표**: Precision보다 **Recall** 최대화

---

## 1. 학습 개요

### 1.1 최적화 전략
| 항목 | 설명 |
|------|------|
| **목표** | Recall(재현율) 최대화 - 이탈 고객 최대한 탐지 |
| **비교 모델** | LightGBM vs CatBoost |
| **선택 기준** | Validation Recall이 높은 모델 |
| **피처 처리** | 중복 피처 제거 (recency_*_ratio) |

### 1.2 데이터 정보
| 항목 | 값 |
|------|-----|
| 피처 수 | {len(feature_cols)} |
| 제거된 중복 피처 | {', '.join(DUPLICATE_FEATURES) if DUPLICATE_FEATURES else '없음'} |

---

## 2. 모델 비교 (Validation Set)

### 2.1 Recall 비교 (기본 Threshold 0.5)

| 모델 | Recall | ROC-AUC | 선택 |
|------|--------|---------|------|
| **LightGBM** | {lgb_comp['recall']:.4f} | {lgb_comp['roc_auc']:.4f} | {'✅ 선택' if selected == 'LightGBM' else ''} |
| **CatBoost** | {cb_comp['recall']:.4f} | {cb_comp['roc_auc']:.4f} | {'✅ 선택' if selected == 'CatBoost' else ''} |

### 2.2 선택된 모델
- **선택**: {selected}
- **선택 사유**: Recall이 더 높음 ({max(lgb_comp['recall'], cb_comp['recall']):.4f})

---

## 3. Threshold 최적화

### 3.1 최적 Threshold 선정
| 항목 | 기본값 (0.50) | 최적값 ({opt_thresh:.2f}) | 변화 |
|------|---------------|---------------------------|------|
| **Recall** | {test_default['recall']:.4f} | {test_optimal['recall']:.4f} | {'+' if test_optimal['recall'] > test_default['recall'] else ''}{(test_optimal['recall'] - test_default['recall'])*100:.2f}%p |
| **Precision** | {test_default['precision']:.4f} | {test_optimal['precision']:.4f} | {'+' if test_optimal['precision'] > test_default['precision'] else ''}{(test_optimal['precision'] - test_default['precision'])*100:.2f}%p |
| **F1-Score** | {test_default['f1']:.4f} | {test_optimal['f1']:.4f} | {'+' if test_optimal['f1'] > test_default['f1'] else ''}{(test_optimal['f1'] - test_default['f1'])*100:.2f}%p |

---

## 4. Test Set 최종 성능

### 4.1 성능 지표 (최적 Threshold)

| 지표 | 값 |
|------|-----|
| **ROC-AUC** | {test_optimal['roc_auc']:.4f} |
| **PR-AUC** | {test_optimal['pr_auc']:.4f} |
| **Recall** | **{test_optimal['recall']:.4f}** |
| **Precision** | {test_optimal['precision']:.4f} |
| **F1-Score** | {test_optimal['f1']:.4f} |
| **Specificity** | {test_optimal['specificity']:.4f} |

### 4.2 Confusion Matrix (최적 Threshold)

```
              Predicted
              0        1
Actual  0    {test_optimal['true_negative']:,}    {test_optimal['false_positive']:,}
        1    {test_optimal['false_negative']:,}    {test_optimal['true_positive']:,}
```

### 4.3 성능 해석
- **탐지된 이탈자**: {test_optimal['true_positive']:,}명 / {test_optimal['true_positive'] + test_optimal['false_negative']:,}명 (Recall: {test_optimal['recall']*100:.1f}%)
- **놓친 이탈자**: {test_optimal['false_negative']:,}명 (False Negative)
- **잘못 예측된 유지 고객**: {test_optimal['false_positive']:,}명 (False Positive)

---

## 5. Feature Importance (Top 20)

| 순위 | Feature | Importance |
|------|---------|------------|
"""

    for i, feat in enumerate(feat_imp[:20], 1):
        md += f"| {i} | `{feat['feature']}` | {feat['importance']:.2f} |\n"

    md += f"""
---

## 6. 결론

### 6.1 최종 결과 요약

| 항목 | 값 |
|------|-----|
| **선택된 모델** | {selected} |
| **최적 Threshold** | {opt_thresh:.2f} |
| **최종 Recall** | **{test_optimal['recall']*100:.1f}%** |
| **ROC-AUC** | {test_optimal['roc_auc']:.4f} |

### 6.2 비즈니스 활용
1. **높은 Recall**: 이탈 가능 고객의 {test_optimal['recall']*100:.1f}%를 사전 식별
2. **선제적 대응**: 식별된 고객에게 리텐션 캠페인 진행
3. **Trade-off**: Precision이 낮아 False Positive 증가 → 마케팅 비용 고려 필요

---

## 7. 저장된 파일

| 파일 | 경로 | 설명 |
|------|------|------|
| 모델 | `models/{selected.lower()}_recall_selected.txt/cbm` | Recall 최적화 모델 |
| 결과 JSON | `models/recall_selected_results.json` | 전체 결과 |
| 리포트 | `docs/02_training_report/05_recall_model_selection.md` | 이 문서 |

---

> **핵심 메시지**: {selected} 모델을 Threshold {opt_thresh:.2f}로 설정하면 이탈 고객의 **{test_optimal['recall']*100:.1f}%**를 탐지할 수 있습니다.
"""

    report_path = report_dir / "05_recall_model_selection.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n리포트 저장: {report_path}")


# ============================================
# 메인 파이프라인
# ============================================
def run_recall_optimized_training() -> Dict[str, Any]:
    """전체 파이프라인 실행"""
    print("=" * 60)
    print("KKBox - Recall 최적화 모델 선택")
    print("=" * 60)

    # 1. 데이터 로드
    train, valid, test = load_datasets()

    # 2. 피처 준비
    X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols = prepare_features(
        train, valid, test
    )

    # 3. LightGBM 학습
    lgb_model, lgb_results, lgb_valid_prob = train_lightgbm_recall(
        X_train, y_train, X_valid, y_valid
    )

    # 4. CatBoost 학습
    cb_model, cb_results, cb_valid_prob = train_catboost_recall(
        X_train, y_train, X_valid, y_valid
    )

    # 5. 모델 비교 및 선택
    final_results, selected_model, selected_name = compare_and_select_model(
        lgb_model, lgb_results, lgb_valid_prob,
        cb_model, cb_results, cb_valid_prob,
        y_valid, X_test, y_test
    )

    # 6. 모델 저장
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    if selected_name == "LightGBM":
        selected_model.save_model(str(MODEL_DIR / "lightgbm_recall_selected.txt"))
        print(f"\n모델 저장: {MODEL_DIR / 'lightgbm_recall_selected.txt'}")
    else:
        selected_model.save_model(str(MODEL_DIR / "catboost_recall_selected.cbm"))
        print(f"\n모델 저장: {MODEL_DIR / 'catboost_recall_selected.cbm'}")

    # 결과 JSON 저장
    with open(MODEL_DIR / "recall_selected_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    # 7. 리포트 생성
    generate_report(final_results, feature_cols, REPORT_DIR)

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)

    return final_results


# ============================================
# 실행
# ============================================
if __name__ == "__main__":
    results = run_recall_optimized_training()

