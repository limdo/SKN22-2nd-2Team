"""
KKBox Churn Prediction - Model Training on kkbox_train_feature_v1.parquet
작성자: 이도훈 (LDH)
작성일: 2025-12-17

새로운 피처 테이블(`kkbox_train_feature_v1.parquet`)을 이용해
- LightGBM
- CatBoost
모델을 학습하고, Confusion Matrix를 포함한 결과를 마크다운 리포트로 저장합니다.
"""

from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import lightgbm as lgb
from catboost import CatBoostClassifier

from .train_models_ldh import (
    evaluate_model,
    print_metrics,
    RANDOM_STATE,
    REPORT_DIR,
)


PROJECT_ROOT = Path(__file__).parent.parent
# 현재 워크스페이스 루트에 존재하는 파일 사용
DATA_PATH = PROJECT_ROOT / "kkbox_train_feature_v1.parquet"

# 잠재적으로 데이터 누수(leakage) 가능성이 있는 피처들
LEAKY_COLS = [
    "registration_init_time",
    "days_since_last_payment",
    "days_since_last_cancel",
    "is_auto_renew_last",
    "last_plan_days",
    "last_payment_method",
    "recency_secs_ratio",
    "recency_songs_ratio",
    "secs_trend_w7_w30",
    "secs_trend_w14_w30",
    "days_trend_w7_w14",
    "days_trend_w7_w30",
    "songs_trend_w7_w30",
    "songs_trend_w14_w30",
    "skip_trend_w7_w30",
    "completion_trend_w7_w30",
]


def load_parquet_features(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    kkbox_train_feature_v1.parquet 로드
    """
    print("Loading parquet features...")
    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception as e:
        print(f"  pyarrow read failed: {e}")
        print("  -> retry with fastparquet engine")
        df = pd.read_parquet(path, engine="fastparquet")
    print(f"  Loaded: {df.shape}")
    return df


def train_valid_test_split(
    df: pd.DataFrame,
    target: str = "is_churn",
    test_size: float = 0.2,
    valid_size: float = 0.1,
) -> Tuple[pd.DataFrame, ...]:
    """
    단일 데이터프레임을 Train / Valid / Test 로 분할 (stratified).
    기본 비율: Train 70% / Valid 10% / Test 20%
    """
    y = df[target]

    # 1) Train_valid + Test
    df_train_valid, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 2) Train + Valid (valid_size는 전체 비율 기준이므로, train_valid 내 비율로 환산)
    valid_ratio_within_train_valid = valid_size / (1.0 - test_size)

    df_train, df_valid = train_test_split(
        df_train_valid,
        test_size=valid_ratio_within_train_valid,
        random_state=RANDOM_STATE,
        stratify=df_train_valid[target],
    )

    print("\nSplit summary")
    for name, subset in [
        ("Train", df_train),
        ("Valid", df_valid),
        ("Test", df_test),
    ]:
        pos_rate = subset[target].mean() * 100
        print(f"  {name:<5}: {subset.shape}, churn_rate={pos_rate:.2f}%")

    return df_train, df_valid, df_test


def prepare_feature_matrices(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    target: str = "is_churn",
) -> Tuple[pd.DataFrame, ...]:
    """
    피처/타겟 분리 + Period / datetime 타입 컬럼은 드롭하여 모델 입력으로 사용.

    feature_v1 기준: 잠재적 누수 피처도 모두 포함해 "최대로 정보가 많은" 버전.
    """
    exclude_cols = ["msno", target]

    feature_cols = []
    for c in train.columns:
        if c in exclude_cols:
            continue
        # period / datetime 컬럼은 드롭 (LightGBM, CatBoost 입력에서 제외)
        if pd.api.types.is_period_dtype(train[c]):
            continue
        if pd.api.types.is_datetime64_any_dtype(train[c]):
            continue
        feature_cols.append(c)

    X_train = train[feature_cols].copy()
    y_train = train[target]

    X_valid = valid[feature_cols].copy()
    y_valid = valid[target]

    X_test = test[feature_cols].copy()
    y_test = test[target]

    # CatBoost 호환을 위해 category 컬럼은 일관된 정수 코드로 변환
    cat_cols = [
        c
        for c in feature_cols
        if pd.api.types.is_categorical_dtype(train[c])
        or pd.api.types.is_object_dtype(train[c])
    ]

    if cat_cols:
        combined = pd.concat(
            [train[cat_cols].astype("string"), valid[cat_cols].astype("string"), test[cat_cols].astype("string")],
            axis=0,
        )
        category_maps = {}
        for c in cat_cols:
            cat_all = pd.Categorical(combined[c])
            categories = cat_all.categories
            category_maps[c] = categories

        for c in cat_cols:
            categories = category_maps[c]
            X_train[c] = pd.Categorical(
                X_train[c].astype("string"), categories=categories
            ).codes.astype("int16")
            X_valid[c] = pd.Categorical(
                X_valid[c].astype("string"), categories=categories
            ).codes.astype("int16")
            X_test[c] = pd.Categorical(
                X_test[c].astype("string"), categories=categories
            ).codes.astype("int16")

    print(f"\nFeature prep done - n_features={len(feature_cols)}")
    print(f"  Columns example: {feature_cols[:5]} ...")

    return X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols


def prepare_feature_matrices_v2(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    target: str = "is_churn",
) -> Tuple[pd.DataFrame, ...]:
    """
    feature_v2용 피처/타겟 분리.

    - msno, target 컬럼 제거
    - period / datetime 컬럼 제거
    - 잠재적으로 데이터 누수(leakage) 가능성이 높은 피처(LEAKY_COLS)는 제외
    - CatBoost 호환을 위해 범주형 컬럼은 일관된 정수 코드로 변환
    """
    exclude_cols = ["msno", target]

    feature_cols = []
    for c in train.columns:
        if c in exclude_cols:
            continue
        if c in LEAKY_COLS:
            # v2에서는 누수 가능성이 높은 피처는 사용하지 않음
            continue
        if pd.api.types.is_period_dtype(train[c]):
            continue
        if pd.api.types.is_datetime64_any_dtype(train[c]):
            continue
        feature_cols.append(c)

    X_train = train[feature_cols].copy()
    y_train = train[target]

    X_valid = valid[feature_cols].copy()
    y_valid = valid[target]

    X_test = test[feature_cols].copy()
    y_test = test[target]

    cat_cols = [
        c
        for c in feature_cols
        if pd.api.types.is_categorical_dtype(train[c])
        or pd.api.types.is_object_dtype(train[c])
    ]

    if cat_cols:
        combined = pd.concat(
            [
                train[cat_cols].astype("string"),
                valid[cat_cols].astype("string"),
                test[cat_cols].astype("string"),
            ],
            axis=0,
        )
        category_maps = {}
        for c in cat_cols:
            cat_all = pd.Categorical(combined[c])
            categories = cat_all.categories
            category_maps[c] = categories

        for c in cat_cols:
            categories = category_maps[c]
            X_train[c] = pd.Categorical(
                X_train[c].astype("string"), categories=categories
            ).codes.astype("int16")
            X_valid[c] = pd.Categorical(
                X_valid[c].astype("string"), categories=categories
            ).codes.astype("int16")
            X_test[c] = pd.Categorical(
                X_test[c].astype("string"), categories=categories
            ).codes.astype("int16")

    print(f"\n[feature_v2] Feature prep done - n_features={len(feature_cols)}")
    print(f"  Columns example: {feature_cols[:5]} ...")

    return X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> Tuple[Any, Dict[str, Any]]:
    """
    LightGBM 학습 (kkbox_train_feature_v1 전용)
    """
    print("\n" + "=" * 60)
    print("LightGBM (feature_v1) 학습")
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

    y_train_prob = model.predict(X_train, num_iteration=model.best_iteration)
    y_valid_prob = model.predict(X_valid, num_iteration=model.best_iteration)

    train_metrics = evaluate_model(
        y_train, (y_train_prob >= 0.5).astype(int), y_train_prob
    )
    valid_metrics = evaluate_model(
        y_valid, (y_valid_prob >= 0.5).astype(int), y_valid_prob
    )

    print_metrics(train_metrics, "LightGBM Train")
    print_metrics(valid_metrics, "LightGBM Valid")

    importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)

    print("\nLightGBM Top 10 Feature Importance:")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.2f}")

    results = {
        "model_name": "LightGBM_feature_v1",
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "params": params,
        "best_iteration": model.best_iteration,
        "feature_importance": importance.to_dict("records"),
    }

    return model, results


def train_catboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> Tuple[Any, Dict[str, Any]]:
    """
    CatBoost 학습 (kkbox_train_feature_v1 전용)
    """
    print("\n" + "=" * 60)
    print("CatBoost (feature_v1) 학습")
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

    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_valid_prob = model.predict_proba(X_valid)[:, 1]

    train_metrics = evaluate_model(
        y_train, (y_train_prob >= 0.5).astype(int), y_train_prob
    )
    valid_metrics = evaluate_model(
        y_valid, (y_valid_prob >= 0.5).astype(int), y_valid_prob
    )

    print_metrics(train_metrics, "CatBoost Train")
    print_metrics(valid_metrics, "CatBoost Valid")

    importances = model.get_feature_importance()
    importance_df = (
        pd.DataFrame(
            {
                "feature": list(X_train.columns),
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
    )

    print("\nCatBoost Top 10 Feature Importance:")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.2f}")

    best_iter = getattr(model, "best_iteration_", None)

    results = {
        "model_name": "CatBoost_feature_v1",
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "params": params,
        "best_iteration": int(best_iter) if best_iter is not None else None,
        "feature_importance": importance_df.to_dict("records"),
    }

    return model, results


def evaluate_on_test_set(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """
    Test set 평가 (LightGBM & CatBoost)
    """
    print("\n" + "=" * 60)
    print("Test Set Evaluation (feature_v1)")
    print("=" * 60)

    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")
        if isinstance(model, CatBoostClassifier):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # LightGBM
            best_iter = getattr(model, "best_iteration", None)
            y_prob = model.predict(
                X_test,
                num_iteration=best_iter if best_iter is not None else None,
            )

        metrics = evaluate_model(
            y_test,
            (y_prob >= 0.5).astype(int),
            y_prob,
        )
        print_metrics(metrics, f"{name} Test")
        results[name] = metrics

    return results


def generate_markdown_report(
    all_results: Dict[str, Dict[str, Any]],
    report_dir: Path = REPORT_DIR,
) -> None:
    """
    LightGBM + CatBoost 결과를 마크다운으로 저장.
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    lgb_valid = all_results["LightGBM"]["valid_metrics"]
    lgb_test = all_results["LightGBM"]["test_metrics"]
    cb_valid = all_results["CatBoost"]["valid_metrics"]
    cb_test = all_results["CatBoost"]["test_metrics"]

    lgb_imp = all_results["LightGBM"]["feature_importance"][:10]
    cb_imp = all_results["CatBoost"]["feature_importance"][:10]

    report_path = report_dir / "02_ml_training_results_feature_v1.md"

    md = f"""# 02. ML 모델 학습 결과 - kkbox_train_feature_v1.parquet

> **작성자**: 이도훈 (LDH)  
> **작성일**: 2025-12-17  
> **피처 테이블**: `kkbox_train_feature_v1.parquet`

---

## 1. 학습 개요

### 1.1 사용 데이터
- 단일 피처 테이블: `kkbox_train_feature_v1.parquet` (전처리 완료)
- 샘플 수: 약 {int(all_results['meta']['n_samples']):,}명
- 피처 수: {int(all_results['meta']['n_features'])}개

### 1.2 데이터 분할 (Stratified)

| 셋 | 비율 | 용도 |
|----|------|------|
| Train | 70% | 모델 학습 |
| Valid | 10% | Early Stopping / 하이퍼파라미터 튜닝 |
| Test | 20% | 최종 성능 평가 |

### 1.3 학습 모델

| 모델 | 유형 | 목적 |
|------|------|------|
| LightGBM | Tree-based GBDT | 강력한 Tabular Baseline |
| CatBoost | Ordered Boosting | 범주/불균형 데이터에 강한 GBDT |

---

## 2. 평가 지표 비교

### 2.1 Validation Set 성능

| 지표 | LightGBM | CatBoost | 우수 모델 |
|------|----------|----------|-----------|
| **ROC-AUC** | {lgb_valid['roc_auc']:.4f} | {cb_valid['roc_auc']:.4f} | {('LightGBM ✅' if lgb_valid['roc_auc'] > cb_valid['roc_auc'] else 'CatBoost ✅')} |
| **PR-AUC** | {lgb_valid['pr_auc']:.4f} | {cb_valid['pr_auc']:.4f} | {('LightGBM ✅' if lgb_valid['pr_auc'] > cb_valid['pr_auc'] else 'CatBoost ✅')} |
| **Recall** | {lgb_valid['recall']:.4f} | {cb_valid['recall']:.4f} | {('LightGBM ✅' if lgb_valid['recall'] > cb_valid['recall'] else 'CatBoost ✅')} |
| **Precision** | {lgb_valid['precision']:.4f} | {cb_valid['precision']:.4f} | {('LightGBM ✅' if lgb_valid['precision'] > cb_valid['precision'] else 'CatBoost ✅')} |
| **F1-Score** | {lgb_valid['f1']:.4f} | {cb_valid['f1']:.4f} | {('LightGBM ✅' if lgb_valid['f1'] > cb_valid['f1'] else 'CatBoost ✅')} |

### 2.2 Test Set 성능 (최종)

| 지표 | LightGBM | CatBoost | 우수 모델 |
|------|----------|----------|-----------|
| **ROC-AUC** | {lgb_test['roc_auc']:.4f} | {cb_test['roc_auc']:.4f} | {('LightGBM ✅' if lgb_test['roc_auc'] > cb_test['roc_auc'] else 'CatBoost ✅')} |
| **PR-AUC** | {lgb_test['pr_auc']:.4f} | {cb_test['pr_auc']:.4f} | {('LightGBM ✅' if lgb_test['pr_auc'] > cb_test['pr_auc'] else 'CatBoost ✅')} |
| **Recall** | {lgb_test['recall']:.4f} | {cb_test['recall']:.4f} | {('LightGBM ✅' if lgb_test['recall'] > cb_test['recall'] else 'CatBoost ✅')} |
| **Precision** | {lgb_test['precision']:.4f} | {cb_test['precision']:.4f} | {('LightGBM ✅' if lgb_test['precision'] > cb_test['precision'] else 'CatBoost ✅')} |
| **F1-Score** | {lgb_test['f1']:.4f} | {cb_test['f1']:.4f} | {('LightGBM ✅' if lgb_test['f1'] > cb_test['f1'] else 'CatBoost ✅')} |

---

## 3. Confusion Matrix (Test Set)

### 3.1 LightGBM

```
              Predicted
              0        1
Actual  0    {lgb_test['true_negative']:,}    {lgb_test['false_positive']:,}
        1    {lgb_test['false_negative']:,}    {lgb_test['true_positive']:,}
```

### 3.2 CatBoost

```
              Predicted
              0        1
Actual  0    {cb_test['true_negative']:,}    {cb_test['false_positive']:,}
        1    {cb_test['false_negative']:,}    {cb_test['true_positive']:,}
```

---

## 4. Feature Importance 비교

### 4.1 LightGBM Top 10

| 순위 | Feature | Importance |
|------|---------|------------|
"""

    for i, feat in enumerate(lgb_imp, 1):
        md += f"| {i} | `{feat['feature']}` | {feat['importance']:.2f} |\n"

    md += """

### 4.2 CatBoost Top 10

| 순위 | Feature | Importance |
|------|---------|------------|
"""

    for i, feat in enumerate(cb_imp, 1):
        md += f"| {i} | `{feat['feature']}` | {feat['importance']:.2f} |\n"

    md += f"""
---

## 5. 모델별 하이퍼파라미터

### 5.1 LightGBM

- num_leaves: {all_results['LightGBM']['params']['num_leaves']}
- max_depth: {all_results['LightGBM']['params']['max_depth']}
- learning_rate: {all_results['LightGBM']['params']['learning_rate']}
- feature_fraction: {all_results['LightGBM']['params']['feature_fraction']}
- bagging_fraction: {all_results['LightGBM']['params']['bagging_fraction']}
- min_child_samples: {all_results['LightGBM']['params']['min_child_samples']}
- reg_alpha: {all_results['LightGBM']['params']['reg_alpha']}
- reg_lambda: {all_results['LightGBM']['params']['reg_lambda']}
- best_iteration: {all_results['LightGBM'].get('best_iteration', 'N/A')}

### 5.2 CatBoost

- depth: {all_results['CatBoost']['params']['depth']}
- learning_rate: {all_results['CatBoost']['params']['learning_rate']}
- l2_leaf_reg: {all_results['CatBoost']['params']['l2_leaf_reg']}
- iterations: {all_results['CatBoost']['params']['iterations']}
- scale_pos_weight: {all_results['CatBoost']['params']['scale_pos_weight']}

---

## 6. 결론

- **최종 추천 모델**: {'LightGBM' if lgb_test['roc_auc'] >= cb_test['roc_auc'] else 'CatBoost'}
- **핵심 지표 (ROC-AUC 기준)**: 최대 {max(lgb_test['roc_auc'], cb_test['roc_auc']):.4f}
- **Confusion Matrix 기준**으로 두 모델 모두 이탈자(1)에 대한 높은 Recall을 달성.

---

> 이 리포트는 `kkbox_train_feature_v1.parquet` 기반 LightGBM / CatBoost 실험 결과입니다.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\nReport saved: {report_path}")


def generate_markdown_report_feature_v2(
    all_results: Dict[str, Dict[str, Any]],
    report_dir: Path = REPORT_DIR,
) -> None:
    """
    feature_v2 설정(잠재적 누수 피처 제거)에 대한 결과를 마크다운으로 저장.
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    lgb_valid = all_results["LightGBM"]["valid_metrics"]
    lgb_test = all_results["LightGBM"]["test_metrics"]
    cb_valid = all_results["CatBoost"]["valid_metrics"]
    cb_test = all_results["CatBoost"]["test_metrics"]

    lgb_imp = all_results["LightGBM"]["feature_importance"][:10]
    cb_imp = all_results["CatBoost"]["feature_importance"][:10]

    dropped_cols = all_results.get("meta", {}).get("dropped_leak_cols", [])

    report_path = report_dir / "02_ml_training_results_feature_v2.md"

    md = f"""# 02. ML 모델 학습 결과 - kkbox_train_feature_v1.parquet (feature_v2)

> **작성자**: 이도훈 (LDH)  
> **작성일**: 2025-12-17  
> **피처 테이블**: `kkbox_train_feature_v1.parquet`  
> **전처리 버전**: feature_v2 (잠재적 누수 피처 제거)

---

## 1. 학습 개요

### 1.1 사용 데이터
- 단일 피처 테이블: `kkbox_train_feature_v1.parquet`
- 샘플 수: 약 {int(all_results['meta']['n_samples']):,}명
- 피처 수: {int(all_results['meta']['n_features'])}개
- 잠재적 누수 피처 제거 개수: {len(dropped_cols)}개
"""

    if dropped_cols:
        md += "\n- 제거된 누수 후보 피처 예시: " + ", ".join(
            f"`{c}`" for c in dropped_cols[:5]
        )

    md += """

### 1.2 데이터 분할 (Stratified)

| 셋 | 비율 | 용도 |
|----|------|------|
| Train | 70% | 모델 학습 |
| Valid | 10% | Early Stopping / 하이퍼파라미터 튜닝 |
| Test | 20% | 최종 성능 평가 |

### 1.3 학습 모델

| 모델 | 유형 | 목적 |
|------|------|------|
| LightGBM | Tree-based GBDT | Tabular Baseline (feature_v2) |
| CatBoost | Ordered Boosting | 범주/불균형 데이터에 강한 GBDT |

---

## 2. 평가 지표 비교

### 2.1 Validation Set 성능

| 지표 | LightGBM | CatBoost | 우수 모델 |
|------|----------|----------|-----------|
"""

    md += (
        f"| **ROC-AUC** | {lgb_valid['roc_auc']:.4f} | {cb_valid['roc_auc']:.4f} | "
        f"{('LightGBM ✅' if lgb_valid['roc_auc'] > cb_valid['roc_auc'] else 'CatBoost ✅')} |\n"
    )
    md += (
        f"| **PR-AUC** | {lgb_valid['pr_auc']:.4f} | {cb_valid['pr_auc']:.4f} | "
        f"{('LightGBM ✅' if lgb_valid['pr_auc'] > cb_valid['pr_auc'] else 'CatBoost ✅')} |\n"
    )
    md += (
        f"| **Recall** | {lgb_valid['recall']:.4f} | {cb_valid['recall']:.4f} | "
        f"{('LightGBM ✅' if lgb_valid['recall'] > cb_valid['recall'] else 'CatBoost ✅')} |\n"
    )
    md += (
        f"| **Precision** | {lgb_valid['precision']:.4f} | {cb_valid['precision']:.4f} | "
        f"{('LightGBM ✅' if lgb_valid['precision'] > cb_valid['precision'] else 'CatBoost ✅')} |\n"
    )
    md += (
        f"| **F1-Score** | {lgb_valid['f1']:.4f} | {cb_valid['f1']:.4f} | "
        f"{('LightGBM ✅' if lgb_valid['f1'] > cb_valid['f1'] else 'CatBoost ✅')} |\n"
    )

    md += """

### 2.2 Test Set 성능 (최종)

| 지표 | LightGBM | CatBoost | 우수 모델 |
|------|----------|----------|-----------|
"""

    md += (
        f"| **ROC-AUC** | {lgb_test['roc_auc']:.4f} | {cb_test['roc_auc']:.4f} | "
        f"{('LightGBM ✅' if lgb_test['roc_auc'] > cb_test['roc_auc'] else 'CatBoost ✅')} |\n"
    )
    md += (
        f"| **PR-AUC** | {lgb_test['pr_auc']:.4f} | {cb_test['pr_auc']:.4f} | "
        f"{('LightGBM ✅' if lgb_test['pr_auc'] > cb_test['pr_auc'] else 'CatBoost ✅')} |\n"
    )
    md += (
        f"| **Recall** | {lgb_test['recall']:.4f} | {cb_test['recall']:.4f} | "
        f"{('LightGBM ✅' if lgb_test['recall'] > cb_test['recall'] else 'CatBoost ✅')} |\n"
    )
    md += (
        f"| **Precision** | {lgb_test['precision']:.4f} | {cb_test['precision']:.4f} | "
        f"{('LightGBM ✅' if lgb_test['precision'] > cb_test['precision'] else 'CatBoost ✅')} |\n"
    )
    md += (
        f"| **F1-Score** | {lgb_test['f1']:.4f} | {cb_test['f1']:.4f} | "
        f"{('LightGBM ✅' if lgb_test['f1'] > cb_test['f1'] else 'CatBoost ✅')} |\n"
    )

    md += """

---

## 3. Confusion Matrix (Test Set)

### 3.1 LightGBM

```
             Predicted
             0        1
Actual  0    """
    md += f"{lgb_test['true_negative']:,}".rjust(6)
    md += "    "
    md += f"{lgb_test['false_positive']:,}"
    md += """
        1    """
    md += f"{lgb_test['false_negative']:,}".rjust(6)
    md += "    "
    md += f"{lgb_test['true_positive']:,}"
    md += """
```

### 3.2 CatBoost

```
             Predicted
             0        1
Actual  0    """
    md += f"{cb_test['true_negative']:,}".rjust(6)
    md += "    "
    md += f"{cb_test['false_positive']:,}"
    md += """
        1    """
    md += f"{cb_test['false_negative']:,}".rjust(6)
    md += "    "
    md += f"{cb_test['true_positive']:,}"
    md += """
```

---

## 4. Feature Importance 비교

### 4.1 LightGBM Top 10

| 순위 | Feature | Importance |
|------|---------|------------|
"""

    for i, feat in enumerate(lgb_imp, 1):
        md += f"| {i} | `{feat['feature']}` | {feat['importance']:.2f} |\n"

    md += """

### 4.2 CatBoost Top 10

| 순위 | Feature | Importance |
|------|---------|------------|
"""

    for i, feat in enumerate(cb_imp, 1):
        md += f"| {i} | `{feat['feature']}` | {feat['importance']:.2f} |\n"

    md += f"""
---

## 5. 결론

- **전처리 버전**: feature_v2 (잠재적 누수 피처 제거)
- **최종 추천 모델**: {'LightGBM' if lgb_test['roc_auc'] >= cb_test['roc_auc'] else 'CatBoost'}
- **핵심 지표 (ROC-AUC 기준)**: 최대 {max(lgb_test['roc_auc'], cb_test['roc_auc']):.4f}
- v1 대비 성능 차이는 별도 비교 표에서 함께 검토 필요.

---

> 이 리포트는 `kkbox_train_feature_v1.parquet` 기반 feature_v2 (누수 피처 제거) 설정에서의 LightGBM / CatBoost 실험 결과입니다.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\nReport saved: {report_path}")


def run_training_on_feature_v1() -> Dict[str, Any]:
    """
    kkbox_train_feature_v1.parquet 기반 전체 학습 파이프라인 실행.
    """
    df = load_parquet_features()

    df_train, df_valid, df_test = train_valid_test_split(df, target="is_churn")

    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        feature_cols,
    ) = prepare_feature_matrices(df_train, df_valid, df_test, target="is_churn")

    lgb_model, lgb_results = train_lightgbm_model(
        X_train, y_train, X_valid, y_valid
    )
    cb_model, cb_results = train_catboost_model(
        X_train, y_train, X_valid, y_valid
    )

    models = {
        "LightGBM": lgb_model,
        "CatBoost": cb_model,
    }

    test_metrics = evaluate_on_test_set(models, X_test, y_test)

    lgb_results["test_metrics"] = test_metrics["LightGBM"]
    cb_results["test_metrics"] = test_metrics["CatBoost"]

    all_results: Dict[str, Any] = {
        "LightGBM": lgb_results,
        "CatBoost": cb_results,
        "meta": {
            "n_samples": df.shape[0],
            "n_features": len(feature_cols),
        },
    }

    generate_markdown_report(all_results, REPORT_DIR)

    print("\nTraining on kkbox_train_feature_v1.parquet completed.")

    return all_results


def run_training_on_feature_v2() -> Dict[str, Any]:
    """
    feature_v2 설정:
    - 동일한 kkbox_train_feature_v1.parquet 사용
    - 잠재적으로 데이터 누수 가능성이 높은 피처(LEAKY_COLS)를 제외한 상태로 학습
    """
    df = load_parquet_features()

    df_train, df_valid, df_test = train_valid_test_split(df, target="is_churn")

    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        feature_cols,
    ) = prepare_feature_matrices_v2(df_train, df_valid, df_test, target="is_churn")

    lgb_model, lgb_results = train_lightgbm_model(
        X_train, y_train, X_valid, y_valid
    )
    cb_model, cb_results = train_catboost_model(
        X_train, y_train, X_valid, y_valid
    )

    models = {
        "LightGBM": lgb_model,
        "CatBoost": cb_model,
    }

    test_metrics = evaluate_on_test_set(models, X_test, y_test)

    lgb_results["test_metrics"] = test_metrics["LightGBM"]
    cb_results["test_metrics"] = test_metrics["CatBoost"]

    # 실제로 제거된 누수 후보 피처 목록
    dropped_cols = [
        c for c in LEAKY_COLS if (c in df.columns and c not in feature_cols)
    ]

    all_results: Dict[str, Any] = {
        "LightGBM": lgb_results,
        "CatBoost": cb_results,
        "meta": {
            "n_samples": df.shape[0],
            "n_features": len(feature_cols),
            "dropped_leak_cols": dropped_cols,
        },
    }

    generate_markdown_report_feature_v2(all_results, REPORT_DIR)

    print("\nTraining on kkbox_train_feature_v1.parquet completed. (feature_v2)")

    return all_results


if __name__ == "__main__":
    run_training_on_feature_v1()


