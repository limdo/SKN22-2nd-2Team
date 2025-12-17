"""
KKBox Churn Prediction - Model Training & Evaluation
작성자: 이도훈 (LDH)
작성일: 2025-12-16

이 모듈은 전처리된 데이터를 사용하여 ML 모델을 학습하고 평가합니다.
- Logistic Regression (Baseline)
- LightGBM (Tree-based)

평가 지표:
- ROC-AUC
- PR-AUC (Average Precision)
- Recall
- Precision
- F1-Score
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import warnings

# ML Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    recall_score, 
    precision_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
import lightgbm as lgb
import joblib

warnings.filterwarnings('ignore')

# ============================================
# 설정
# ============================================
DATA_DIR = Path(__file__).parent.parent / 'data'
MODEL_DIR = Path(__file__).parent.parent / 'models'
REPORT_DIR = Path(__file__).parent.parent / 'docs' / '02_training_report'

# 랜덤 시드
RANDOM_STATE = 42

# 제외할 컬럼
EXCLUDE_COLS = ['msno', 'is_churn']


# ============================================
# 데이터 로드 함수
# ============================================
def load_datasets(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    전처리된 train/valid/test 데이터를 로드합니다.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    print("📂 데이터 로드 중...")
    
    train = pd.read_csv(data_dir / 'train_set.csv')
    valid = pd.read_csv(data_dir / 'valid_set.csv')
    test = pd.read_csv(data_dir / 'test_set.csv')
    
    print(f"  ✓ Train: {train.shape}")
    print(f"  ✓ Valid: {valid.shape}")
    print(f"  ✓ Test:  {test.shape}")
    
    return train, valid, test


def prepare_features(train: pd.DataFrame, 
                     valid: pd.DataFrame, 
                     test: pd.DataFrame) -> Tuple:
    """
    학습에 사용할 피처와 타겟을 분리합니다.
    """
    # 피처 컬럼 선택
    feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]
    
    X_train = train[feature_cols]
    y_train = train['is_churn']
    
    X_valid = valid[feature_cols]
    y_valid = valid['is_churn']
    
    X_test = test[feature_cols]
    y_test = test['is_churn']
    
    print(f"\n📊 피처 준비 완료")
    print(f"  피처 수: {len(feature_cols)}")
    print(f"  Train Churn 비율: {y_train.mean()*100:.2f}%")
    print(f"  Valid Churn 비율: {y_valid.mean()*100:.2f}%")
    print(f"  Test Churn 비율: {y_test.mean()*100:.2f}%")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols


# ============================================
# 평가 함수
# ============================================
def evaluate_model(y_true: np.ndarray, 
                   y_pred: np.ndarray, 
                   y_prob: np.ndarray,
                   threshold: float = 0.5) -> Dict[str, float]:
    """
    모델 성능을 평가합니다.
    
    Returns:
        평가 지표 딕셔너리
    """
    # 이진 예측
    y_pred_binary = (y_prob >= threshold).astype(int)
    
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'recall': recall_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary),
        'f1': f1_score(y_true, y_pred_binary),
    }
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    metrics['true_negative'] = int(tn)
    metrics['false_positive'] = int(fp)
    metrics['false_negative'] = int(fn)
    metrics['true_positive'] = int(tp)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


def print_metrics(metrics: Dict[str, float], name: str) -> None:
    """
    평가 지표를 출력합니다.
    """
    print(f"\n[{name}] 평가 결과:")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {metrics['true_positive']:,}  FP: {metrics['false_positive']:,}")
    print(f"    FN: {metrics['false_negative']:,}  TN: {metrics['true_negative']:,}")


# ============================================
# Logistic Regression
# ============================================
def train_logistic_regression(X_train: pd.DataFrame,
                               y_train: pd.Series,
                               X_valid: pd.DataFrame,
                               y_valid: pd.Series) -> Tuple[Any, StandardScaler, Dict]:
    """
    Logistic Regression 모델을 학습합니다.
    """
    print("\n" + "=" * 60)
    print("Logistic Regression 학습")
    print("=" * 60)
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    
    # 모델 정의
    model = LogisticRegression(
        C=1.0,
        class_weight='balanced',  # 클래스 불균형 처리
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # 학습
    print("  학습 중...")
    model.fit(X_train_scaled, y_train)
    
    # 예측
    y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
    y_valid_prob = model.predict_proba(X_valid_scaled)[:, 1]
    
    # 평가
    train_metrics = evaluate_model(y_train, model.predict(X_train_scaled), y_train_prob)
    valid_metrics = evaluate_model(y_valid, model.predict(X_valid_scaled), y_valid_prob)
    
    print_metrics(train_metrics, "Train")
    print_metrics(valid_metrics, "Validation")
    
    results = {
        'model_name': 'Logistic Regression',
        'train_metrics': train_metrics,
        'valid_metrics': valid_metrics,
        'params': {
            'C': 1.0,
            'class_weight': 'balanced',
            'max_iter': 1000
        }
    }
    
    return model, scaler, results


# ============================================
# LightGBM
# ============================================
def train_lightgbm(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_valid: pd.DataFrame,
                   y_valid: pd.Series) -> Tuple[Any, Dict]:
    """
    LightGBM 모델을 학습합니다.
    """
    print("\n" + "=" * 60)
    print("LightGBM 학습")
    print("=" * 60)
    
    # 클래스 불균형 계산
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
    
    # 파라미터 설정
    params = {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight,
        'min_child_samples': 100,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': RANDOM_STATE,
        'verbose': -1,
        'n_jobs': -1
    }
    
    # 데이터셋 생성
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    # 학습
    print("  학습 중...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # 예측
    y_train_prob = model.predict(X_train, num_iteration=model.best_iteration)
    y_valid_prob = model.predict(X_valid, num_iteration=model.best_iteration)
    
    # 평가
    train_metrics = evaluate_model(y_train, (y_train_prob >= 0.5).astype(int), y_train_prob)
    valid_metrics = evaluate_model(y_valid, (y_valid_prob >= 0.5).astype(int), y_valid_prob)
    
    print_metrics(train_metrics, "Train")
    print_metrics(valid_metrics, "Validation")
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 10 Feature Importance:")
    for i, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.2f}")
    
    results = {
        'model_name': 'LightGBM',
        'train_metrics': train_metrics,
        'valid_metrics': valid_metrics,
        'params': params,
        'best_iteration': model.best_iteration,
        'feature_importance': importance.to_dict('records')
    }
    
    return model, results


# ============================================
# 테스트셋 최종 평가
# ============================================
def evaluate_on_test(models: Dict[str, Any],
                     X_test: pd.DataFrame,
                     y_test: pd.Series,
                     scaler: StandardScaler = None) -> Dict[str, Dict]:
    """
    테스트셋에서 모든 모델을 평가합니다.
    """
    print("\n" + "=" * 60)
    print("테스트셋 최종 평가")
    print("=" * 60)
    
    test_results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        if name == 'Logistic Regression':
            X_test_scaled = scaler.transform(X_test)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = model.predict(X_test, num_iteration=model.best_iteration)
        
        metrics = evaluate_model(y_test, (y_prob >= 0.5).astype(int), y_prob)
        print_metrics(metrics, f"{name} (Test)")
        
        test_results[name] = metrics
    
    return test_results


# ============================================
# 결과 저장
# ============================================
def save_results(all_results: Dict,
                 models: Dict,
                 scaler: StandardScaler,
                 model_dir: Path,
                 feature_cols: list) -> None:
    """
    학습 결과와 모델을 저장합니다.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n모델 저장 중...")
    
    # Logistic Regression 저장
    joblib.dump(models['Logistic Regression'], model_dir / 'logistic_regression.pkl')
    joblib.dump(scaler, model_dir / 'scaler.pkl')
    print(f"  ✓ Logistic Regression 저장: {model_dir / 'logistic_regression.pkl'}")
    
    # LightGBM 저장
    models['LightGBM'].save_model(str(model_dir / 'lightgbm.txt'))
    print(f"  ✓ LightGBM 저장: {model_dir / 'lightgbm.txt'}")
    
    # 피처 목록 저장
    with open(model_dir / 'feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)
    print(f"  ✓ Feature 목록 저장: {model_dir / 'feature_cols.json'}")
    
    # 결과 JSON 저장
    # numpy/pandas 타입을 Python 기본 타입으로 변환
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj
    
    results_to_save = convert_types(all_results)
    with open(model_dir / 'training_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"  ✓ 결과 저장: {model_dir / 'training_results.json'}")


def generate_report(all_results: Dict, report_dir: Path) -> None:
    """
    마크다운 리포트를 생성합니다.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    
    lr_valid = all_results['Logistic Regression']['valid_metrics']
    lr_test = all_results['Logistic Regression']['test_metrics']
    lgb_valid = all_results['LightGBM']['valid_metrics']
    lgb_test = all_results['LightGBM']['test_metrics']
    
    # Feature Importance (LightGBM)
    feature_imp = all_results['LightGBM'].get('feature_importance', [])[:10]
    
    report = f"""# 01. ML 모델 학습 결과 (ML Training Results)

> **작성자**: 이도훈 (LDH)  
> **작성일**: 2025-12-16  
> **버전**: v1.0

---

## 1. 학습 개요

### 1.1 모델 목록
| 모델 | 유형 | 목적 |
|------|------|------|
| Logistic Regression | Linear | Baseline 모델 |
| LightGBM | Tree-based | 성능 향상 모델 |

### 1.2 데이터 분할
| 셋 | 비율 | 용도 |
|----|------|------|
| Train | 70% | 모델 학습 |
| Valid | 10% | 하이퍼파라미터 튜닝 / Early Stopping |
| Test | 20% | 최종 성능 평가 |

### 1.3 클래스 불균형 처리
- **Logistic Regression**: `class_weight='balanced'`
- **LightGBM**: `scale_pos_weight` 적용

---

## 2. 평가 지표 비교

### 2.1 Validation Set 성능

| 지표 | Logistic Regression | LightGBM | 우수 모델 |
|------|---------------------|----------|-----------|
| **ROC-AUC** | {lr_valid['roc_auc']:.4f} | {lgb_valid['roc_auc']:.4f} | {'LightGBM ✅' if lgb_valid['roc_auc'] > lr_valid['roc_auc'] else 'Logistic ✅'} |
| **PR-AUC** | {lr_valid['pr_auc']:.4f} | {lgb_valid['pr_auc']:.4f} | {'LightGBM ✅' if lgb_valid['pr_auc'] > lr_valid['pr_auc'] else 'Logistic ✅'} |
| **Recall** | {lr_valid['recall']:.4f} | {lgb_valid['recall']:.4f} | {'LightGBM ✅' if lgb_valid['recall'] > lr_valid['recall'] else 'Logistic ✅'} |
| **Precision** | {lr_valid['precision']:.4f} | {lgb_valid['precision']:.4f} | {'LightGBM ✅' if lgb_valid['precision'] > lr_valid['precision'] else 'Logistic ✅'} |
| **F1-Score** | {lr_valid['f1']:.4f} | {lgb_valid['f1']:.4f} | {'LightGBM ✅' if lgb_valid['f1'] > lr_valid['f1'] else 'Logistic ✅'} |

### 2.2 Test Set 성능 (최종)

| 지표 | Logistic Regression | LightGBM | 우수 모델 |
|------|---------------------|----------|-----------|
| **ROC-AUC** | {lr_test['roc_auc']:.4f} | {lgb_test['roc_auc']:.4f} | {'LightGBM ✅' if lgb_test['roc_auc'] > lr_test['roc_auc'] else 'Logistic ✅'} |
| **PR-AUC** | {lr_test['pr_auc']:.4f} | {lgb_test['pr_auc']:.4f} | {'LightGBM ✅' if lgb_test['pr_auc'] > lr_test['pr_auc'] else 'Logistic ✅'} |
| **Recall** | {lr_test['recall']:.4f} | {lgb_test['recall']:.4f} | {'LightGBM ✅' if lgb_test['recall'] > lr_test['recall'] else 'Logistic ✅'} |
| **Precision** | {lr_test['precision']:.4f} | {lgb_test['precision']:.4f} | {'LightGBM ✅' if lgb_test['precision'] > lr_test['precision'] else 'Logistic ✅'} |
| **F1-Score** | {lr_test['f1']:.4f} | {lgb_test['f1']:.4f} | {'LightGBM ✅' if lgb_test['f1'] > lr_test['f1'] else 'Logistic ✅'} |

---

## 3. Confusion Matrix (Test Set)

### 3.1 Logistic Regression

```
              Predicted
              0        1
Actual  0    {lr_test['true_negative']:,}    {lr_test['false_positive']:,}
        1    {lr_test['false_negative']:,}    {lr_test['true_positive']:,}
```

### 3.2 LightGBM

```
              Predicted
              0        1
Actual  0    {lgb_test['true_negative']:,}    {lgb_test['false_positive']:,}
        1    {lgb_test['false_negative']:,}    {lgb_test['true_positive']:,}
```

---

## 4. Feature Importance (LightGBM)

| 순위 | Feature | Importance |
|------|---------|------------|
"""
    
    for i, feat in enumerate(feature_imp, 1):
        report += f"| {i} | `{feat['feature']}` | {feat['importance']:.2f} |\n"
    
    report += f"""
---

## 5. 모델별 하이퍼파라미터

### 5.1 Logistic Regression

| 파라미터 | 값 |
|----------|-----|
| C (규제 강도) | 1.0 |
| class_weight | balanced |
| max_iter | 1000 |
| solver | lbfgs |

### 5.2 LightGBM

| 파라미터 | 값 |
|----------|-----|
| num_leaves | 31 |
| max_depth | 6 |
| learning_rate | 0.05 |
| feature_fraction | 0.8 |
| bagging_fraction | 0.8 |
| min_child_samples | 100 |
| reg_alpha | 0.1 |
| reg_lambda | 0.1 |
| best_iteration | {all_results['LightGBM'].get('best_iteration', 'N/A')} |

---

## 6. 결론

### 6.1 최종 모델 선정
- **추천 모델**: {'LightGBM' if lgb_test['roc_auc'] > lr_test['roc_auc'] else 'Logistic Regression'}
- **선정 사유**: ROC-AUC 기준 우수한 성능

### 6.2 성능 요약
- **ROC-AUC**: {max(lgb_test['roc_auc'], lr_test['roc_auc']):.4f}
- **PR-AUC**: {max(lgb_test['pr_auc'], lr_test['pr_auc']):.4f}
- **Recall**: {max(lgb_test['recall'], lr_test['recall']):.4f}

### 6.3 주요 이탈 예측 피처
1. **`{feature_imp[0]['feature'] if feature_imp else 'N/A'}`**: 가장 중요한 이탈 신호
2. **`{feature_imp[1]['feature'] if len(feature_imp) > 1 else 'N/A'}`**: 두 번째 중요 피처
3. **`{feature_imp[2]['feature'] if len(feature_imp) > 2 else 'N/A'}`**: 세 번째 중요 피처

---

## 7. 저장된 파일

| 파일 | 경로 | 설명 |
|------|------|------|
| Logistic Regression | `models/logistic_regression.pkl` | Baseline 모델 |
| LightGBM | `models/lightgbm.txt` | Tree 모델 |
| Scaler | `models/scaler.pkl` | 표준화 스케일러 |
| Feature 목록 | `models/feature_cols.json` | 학습 피처 목록 |
| 결과 JSON | `models/training_results.json` | 전체 결과 |

---

> **다음 단계**: 딥러닝 모델 학습 또는 Risk Score 생성
"""
    
    # 저장
    report_path = report_dir / '01_ml_training_results.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 리포트 저장: {report_path}")


# ============================================
# 메인 파이프라인
# ============================================
def run_training_pipeline(data_dir: Optional[Path] = None,
                          model_dir: Optional[Path] = None,
                          report_dir: Optional[Path] = None) -> Dict:
    """
    전체 학습 파이프라인을 실행합니다.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if model_dir is None:
        model_dir = MODEL_DIR
    if report_dir is None:
        report_dir = REPORT_DIR
    
    print("=" * 60)
    print("🚀 KKBox Churn Prediction - Model Training Pipeline")
    print("=" * 60)
    
    # 1. 데이터 로드
    train, valid, test = load_datasets(data_dir)
    
    # 2. 피처 준비
    X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols = prepare_features(
        train, valid, test
    )
    
    # 3. Logistic Regression 학습
    lr_model, scaler, lr_results = train_logistic_regression(
        X_train, y_train, X_valid, y_valid
    )
    
    # 4. LightGBM 학습
    lgb_model, lgb_results = train_lightgbm(
        X_train, y_train, X_valid, y_valid
    )
    
    # 5. 테스트셋 평가
    models = {
        'Logistic Regression': lr_model,
        'LightGBM': lgb_model
    }
    test_results = evaluate_on_test(models, X_test, y_test, scaler)
    
    # 결과 통합
    lr_results['test_metrics'] = test_results['Logistic Regression']
    lgb_results['test_metrics'] = test_results['LightGBM']
    
    all_results = {
        'Logistic Regression': lr_results,
        'LightGBM': lgb_results
    }
    
    # 6. 결과 저장
    save_results(all_results, models, scaler, model_dir, feature_cols)
    
    # 7. 리포트 생성
    generate_report(all_results, report_dir)
    
    print("\n" + "=" * 60)
    print("✅ 학습 파이프라인 완료!")
    print("=" * 60)
    
    return all_results


# ============================================
# 실행
# ============================================
if __name__ == "__main__":
    results = run_training_pipeline()

