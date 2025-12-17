# 02. ML 모델 학습 결과 - kkbox_train_feature_v1.parquet

> **작성자**: 이도훈 (LDH)  
> **작성일**: 2025-12-17  
> **피처 테이블**: `kkbox_train_feature_v1.parquet`

---

## 1. 학습 개요

### 1.1 사용 데이터
- 단일 피처 테이블: `kkbox_train_feature_v1.parquet` (전처리 완료)
- 샘플 수: 약 860,966명
- 피처 수: 87개

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
| **ROC-AUC** | 0.9873 | 0.9886 | CatBoost ✅ |
| **PR-AUC** | 0.9177 | 0.9295 | CatBoost ✅ |
| **Recall** | 0.9181 | 0.9324 | CatBoost ✅ |
| **Precision** | 0.7340 | 0.7024 | LightGBM ✅ |
| **F1-Score** | 0.8158 | 0.8012 | LightGBM ✅ |

### 2.2 Test Set 성능 (최종)

| 지표 | LightGBM | CatBoost | 우수 모델 |
|------|----------|----------|-----------|
| **ROC-AUC** | 0.9878 | 0.9892 | CatBoost ✅ |
| **PR-AUC** | 0.9219 | 0.9334 | CatBoost ✅ |
| **Recall** | 0.9195 | 0.9335 | CatBoost ✅ |
| **Precision** | 0.7411 | 0.7068 | LightGBM ✅ |
| **F1-Score** | 0.8207 | 0.8045 | LightGBM ✅ |

---

## 3. Confusion Matrix (Test Set)

### 3.1 LightGBM

```
              Predicted
              0        1
Actual  0    150,672    5,232
        1    1,311    14,979
```

### 3.2 CatBoost

```
              Predicted
              0        1
Actual  0    149,595    6,309
        1    1,083    15,207
```

---

## 4. Feature Importance 비교

### 4.1 LightGBM Top 10

| 순위 | Feature | Importance |
|------|---------|------------|
| 1 | `days_since_last_cancel` | 3772335.45 |
| 2 | `days_since_last_payment` | 1049580.38 |
| 3 | `total_amount_paid` | 740518.87 |
| 4 | `subscription_months_est` | 493696.55 |
| 5 | `is_auto_renew_last` | 465360.93 |
| 6 | `has_ever_cancelled` | 309679.03 |
| 7 | `last_plan_days` | 280369.97 |
| 8 | `payment_count_last_90d` | 230778.69 |
| 9 | `last_payment_method` | 186489.62 |
| 10 | `total_payment_count` | 169425.91 |


### 4.2 CatBoost Top 10

| 순위 | Feature | Importance |
|------|---------|------------|
| 1 | `days_since_last_cancel` | 24.08 |
| 2 | `days_since_last_payment` | 17.65 |
| 3 | `total_amount_paid` | 9.89 |
| 4 | `subscription_months_est` | 9.44 |
| 5 | `payment_count_last_90d` | 9.00 |
| 6 | `last_payment_method` | 6.71 |
| 7 | `registration_month` | 5.01 |
| 8 | `is_auto_renew_last` | 3.50 |
| 9 | `avg_amount_per_payment` | 2.77 |
| 10 | `payment_count_last_30d` | 2.75 |

---

## 5. 모델별 하이퍼파라미터

### 5.1 LightGBM

- num_leaves: 31
- max_depth: 6
- learning_rate: 0.05
- feature_fraction: 0.8
- bagging_fraction: 0.8
- min_child_samples: 100
- reg_alpha: 0.1
- reg_lambda: 0.1
- best_iteration: 46

### 5.2 CatBoost

- depth: 6
- learning_rate: 0.05
- l2_leaf_reg: 3.0
- iterations: 500
- scale_pos_weight: 9.57083472190553

---

## 6. 결론

- **최종 추천 모델**: CatBoost
- **핵심 지표 (ROC-AUC 기준)**: 최대 0.9892
- **Confusion Matrix 기준**으로 두 모델 모두 이탈자(1)에 대한 높은 Recall을 달성.

---

> 이 리포트는 `kkbox_train_feature_v1.parquet` 기반 LightGBM / CatBoost 실험 결과입니다.
