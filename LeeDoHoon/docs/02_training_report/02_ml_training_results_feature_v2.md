# 02. ML 모델 학습 결과 - kkbox_train_feature_v1.parquet (feature_v2)

> **작성자**: 이도훈 (LDH)  
> **작성일**: 2025-12-17  
> **피처 테이블**: `kkbox_train_feature_v1.parquet`  
> **전처리 버전**: feature_v2 (잠재적 누수 피처 제거)

---

## 1. 학습 개요

### 1.1 사용 데이터
- 단일 피처 테이블: `kkbox_train_feature_v1.parquet`
- 샘플 수: 약 860,966명
- 피처 수: 72개
- 잠재적 누수 피처 제거 개수: 16개

- 제거된 누수 후보 피처 예시: `registration_init_time`, `days_since_last_payment`, `days_since_last_cancel`, `is_auto_renew_last`, `last_plan_days`

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
| **ROC-AUC** | 0.9784 | 0.9801 | CatBoost ✅ |
| **PR-AUC** | 0.8940 | 0.9060 | CatBoost ✅ |
| **Recall** | 0.8902 | 0.9063 | CatBoost ✅ |
| **Precision** | 0.7416 | 0.6850 | LightGBM ✅ |
| **F1-Score** | 0.8092 | 0.7803 | LightGBM ✅ |


### 2.2 Test Set 성능 (최종)

| 지표 | LightGBM | CatBoost | 우수 모델 |
|------|----------|----------|-----------|
| **ROC-AUC** | 0.9793 | 0.9810 | CatBoost ✅ |
| **PR-AUC** | 0.8961 | 0.9089 | CatBoost ✅ |
| **Recall** | 0.8881 | 0.9052 | CatBoost ✅ |
| **Precision** | 0.7453 | 0.6864 | LightGBM ✅ |
| **F1-Score** | 0.8104 | 0.7807 | LightGBM ✅ |


---

## 3. Confusion Matrix (Test Set)

### 3.1 LightGBM

```
             Predicted
             0        1
Actual  0    150,959    4,945
        1     1,823    14,467
```

### 3.2 CatBoost

```
             Predicted
             0        1
Actual  0    149,167    6,737
        1     1,545    14,745
```

---

## 4. Feature Importance 비교

### 4.1 LightGBM Top 10

| 순위 | Feature | Importance |
|------|---------|------------|
| 1 | `has_ever_cancelled` | 2077993.10 |
| 2 | `payment_count_last_90d` | 1915468.00 |
| 3 | `payment_count_last_30d` | 1224244.32 |
| 4 | `total_amount_paid` | 607536.84 |
| 5 | `subscription_months_est` | 590967.40 |
| 6 | `avg_amount_per_payment` | 181738.12 |
| 7 | `total_secs_w7` | 103940.87 |
| 8 | `city` | 91623.11 |
| 9 | `registration_month` | 83727.71 |
| 10 | `registered_via` | 78316.38 |


### 4.2 CatBoost Top 10

| 순위 | Feature | Importance |
|------|---------|------------|
| 1 | `has_ever_cancelled` | 24.82 |
| 2 | `payment_count_last_90d` | 18.50 |
| 3 | `total_amount_paid` | 10.55 |
| 4 | `payment_count_last_30d` | 7.77 |
| 5 | `subscription_months_est` | 6.63 |
| 6 | `avg_amount_per_payment` | 6.61 |
| 7 | `registration_month` | 6.41 |
| 8 | `registered_via` | 5.77 |
| 9 | `has_ever_paid` | 2.96 |
| 10 | `city` | 0.90 |

---

## 5. 결론

- **전처리 버전**: feature_v2 (잠재적 누수 피처 제거)
- **최종 추천 모델**: CatBoost
- **핵심 지표 (ROC-AUC 기준)**: 최대 0.9810
- v1 대비 성능 차이는 별도 비교 표에서 함께 검토 필요.

---

> 이 리포트는 `kkbox_train_feature_v1.parquet` 기반 feature_v2 (누수 피처 제거) 설정에서의 LightGBM / CatBoost 실험 결과입니다.
