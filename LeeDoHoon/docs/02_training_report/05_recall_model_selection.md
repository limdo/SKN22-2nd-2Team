# 05. Recall 최적화 모델 선택 결과

> **작성자**: 이도훈 (LDH)  
> **작성일**: 2025-12-18  
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
| 피처 수 | 35 |
| 제거된 중복 피처 | recency_secs_ratio, recency_songs_ratio |

---

## 2. 모델 비교 (Validation Set)

### 2.1 Recall 비교 (기본 Threshold 0.5)

| 모델 | Recall | ROC-AUC | 선택 |
|------|--------|---------|------|
| **LightGBM** | 0.8896 | 0.9831 |  |
| **CatBoost** | 0.9634 | 0.9881 | ✅ 선택 |

### 2.2 선택된 모델
- **선택**: CatBoost
- **선택 사유**: Recall이 더 높음 (0.9634 > 0.8896, +7.4%p)

---

## 3. Threshold 최적화

### 3.1 최적 Threshold 선정
| 항목 | 기본값 (0.50) | 최적값 (0.58) | 변화 |
|------|---------------|---------------|------|
| **Recall** | 0.9675 | 0.9514 | -1.61%p |
| **Precision** | 0.5283 | 0.5818 | +5.35%p |
| **F1-Score** | 0.6834 | 0.7220 | +3.86%p |

---

## 4. Test Set 최종 성능

### 4.1 성능 지표 (최적 Threshold)

| 지표 | 값 |
|------|-----|
| **ROC-AUC** | 0.9883 |
| **PR-AUC** | 0.9222 |
| **Recall** | **0.9514** |
| **Precision** | 0.5818 |
| **F1-Score** | 0.7220 |
| **Specificity** | 0.9324 |

### 4.2 Confusion Matrix (최적 Threshold)

```
              Predicted
              0        1
Actual  0    164,779    11,947
        1    848    16,618
```

### 4.3 성능 해석
- **탐지된 이탈자**: 16,618명 / 17,466명 (Recall: 95.1%)
- **놓친 이탈자**: 848명 (False Negative)
- **잘못 예측된 유지 고객**: 11,947명 (False Positive)

---

## 5. Feature Importance (Top 20)

| 순위 | Feature | Importance |
|------|---------|------------|
| 1 | `days_to_expire` | 23.24 |
| 2 | `payment_method_last` | 14.88 |
| 3 | `cancel_count` | 11.91 |
| 4 | `total_payment` | 11.55 |
| 5 | `auto_renew_rate` | 10.02 |
| 6 | `has_cancelled` | 6.74 |
| 7 | `tenure_days` | 2.94 |
| 8 | `avg_payment` | 2.90 |
| 9 | `transaction_count` | 2.74 |
| 10 | `avg_list_price` | 2.72 |
| 11 | `avg_discount_rate` | 2.11 |
| 12 | `is_auto_renew_last` | 1.68 |
| 13 | `active_days` | 1.00 |
| 14 | `plan_days_last` | 0.71 |
| 15 | `num_985_sum` | 0.48 |
| 16 | `skip_ratio` | 0.46 |
| 17 | `listening_variety` | 0.45 |
| 18 | `city` | 0.36 |
| 19 | `age` | 0.33 |
| 20 | `total_secs` | 0.28 |

---

## 6. CatBoost 하이퍼파라미터

| 파라미터 | 값 |
|----------|-----|
| loss_function | Logloss |
| eval_metric | AUC |
| learning_rate | 0.05 |
| depth | 6 |
| l2_leaf_reg | 3.0 |
| iterations | 500 |
| early_stopping_rounds | 50 |
| scale_pos_weight | 15.18 |
| best_iteration | 493 |

---

## 7. 결론

### 7.1 최종 결과 요약

| 항목 | 값 |
|------|-----|
| **선택된 모델** | CatBoost |
| **최적 Threshold** | 0.58 |
| **최종 Recall** | **95.1%** |
| **ROC-AUC** | 0.9883 |

### 7.2 비즈니스 활용
1. **높은 Recall**: 이탈 가능 고객의 95.1%를 사전 식별
2. **선제적 대응**: 식별된 고객에게 리텐션 캠페인 진행
3. **Trade-off**: Precision이 낮아 False Positive 증가 → 마케팅 비용 고려 필요

---

## 8. 저장된 파일

| 파일 | 경로 | 설명 |
|------|------|------|
| 모델 | `models/catboost_recall_selected.cbm` | Recall 최적화 모델 |
| 결과 JSON | `models/recall_selected_results.json` | 전체 결과 |
| 리포트 | `docs/02_training_report/05_recall_model_selection.md` | 이 문서 |

---

> **핵심 메시지**: CatBoost 모델을 Threshold 0.58로 설정하면 이탈 고객의 **95.1%**를 탐지할 수 있습니다.

