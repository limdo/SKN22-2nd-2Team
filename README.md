# KKBox 구독 이탈 관리 시스템 (Churn Control Center)

## 프로젝트 개요
본 프로젝트는 **KKBox** 음원 스트리밍 서비스의 사용자 이탈을 방지하고 구독 유지를 관리하기 위해 개발된 **이탈 예측 및 관리 시스템**입니다.
데이터 전처리, 파생 변수 생성, 머신러닝 모델링 과정을 거쳐 이탈 가능성이 높은 사용자를 조기에 식별하고, Streamlit 기반의 대시보드를 통해 비즈니스 인사이트를 제공합니다.


## SKN22-2nd-2Team "에용"
- 안민제, 임도형, 이규빈, 이도훈, 김희준
- 25.12.22

## 사용 기술 (Tech Stack)

### Languages & Libraries
<div align="left">
  <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/CatBoost-E9711C?style=for-the-badge&logo=CatBoost&logoColor=white">
  <img src="https://img.shields.io/badge/Optuna-5E87F5?style=for-the-badge&logo=lightning&logoColor=white">
</div>

### Dashboard & Visualization
<div align="left">
  <img src="https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white">
  <img src="https://img.shields.io/badge/seaborn-76B900?style=for-the-badge&logo=seaborn&logoColor=white">
  <img src="https://img.shields.io/badge/matplotlib-0B579E?style=for-the-badge&logo=matplotlib&logoColor=white">
</div>

### Development Environment
<div align="left">
  <img src="https://img.shields.io/badge/Visual%20Studio%20Code-007ACC?style=for-the-badge&logo=Visual%20Studio%20Code&logoColor=white">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white">
</div>

 **마케팅 시뮬레이터 (Marketing Simulator)**: 타겟팅 범위에 따른 기대 효과(Lift)와 비용을 **실시간으로 시뮬레이션**하여 최적의 마케팅 전략 수립을 지원합니다.

---

## 서비스 로직 및 페이지 구성 (Service Logic & Pages)
본 프로젝트는 **Streamlit**을 활용하여 데이터 분석 결과를 시각화하고 타겟팅 전략을 수립할 수 있는 대시보드를 제공합니다. (상세 명세: [docs/page_specification.md](docs/page_specification.md))

| 페이지 (Page) | 핵심 기능 (Key Function) | 비즈니스 가치 (Business Value) |
| :--- | :--- | :--- |
| **0. Home (Overview)** | 데이터 분석 범위 및 핵심 파생 변수 명세 정의 | 예측의 신뢰도 및 투명성 확보 |
| **2. Model Guideline** | 이탈 시그널 사전 및 마케팅 개입 지도 제공 | 위험 징후에 대한 명확한 판단 기준 제시 |
| **3. Model Explainability** | Two-Track 모델 전략 설명 및 행동 편차(Z-Score) 분석 | 모델의 "Why"를 설명하여 비즈니스 설득력 제고 |
| **4. Risk Matrix** | 이력(Status) vs 행동(Behavior) 4분면 세그먼트 분석 | 단순 이탈 확률을 넘어선 입체적 고객 분류 |
| **5. Marketing Simulator** | 실시간 타겟팅 시뮬레이션 및 자동 처방(Auto-Prescription) | 마케팅 ROI 최적화 및 실행 가능한 액션 플랜 제공 |

---

## 분석 및 개발 프로세스 (Process)

### 1. 데이터 전처리 파이프라인 (Preprocessing Pipeline V1~V4)
모델 성능과 학습 효율성을 극대화하기 위해 4단계의 체계적인 전처리 파이프라인을 구축했습니다.

#### **V1: 대용량 데이터 최적화 (Optimization & Aggregation)**
- **참고**: `notebooks/preprocessing/optimize_user_logs_aggregated.ipynb`
- **목적**: 3억 건 이상의 `user_logs` 데이터를 로컬 환경에서 처리 가능하도록 경량화.
- **내용**: 
    - 데이터 타입 다운캐스팅 (Float64 → Float32, Int64 → UInt)으로 메모리 용량 **70% 절감**.
    - 일별 로그 데이터를 주간(W7, W14, W30) 단위로 집계(Aggregating)하여 피처 차원 축소.

#### **V2: 비즈니스 로직 정제 (Leakage-Safe Transactions)**
- **참고**: `notebooks/preprocessing/transactions_v2_aggregation.ipynb`
- **목적**: 복잡한 결제 이력(`transactions`)에서 유효한 구독 상태를 정의하고 **Time-Series Leakage** 방지.
- **내용**: 
    - **Payment/Cancel 정의**: 실제 지불 금액 > 0인 경우만 결제로 인정.
    - **Cutoff Date 적용**: 예측 시점(2017-04-01) 이전의 데이터만 사용하여 미래 정보 참조 방지.

#### **V3: 통합 학습 테이블 구축 (Wide Table Construction)**
- **참고**: `notebooks/preprocessing/build_train_feature_table.ipynb`
- **목적**: 모델에 주입할 수 있는 형태의 단일 Base Table 생성.
- **내용**: 
    - Members + Transactions(V2) + User Logs(V1) 을 **msno** 기준 INNER JOIN.
    - 결측치(NaN)에 대한 전략적 대체 (예: 로그 없음 → 활동 없음으로 간주하여 0 처리).

#### **V4: 전략적 파생 변수 (Strategic Feature Engineering)**
- **참고**: `notebooks/preprocessing/build_train_feature_v4.ipynb`
- **목적**: 단순 집계 변수(V3)를 넘어선 유저의 **행동 변화**와 **속성** 포착.
- **내용**:
    - **Trend Features**: 활동 감소율(`active_decay_rate`), 청취 가속도(`listening_velocity`) 등 시계열 변화량 계산.
    - **Context Features**: 스킵 성향(`skip_passion_index`), 탐색 지수(`discovery_index`) 등 유저 행동 패턴 정량화.
    - **Filtering**: 변수 중요도(Feature Importance) 및 결측 비율에 따른 불필요한 컬럼 제거.

### 2. 모델 고도화 전략 (Modeling Strategy V4 & V5.2)
- **V5.2 (Safe Context)**: 결제 상태(Status) 정보를 배제하고, 유저의 성향(Context)과 순수 행동 패턴에 집중하여 과적합을 방지하고 조기 경보 능력을 강화했습니다.

### 3. 모델 선정을 위한 실험 (Model Selection Experiments)
- **실험 계획**: `02_training_report/model_experiment_plan.md` (Feature Ablation)
- **실험 단계 및 팩트 요약**:
    - **e0 (Baseline)**: 전체 컬럼 사용. 모델이 달성 가능한 이론적 최대 성능 확인.
    - **e1 (Leakage-Safe)**: `days_since_last_payment` 등 정답과 직결되는 상태 변수 제거. 미래 정보 의존성 검증.
    - **e2 (Logs-only)**: 결제(Transaction) 정보를 모두 제거하고 오직 행동 로그만 사용. 순수 행동 기반 예측력 확인.
    - **e2.1 (Conservative)**: e2에 Leakage 변수까지 제거. 가장 보수적인 정보 환경에서의 성능 하한선 확인.
    - **e3 (Short-term)**: 장기(w30) 윈도우 제거. 단기 행동 정보만으로의 예측 가능성 검증.
    - **e3.1 (Robustness)**: 최악의 조건(w30 Off + Logs-only). 시스템 악조건에서도 견딜 수 있는 모델의 Robustness 검증.
- **결과**: CatBoost는 정보가 제거됨에 따라 성능이 완만하게 하락(Gradient Drop)하여 가장 강건한(Robust) 모델로 선정됨. (XGBoost는 e2에서 급락)

#### 실험 결과 요약 (Experiment Summary)
> 모델별 성능 변화 (Delta vs Baseline e0)

![Experiment Summary](docs/images/experiment_summary.png)

위 그림은 Baseline(e0) 대비 각 모델의 실험 단계별 성능 변화(평균)를 보여줍니다. CatBoost는 전반적으로 안정적인 성능 유지(Accuracy, Recall)와 높은 재현율을 보여 최종 모델로 선정되었습니다.

### 4. 하이퍼파라미터 튜닝 (Hyperparameter Tuning with Optuna)
- **도구**: **Optuna** (Bayesian Optimization Framework)
- **목적**: 불균형 데이터(Imbalanced Data)에서 이탈자(Class 1)를 잘 찾아내기 위한 **F1-Score 극대화**.
- **튜닝 절차**:
    1. **Search Space 설정**: `learning_rate` (Log scale), `depth` (4~10), `l2_leaf_reg` (Regularization), `random_strength` (Noise for robustness) 등.
    2. **Objective Function**: `scale_pos_weight` 등 클래스 가중치를 적용한 후, 검증 셋(Validation Set)의 F1-Score를 최적화 목표로 설정.
    3. **Best Params 선정**: 약 2,000~3,000회의 Iteration 내에서 조기 종료(Early Stopping)를 적용하여 최적 파라미터 탐색.
    4. **결과 적용**: V4 및 V5.2 모델 각각에 대해 별도의 최적 파라미터 세트를 도출하여 적용 (Metadata JSON 저장).

### 5. 모델 학습 및 운영 전략 (Final Model Training & Strategy)
- 정의된 파생변수(Feature)를 기반으로, **user_logs의 행동 데이터를 비중 있게 반영**하기 위해 두 가지 트랙으로 모델을 학습시켰습니다.

| 모델 | 정의 (Definition) | 활용 (Usage) |
| :--- | :--- | :--- |
| **V4 (Main)** | **History + Behavior + Status**. 모든 정보를 활용하여 현재 이탈 확률을 정밀 예측. | **확정적 이탈 방어**. 이미 갱신을 해지했거나 징후가 뚜렷한 고위험군에게 즉각적인 오퍼(쿠폰 등) 제공. (High Precision) |
| **V5.2 (Behavior)** | **History + Behavior + Trends**. 결제 상태(Status)를 **강제로 가리고(Masking)** 학습. | **잠재적 이탈 탐지**. 구독은 유지 중이나 활동이 급감하는 등 '마음이 떠난' 유저를 조기에 포착하여 Engagement 유도. (Early Warning) |

> **💡 왜 두 개의 모델을 쓰나요? (Two-Track Strategy)**
> 결제 만료가 임박하거나 해지한 사용자(Active Churn)뿐만 아니라, **아직 돈은 내고 있지만 마음은 떠난 사용자(Silent Churn)를 놓치지 않기 위함**입니다. 이 두 모델의 시너지를 통해 빈틈없는 이탈 관리가 가능합니다.

## 📊 데이터 스키마 (Data Schema & Feature Summary)
모델 학습에 활용된 핵심 파생 변수(Derived Features) 요약입니다. (V4 Dataset 기준)

### 1. 이력 및 환경 변수 (History & Environment) - V4 Key Features
| 변수명 (Feature) | 설명 (Description) | 비즈니스 의미 |
| :--- | :--- | :--- |
| `days_since_last_cancel` | 최근 취소 경과일 | "이전에 해지한 적이 있는가?" (습관적 이탈) |
| `days_since_last_payment` | 결제 공백기 | "마지막 결제로부터 며칠이 지났는가?" (이탈 임박) |
| `is_auto_renew_last` | 자동 갱신 여부 | "자동 갱신을 켜두었는가?" (가장 강력한 방어선) |
| `subscription_months` | 구독 유지 기간 | "얼마나 오래 사용한 충성 고객인가?" |

### 2. 행동 변수 (User Behavior) - V5.2 Key Features
| 변수명 (Feature) | 설명 (Description) | 비즈니스 의미 |
| :--- | :--- | :--- |
| `active_decay_rate` | 활동 감소율 | "평소(30일) 대비 최근(7일) 접속이 얼마나 줄었는가?" |
| `listening_velocity` | 청취 가속도 | "최근 2주간 청취 시간이 급격히 줄어들고 있는가?" |
| `skip_passion_index` | 스킵 열정도 | "곡을 듣지 않고 넘기는 비율이 비정상적으로 높은가?" |
| `last_active_gap` | 마지막 활동 경과일 | "구독은 되어 있는데, 접속을 안 한지 며칠째인가?" |

### 5. 대시보드 구축 (Interactive Dashboard)
- **Streamlit**을 활용하여 예측 결과와 주요 지표를 시각화
- **주요 기능**:
    - Model Guideline
    - Model Explainability (Z-score 분석 등 V5.2 주요 변수 해석)
    - Risk Matrix
    - Marketing Simulator

---

## 프로젝트 구조 (Directory Structure)

```bash
SKN22-2nd-2Team
├── app.py                      # Streamlit Main App
├── requirements.txt            # Project Dependencies
├── data/                       # Data (Raw & Processed, Ignored in Git)
├── notebooks/                  # Jupyter Notebooks
│   ├── eda/                    # Exploratory Data Analysis
│   ├── preprocessing/          # Data Preprocessing & Feature Engineering
│   └── modeling/               # Model Training & Experiments
├── src/                        # Source Code
│   ├── preprocessing/          # Preprocessing Logic Modules
│   └── modeling/               # Model Pipeline Modules
├── pages/                      # Streamlit Pages
│   ├── 2_Model_Guideline.py
│   ├── 3_Model_Explainability.py
│   ├── 4_Risk_Matrix.py
│   └── 5_Marketing_Simulator.py
├── 01_preprocessing_report/    # Preprocessing Reports
├── 02_training_report/         # Model Training Reports
└── 03_trained_model/           # Model Artifacts & Reports
```

---

## 설치 및 실행 (Installation & Usage)

### 1. 요구 사항 (Requirements)
본 프로젝트는 Python 3.8+ 환경에서 실행을 권장합니다.
필요한 패키지는 `requirements.txt`에 명시되어 있습니다.

```bash
pip install -r requirements.txt
```

### 2. 실행 방법 (Usage)
Streamlit 대시보드를 실행하여 시스템을 확인할 수 있습니다.

```bash
streamlit run app.py
```

---