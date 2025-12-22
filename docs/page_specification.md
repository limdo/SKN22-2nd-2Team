# 서비스 페이지 명세 (Service Page Specification)

본 문서는 `app.py` 및 `pages/` 디렉토리의 스트림릿(Streamlit) 애플리케이션 구조와 각 페이지의 핵심 기능을 기술합니다.

## 1. 개요 (Overview) - `app.py`
**목적**: 프로젝트의 데이터 분석 범위(Boundary)와 핵심 파생 변수(Feature Spec)를 정의하여 분석의 신뢰도를 확보합니다.

### 주요 구성 요소
*   **Data Boundary**:
    *   **기준 시점 (Target Date)**: 2017-04-01 (예측 기준일 T).
    *   **이탈 정의 (Churn)**: 만료 후 30일 미결제.
    *   **이력 집계 (History)**: 가입 시점 ~ T (전체 라이프사이클).
    *   **행동 집계 (Behavior)**: 2017-03-01 ~ 03-31 (T 기준 과거 30일).
*   **Feature Specification**:
    *   비즈니스 가치가 검증된 핵심 파생 변수 (`active_decay_rate`, `listening_velocity`, `skip_passion_index` 등)의 정의와 가치를 테이블로 제시.

---

## 2. 이탈 예측 가이드라인 (Model Guideline) - `pages/2_Model_Guideline.py`
**목적**: 모델이 유저를 진단하는 로직을 시각화하고, 마케팅 액션의 기준(Signal)을 가이드합니다.

### 주요 구성 요소
*   **Interactive Persona**:
    *   체크박스 시뮬레이션: 유저의 행동(접속 공백, 스킵 급증, 갱신 임박 등)에 따라 위험 점수가 어떻게 변하는지 실시간으로 체험.
    *   점수 구간별 상태 진단 (Safe, Watch-out, Warning, Danger).
*   **Signal Dictionary (신호등)**:
    *   각 위험 지대별 판단 기준(Metric Threshold)과 권장 액션(Action) 정의.
    *   예: **Danger** (Active Decay < 0.5 → 쿠폰 지급), **Warning** (Velocity 하락 → 결제수단 업데이트).
*   **Actionability Map**:
    *   **X축**: 모델 중요도 (Contribution).
    *   **Y축**: 마케팅 개입 가능성 (Actionability).
    *   인사이트: **Focus Zone** (중요하고 바꿀 수 있음) vs **Filtering Zone** (중요하지만 바꿀 수 없음) 구분.

---

## 3. 모델 상세 설명 (Model Explainability) - `pages/3_Model_Explainability.py`
**목적**: Two-Track 모델의 전략적 차이와 각 모델이 포착하는 핵심 요인을 설명합니다 (XAI).

### 주요 구성 요소
*   **Two-Track Strategy**:
    *   **V4 (이력 중심)**: 과거 상태(Status) 기반, 환경적 고위험군 선별.
    *   **V5.2 (행동 중심)**: 최근 심리(Sentiment) 기반, 행동 징후 포착.
    *   **Synergy**: V4로 좁히고 V5.2로 핀셋 타겟팅.
*   **Z-Score Analysis**:
    *   이탈자(Churner)와 일반 유저 간의 행동 데이터 편차를 표준화(Standardized)하여 시각화.
    *   주요 지표: `active_decay_rate`, `secs_trend`, `engagement_density` 등.
*   **Feature Importance**:
    *   V4와 V5.2 모델 각각의 상위 중요 변수(Top 10)와 비즈니스 의미(Description), 계산식(Formula) 제공.

---

## 4. 위험도 매트릭스 (Risk Matrix) - `pages/4_Risk_Matrix.py`
**목적**: V4(상태)와 V5.2(행동) 점수를 축으로 하는 4분면 매트릭스를 통해 입체적인 유저 세그멘테이션을 수행합니다.

### 주요 구성 요소
*   **4분면 매트릭스 (Scatter Plot)**:
    *   **X축**: 행동 위험도 (V5.2).
    *   **Y축**: 이력 위험도 (V4).
    *   **Quadrants**:
        1.  **Safe (안전)**: V4 Low, V5.2 Low.
        2.  **Watch-out (주의)**: V4 Low, V5.2 High (결제는 하나 행동 급감).
        3.  **Warning (경보)**: V4 High, V5.2 Low (행동은 있으나 결제 불안).
        4.  **Danger (위험)**: V4 High, V5.2 High (이탈 확정적).
*   **전략적 그룹 정의**: 각 세그먼트별 상태 정의 및 대응 전략 카드.
*   **인사이트**: 전체 유저 중 Danger 그룹의 비중(%) 및 규모 파악.

---

## 5. 마케팅 시뮬레이터 (Marketing Simulator) - `pages/5_Marketing_Simulator.py`
**목적**: 예산 및 전략에 따라 타겟팅 범위를 조절하고, 그에 따른 기대 효과와 비용을 시뮬레이션합니다.

### 주요 구성 요소
*   **Targeting Control (Real-time)**:
    *   **Scope Slider**: 이탈 위험 상위 N% 타겟팅 설정.
    *   **Sensitivity Slider**: 위험 민감도 조절 (임계값 변경).
    *   **KPIs**: 선택된 타겟 유저 수, 평균 위험도(Risk) 증가분(Lift).
*   **Highlighter**: 전체 유저 분포(Scatter) 중 선택된 타겟 유저를 붉은색으로 시각화.
*   **Benchmarking (Lift Analysis)**:
    *   **Target vs Normal**: 활동 감소율, 청취 변화 등 핵심 지표 비교 바 차트.
    *   선택된 타겟 그룹이 일반 유저 대비 얼마나 위험한 행동을 보이는지 정량적 검증.
*   **Auto-Prescription**:
    *   선택된 타겟 내의 세그먼트 구성비(Pie Chart).
    *   각 세그먼트별 맞춤형 자동 대응 가이드 (예: Danger → 쿠폰, Watch-out → 콘텐츠 푸시).
*   **Export**: 타겟 유저 리스트 CSV 다운로드.
