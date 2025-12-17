# Members + Train EDA Report

## 1. 목적
본 문서는 KKBox Churn Prediction 프로젝트에서 `members_v3.csv`와 `train_v2.csv`를 결합하여
모델 학습을 위한 베이스 테이블을 구성하기 전 수행한 EDA 결과를 정리한 보고서이다.

본 EDA의 목표는 다음과 같다.
- 데이터 누수 없는 feature 설계 방향 확정
- 결측치 및 이상치 처리 전략 수립
- 모델 학습에 적합한 파생 변수 정의

---

## 2. 데이터 구성

### 사용 데이터
- `members_v3.csv`: 회원 기본 정보
- `train_v2.csv`: churn 레이블 (`is_churn`)

### 조인 전략
- 기준 테이블: `members`
- 조인 방식: `LEFT JOIN (members ⟵ train)`
- 조인 키: `msno`

조인 결과, 다수의 사용자는 `is_churn` 라벨이 없는 상태로 존재하며,
이는 결측이 아닌 **추론 대상 사용자**로 간주한다.

---

## 3. 타겟 변수 분포

`is_churn` 분포는 다음과 같다.

- churn = 1 : 약 8만 명
- churn = 0 : 약 77만 명
- churn = NaN : 약 590만 명

NaN 값은 학습 단계에서만 제외하며, 데이터셋에서는 유지한다.

---

## 4. 주요 컬럼별 EDA 및 처리 전략

### 4.1 gender
- 결측치는 `unknown`으로 유지
- churn 분리력이 매우 약함
- 보조 feature로 유지하거나 모델 단계에서 제거 가능

### 4.2 city
- 매우 강한 롱테일 분포
- 일부 city 코드에 사용자 집중
- 선형/로그 스케일 분포 시각화 수행
- 희소 city는 추후 모델 단계에서 통합 여부 판단

### 4.3 registered_via
- 가입 경로 코드
- city와 유사한 롱테일 구조
- 분포 시각화만 수행, 추가 전처리는 모델 단계에서 결정

---

## 5. bd (연령) 컬럼 분석

### 문제점
- 음수, 0, 수백~수천 값 다수 존재
- 출생년도(YYYY)로 입력된 값 혼재

### 처리 전략
1. 1900~2017 범위 값은 출생년도로 간주하여 나이로 변환
2. 변환 후 10세 미만 또는 80세 초과 값은 결측 처리
3. 결측 여부를 나타내는 파생 변수 `bd_missing` 생성
4. 정제된 연령은 `bd_clean`으로 유지

이 방식은 정보 손실을 최소화하면서 데이터 신뢰도를 확보한다.

---

## 6. registration_init_time

- yyyymmdd 형식의 가입일 정보
- datetime으로 변환 후 직접 사용하지 않음

### 파생 변수
- `days_since_registration` (예측 시점 T 기준)
- 가입 후 경과 기간을 나타내는 핵심 연속형 feature

원본 날짜 컬럼은 모델 입력 단계에서 제거한다.

---

## 7. 결론 및 다음 단계

### EDA 결론
- members 데이터에 대한 이상치/결측 처리 전략 확정
- 의미 있는 파생 변수 정의 완료
- train 레이블 결합을 통한 베이스 테이블 완성

### 다음 단계
- user_logs_v2: 관측 윈도우 기반 행동 로그 집계
- transactions_v2: 구독/결제 상태 집계
- 통합 feature 구성 후 모델 학습 진행

---

## 부록
본 EDA 결과를 반영한 데이터는 다음 파일로 저장됨.

- `members_with_label_base.parquet`
- `members_with_label_base_opt.parquet`
