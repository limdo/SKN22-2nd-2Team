"""
KKBox Churn Prediction - Data Preprocessing & Feature Engineering
작성자: 이도훈 (LDH)
작성일: 2025-12-16

이 모듈은 KKBox 데이터셋의 전처리 및 피처 엔지니어링을 수행합니다.
EPIC 1에서 정의한 원칙을 준수합니다:
- 예측 시점 (T): 2017-04-01
- 관측 윈도우: 2017-03-01 ~ 2017-03-31 (30일)
- 미래 정보 누수 금지
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# ============================================
# 설정
# ============================================
PREDICTION_TIME = pd.Timestamp('2017-04-01')  # 예측 시점 T
OBSERVATION_START = pd.Timestamp('2017-03-01')
OBSERVATION_END = pd.Timestamp('2017-03-31')

# 데이터 경로
DATA_DIR = Path(__file__).parent.parent / 'data'

# 결측치/이상치 처리 규칙
PREPROCESSING_RULES = {
    'age': {
        'type': '이상치 처리',
        'rule': '0 < age < 100 범위 외 → 중앙값 대체',
        'reason': '비현실적인 나이값 (음수, 0, 100세 이상) 제거'
    },
    'gender': {
        'type': '결측치 처리',
        'rule': 'NaN → "unknown"',
        'reason': '성별 미입력 사용자 별도 범주로 처리'
    },
    'city': {
        'type': '결측치 처리',
        'rule': 'NaN → 0',
        'reason': '도시 미입력을 0으로 처리'
    },
    'registered_via': {
        'type': '결측치 처리',
        'rule': 'NaN → 0',
        'reason': '가입 경로 미입력을 0으로 처리'
    },
    'numeric_features': {
        'type': '결측치 처리',
        'rule': 'NaN → 0',
        'reason': '활동/거래 없는 사용자 = 0 (의미 있는 신호)'
    }
}


# ============================================
# 데이터 로드 함수
# ============================================
def load_raw_data(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    원본 데이터를 로드합니다.
    
    Returns:
        train, user_logs, transactions, members 데이터프레임 튜플
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    print("📂 데이터 로드 중...")
    
    train = pd.read_csv(data_dir / 'train_v2.csv')
    print(f"  ✓ train_v2.csv: {len(train):,} rows")
    
    user_logs = pd.read_csv(data_dir / 'user_logs_v2.csv')
    print(f"  ✓ user_logs_v2.csv: {len(user_logs):,} rows")
    
    transactions = pd.read_csv(data_dir / 'transactions_v2.csv')
    print(f"  ✓ transactions_v2.csv: {len(transactions):,} rows")
    
    members = pd.read_csv(data_dir / 'members_v3.csv')
    print(f"  ✓ members_v3.csv: {len(members):,} rows")
    
    return train, user_logs, transactions, members


# ============================================
# 전처리 함수
# ============================================
def preprocess_dates(user_logs: pd.DataFrame, 
                     transactions: pd.DataFrame, 
                     members: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    날짜 컬럼을 datetime 형식으로 변환합니다.
    """
    print("\n🔧 날짜 형식 변환 중...")
    
    # user_logs
    user_logs = user_logs.copy()
    user_logs['date'] = pd.to_datetime(user_logs['date'], format='%Y%m%d')
    
    # transactions
    transactions = transactions.copy()
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'], format='%Y%m%d')
    transactions['membership_expire_date'] = pd.to_datetime(transactions['membership_expire_date'], format='%Y%m%d')
    
    # members
    members = members.copy()
    members['registration_init_time'] = pd.to_datetime(members['registration_init_time'], format='%Y%m%d')
    
    print("  ✓ 날짜 변환 완료")
    
    return user_logs, transactions, members


def filter_observation_window(user_logs: pd.DataFrame, 
                               transactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    관측 윈도우 및 예측 시점 T 기준으로 데이터를 필터링합니다.
    - user_logs: 2017-03-01 ~ 2017-03-31
    - transactions: T (2017-04-01) 이전 (전체 이력 사용)
    """
    print("\n🔧 관측 윈도우 필터링 중...")
    
    # user_logs: 관측 윈도우 내 데이터만 (30일)
    user_logs_filtered = user_logs[
        (user_logs['date'] >= OBSERVATION_START) & 
        (user_logs['date'] <= OBSERVATION_END)
    ].copy()
    print(f"  ✓ user_logs: {len(user_logs):,} → {len(user_logs_filtered):,} rows (30일 윈도우)")
    
    # transactions: T 이전 데이터만 (2015년~2017년 3월, 약 2년치)
    transactions_filtered = transactions[
        transactions['transaction_date'] < PREDICTION_TIME
    ].copy()
    
    # transactions 기간 확인
    txn_min = transactions_filtered['transaction_date'].min()
    txn_max = transactions_filtered['transaction_date'].max()
    print(f"  ✓ transactions: {len(transactions):,} → {len(transactions_filtered):,} rows")
    print(f"    (기간: {txn_min.strftime('%Y-%m-%d')} ~ {txn_max.strftime('%Y-%m-%d')})")
    
    return user_logs_filtered, transactions_filtered


# ============================================
# Feature Engineering 함수
# ============================================
def create_user_log_features(user_logs: pd.DataFrame) -> pd.DataFrame:
    """
    user_logs_v2에서 사용자별 행동 피처를 생성합니다.
    
    생성되는 피처:
    - total_songs: 총 재생 곡 수
    - total_secs: 총 청취 시간 (초)
    - num_25_sum: 25% 미만 청취 곡 수 (스킵)
    - num_100_sum: 완주 곡 수
    - num_unq_sum: 고유 곡 수
    - active_days: 활동 일수
    - skip_ratio: 스킵율
    - complete_ratio: 완주율
    - avg_songs_per_day: 일평균 재생 곡 수
    - avg_secs_per_day: 일평균 청취 시간
    - listening_variety: 청취 다양성
    """
    print("\n🎵 User Log Features 생성 중...")
    
    df = user_logs.copy()
    
    # 총 곡 수 계산
    df['total_songs'] = (df['num_25'] + df['num_50'] + df['num_75'] + 
                         df['num_985'] + df['num_100'])
    
    # 집계
    agg_dict = {
        'total_songs': 'sum',
        'total_secs': 'sum',
        'num_25': 'sum',
        'num_50': 'sum',
        'num_75': 'sum',
        'num_985': 'sum',
        'num_100': 'sum',
        'num_unq': 'sum',
        'date': 'nunique'
    }
    
    features = df.groupby('msno').agg(agg_dict).reset_index()
    features.columns = ['msno', 'total_songs', 'total_secs', 'num_25_sum', 
                        'num_50_sum', 'num_75_sum', 'num_985_sum', 'num_100_sum',
                        'num_unq_sum', 'active_days']
    
    # 파생 피처 생성
    eps = 1e-9  # 0으로 나누기 방지
    
    features['skip_ratio'] = features['num_25_sum'] / (features['total_songs'] + eps)
    features['complete_ratio'] = features['num_100_sum'] / (features['total_songs'] + eps)
    features['partial_ratio'] = (features['num_50_sum'] + features['num_75_sum']) / (features['total_songs'] + eps)
    features['avg_songs_per_day'] = features['total_songs'] / (features['active_days'] + eps)
    features['avg_secs_per_day'] = features['total_secs'] / (features['active_days'] + eps)
    features['listening_variety'] = features['num_unq_sum'] / (features['total_songs'] + eps)
    features['avg_song_length'] = features['total_secs'] / (features['total_songs'] + eps)
    
    print(f"  ✓ {len(features):,} users, {len(features.columns)-1} features")
    
    return features


def create_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    transactions_v2에서 사용자별 결제 피처를 생성합니다.
    약 2년치 거래 이력을 집계합니다.
    
    생성되는 피처:
    - transaction_count: 거래 횟수
    - total_payment: 총 결제 금액
    - avg_payment: 평균 결제 금액
    - cancel_count: 취소 횟수
    - auto_renew_rate: 자동 갱신 비율
    - is_auto_renew_last: 마지막 거래 자동 갱신 여부
    - plan_days_last: 마지막 구독 기간
    - days_to_expire: 만료까지 남은 일수
    - discount_rate: 평균 할인율
    """
    print("\n💳 Transaction Features 생성 중...")
    
    df = transactions.copy()
    
    # 최신 거래 추출
    df_sorted = df.sort_values(['msno', 'transaction_date'], ascending=[True, False])
    latest = df_sorted.groupby('msno').first().reset_index()
    
    # 할인율 계산
    df['discount_rate'] = 1 - (df['actual_amount_paid'] / (df['plan_list_price'] + 1e-9))
    df['discount_rate'] = df['discount_rate'].clip(0, 1)  # 0~1 범위로 제한
    
    # 집계
    agg_dict = {
        'actual_amount_paid': ['sum', 'mean'],
        'plan_list_price': 'mean',
        'is_cancel': 'sum',
        'is_auto_renew': 'mean',
        'discount_rate': 'mean',
        'transaction_date': 'count'
    }
    
    features = df.groupby('msno').agg(agg_dict).reset_index()
    features.columns = ['msno', 'total_payment', 'avg_payment', 'avg_list_price',
                        'cancel_count', 'auto_renew_rate', 'avg_discount_rate', 
                        'transaction_count']
    
    # 최신 거래 정보 병합
    latest_cols = latest[['msno', 'is_auto_renew', 'payment_plan_days', 
                          'membership_expire_date', 'payment_method_id']]
    latest_cols = latest_cols.rename(columns={
        'is_auto_renew': 'is_auto_renew_last',
        'payment_plan_days': 'plan_days_last',
        'membership_expire_date': 'expire_date',
        'payment_method_id': 'payment_method_last'
    })
    
    features = features.merge(latest_cols, on='msno', how='left')
    
    # 만료까지 남은 일수
    features['days_to_expire'] = (features['expire_date'] - PREDICTION_TIME).dt.days
    features = features.drop('expire_date', axis=1)
    
    # 취소 여부 플래그
    features['has_cancelled'] = (features['cancel_count'] > 0).astype(int)
    
    print(f"  ✓ {len(features):,} users, {len(features.columns)-1} features")
    
    return features


def create_member_features(members: pd.DataFrame) -> pd.DataFrame:
    """
    members_v3에서 사용자별 정적 피처를 생성합니다.
    
    생성되는 피처:
    - tenure_days: 가입 후 경과 일수
    - city: 도시 코드
    - age: 나이 (이상치 처리됨)
    - gender: 성별
    - registered_via: 가입 경로
    """
    print("\n👤 Member Features 생성 중...")
    
    df = members.copy()
    
    # 가입 후 경과 일수
    df['tenure_days'] = (PREDICTION_TIME - df['registration_init_time']).dt.days
    
    # 나이 이상치 처리 (0~100 범위 외 → NaN → 중앙값 대체)
    original_invalid = ((df['bd'] <= 0) | (df['bd'] >= 100)).sum()
    df['bd'] = df['bd'].apply(lambda x: x if 0 < x < 100 else np.nan)
    median_age = df['bd'].median()
    df['bd'] = df['bd'].fillna(median_age)
    df = df.rename(columns={'bd': 'age'})
    print(f"  ✓ 나이 이상치 {original_invalid:,}개 → 중앙값({median_age:.0f})으로 대체")
    
    # 성별 결측치 처리
    gender_missing = df['gender'].isnull().sum()
    df['gender'] = df['gender'].fillna('unknown')
    print(f"  ✓ 성별 결측치 {gender_missing:,}개 → 'unknown'으로 대체")
    
    # 도시 결측치 처리
    df['city'] = df['city'].fillna(0).astype(int)
    
    # 가입 경로 결측치 처리
    df['registered_via'] = df['registered_via'].fillna(0).astype(int)
    
    # 필요한 컬럼만 선택
    features = df[['msno', 'city', 'age', 'gender', 'registered_via', 'tenure_days']]
    
    print(f"  ✓ {len(features):,} users, {len(features.columns)-1} features")
    
    return features


# ============================================
# 인코딩 함수
# ============================================
def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    범주형 변수를 인코딩합니다.
    - gender: One-hot encoding
    - city, registered_via, payment_method_last: 그대로 유지 (수치형으로 처리)
    """
    print("\n🔢 범주형 변수 인코딩 중...")
    
    df = df.copy()
    
    # gender One-hot encoding
    if 'gender' in df.columns:
        gender_dummies = pd.get_dummies(df['gender'], prefix='gender')
        df = pd.concat([df.drop('gender', axis=1), gender_dummies], axis=1)
        print(f"  ✓ gender → One-hot ({gender_dummies.shape[1]} columns)")
    
    return df


# ============================================
# 데이터 병합 함수
# ============================================
def merge_features(train: pd.DataFrame,
                   user_log_features: pd.DataFrame,
                   transaction_features: pd.DataFrame,
                   member_features: pd.DataFrame) -> pd.DataFrame:
    """
    모든 피처를 train 기준으로 LEFT JOIN하여 병합합니다.
    """
    print("\n🔗 피처 병합 중...")
    
    # train 기준으로 병합
    df = train.copy()
    
    df = df.merge(user_log_features, on='msno', how='left')
    print(f"  ✓ + user_log_features: {df.shape}")
    
    df = df.merge(transaction_features, on='msno', how='left')
    print(f"  ✓ + transaction_features: {df.shape}")
    
    df = df.merge(member_features, on='msno', how='left')
    print(f"  ✓ + member_features: {df.shape}")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    결측치를 처리합니다.
    - 수치형: 0으로 채움 (활동 없음 = 0)
    - 범주형: 'unknown' 또는 최빈값
    """
    print("\n🔧 결측치 처리 중...")
    
    df = df.copy()
    
    # 수치형 컬럼 결측치 → 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['msno', 'is_churn']]
    
    missing_before = df[numeric_cols].isnull().sum().sum()
    df[numeric_cols] = df[numeric_cols].fillna(0)
    print(f"  ✓ 수치형 결측치 {missing_before:,}개 → 0으로 대체")
    
    # 범주형 컬럼 결측치
    if 'gender' in df.columns:
        df['gender'] = df['gender'].fillna('unknown')
    
    return df


# ============================================
# 데이터 분할 함수
# ============================================
def split_dataset(df: pd.DataFrame, 
                  test_size: float = 0.15, 
                  valid_size: float = 0.15,
                  random_state: int = 719) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    데이터를 train/valid/test로 분할합니다.
    Stratified split으로 churn 비율을 유지합니다.
    
    Args:
        df: 전체 데이터셋
        test_size: 테스트셋 비율 (default: 0.2)
        valid_size: 검증셋 비율 (default: 0.1)
        random_state: 랜덤 시드
    
    Returns:
        train, valid, test 데이터프레임 튜플
        
    분할 비율:
        - train: 70%
        - valid: 10%
        - test: 20%
    """
    print("\n✂️ 데이터셋 분할 중...")
    
    # 첫 번째 분할: train+valid / test
    train_valid, test = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['is_churn'], 
        random_state=random_state
    )
    
    # 두 번째 분할: train / valid
    valid_ratio = valid_size / (1 - test_size)  # 0.1 / 0.8 = 0.125
    train, valid = train_test_split(
        train_valid, 
        test_size=valid_ratio, 
        stratify=train_valid['is_churn'], 
        random_state=random_state
    )
    
    print(f"  ✓ Train: {len(train):,} rows ({len(train)/len(df)*100:.1f}%)")
    print(f"  ✓ Valid: {len(valid):,} rows ({len(valid)/len(df)*100:.1f}%)")
    print(f"  ✓ Test:  {len(test):,} rows ({len(test)/len(df)*100:.1f}%)")
    
    # Churn 비율 확인
    print(f"\n  Churn 비율:")
    print(f"    - Train: {train['is_churn'].mean()*100:.2f}%")
    print(f"    - Valid: {valid['is_churn'].mean()*100:.2f}%")
    print(f"    - Test:  {test['is_churn'].mean()*100:.2f}%")
    
    return train, valid, test


# ============================================
# Sanity Check 함수
# ============================================
def sanity_check(df: pd.DataFrame, name: str = "Dataset") -> Dict[str, Any]:
    """
    전처리 결과를 검증합니다.
    
    검증 항목:
    1. 데이터 shape
    2. 결측치 개수
    3. 중복 msno 여부
    4. Churn 비율
    5. 수치형 컬럼 음수값 여부
    6. 각 컬럼 기초 통계
    """
    print(f"\n🔍 Sanity Check: {name}")
    print("-" * 50)
    
    results = {}
    
    # 1. Shape
    results['shape'] = df.shape
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # 2. 결측치
    missing_total = df.isnull().sum().sum()
    missing_cols = df.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0]
    results['missing_total'] = missing_total
    print(f"  결측치: {missing_total:,}개")
    if len(missing_cols) > 0:
        print(f"    - 결측 컬럼: {dict(missing_cols)}")
    
    # 3. 중복 msno
    duplicates = df['msno'].duplicated().sum()
    results['duplicates'] = duplicates
    print(f"  중복 msno: {duplicates:,}개")
    if duplicates > 0:
        print("    ⚠️ 경고: 중복된 사용자가 있습니다!")
    
    # 4. Churn 비율
    if 'is_churn' in df.columns:
        churn_rate = df['is_churn'].mean()
        results['churn_rate'] = churn_rate
        print(f"  Churn 비율: {churn_rate*100:.2f}%")
    
    # 5. 수치형 컬럼 음수값 체크
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    negative_check = {}
    for col in numeric_cols:
        if col in ['is_churn', 'msno']:
            continue
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            negative_check[col] = neg_count
    
    results['negative_values'] = negative_check
    if negative_check:
        print(f"  음수값 컬럼: {negative_check}")
    else:
        print(f"  음수값: 없음 ✓")
    
    # 6. 무한값 체크
    inf_check = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_check[col] = inf_count
    
    results['infinite_values'] = inf_check
    if inf_check:
        print(f"  무한값 컬럼: {inf_check}")
    else:
        print(f"  무한값: 없음 ✓")
    
    print("-" * 50)
    
    return results


# ============================================
# 메인 파이프라인
# ============================================
def run_preprocessing_pipeline(data_dir: Optional[Path] = None, 
                                save_dir: Optional[Path] = None,
                                split_data: bool = True) -> Tuple[pd.DataFrame, Optional[Tuple]]:
    """
    전체 전처리 및 피처 엔지니어링 파이프라인을 실행합니다.
    
    Args:
        data_dir: 데이터 디렉토리 경로
        save_dir: 결과 저장 디렉토리 (None이면 저장하지 않음)
        split_data: train/valid/test 분할 여부
    
    Returns:
        (전체 피처 테이블, (train, valid, test) 또는 None)
    """
    print("=" * 60)
    print("🚀 KKBox Preprocessing & Feature Engineering Pipeline")
    print("=" * 60)
    print(f"예측 시점 (T): {PREDICTION_TIME.strftime('%Y-%m-%d')}")
    print(f"관측 윈도우: {OBSERVATION_START.strftime('%Y-%m-%d')} ~ {OBSERVATION_END.strftime('%Y-%m-%d')}")
    
    # 1. 데이터 로드
    train, user_logs, transactions, members = load_raw_data(data_dir)
    
    # 2. 날짜 전처리
    user_logs, transactions, members = preprocess_dates(user_logs, transactions, members)
    
    # 3. 관측 윈도우 필터링
    user_logs, transactions = filter_observation_window(user_logs, transactions)
    
    # 4. Feature Engineering
    user_log_features = create_user_log_features(user_logs)
    transaction_features = create_transaction_features(transactions)
    member_features = create_member_features(members)
    
    # 5. 데이터 병합
    df = merge_features(train, user_log_features, transaction_features, member_features)
    
    # 6. 결측치 처리
    df = handle_missing_values(df)
    
    # 7. 범주형 인코딩
    df = encode_categorical_features(df)
    
    # 8. Sanity Check (전체 데이터)
    sanity_check(df, "Full Dataset")
    
    # 9. 결과 요약
    print("\n" + "=" * 60)
    print("✅ 파이프라인 완료!")
    print("=" * 60)
    print(f"최종 데이터셋: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Churn 비율: {df['is_churn'].mean()*100:.2f}%")
    
    # 10. 데이터 분할
    splits = None
    if split_data:
        train_df, valid_df, test_df = split_dataset(df)
        splits = (train_df, valid_df, test_df)
        
        # 분할 데이터 Sanity Check
        sanity_check(train_df, "Train Set")
        sanity_check(valid_df, "Valid Set")
        sanity_check(test_df, "Test Set")
    
    # 11. 저장 (옵션)
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 전체 데이터 저장
        df.to_csv(save_dir / 'feature_table_ldh.csv', index=False)
        print(f"\n💾 저장 완료: {save_dir / 'feature_table_ldh.csv'}")
        
        # 분할 데이터 저장
        if splits:
            train_df, valid_df, test_df = splits
            train_df.to_csv(save_dir / 'train_set.csv', index=False)
            valid_df.to_csv(save_dir / 'valid_set.csv', index=False)
            test_df.to_csv(save_dir / 'test_set.csv', index=False)
            print(f"💾 저장 완료: train_set.csv, valid_set.csv, test_set.csv")
    
    return df, splits


def get_feature_info() -> pd.DataFrame:
    """
    생성된 피처 정보를 반환합니다.
    """
    feature_info = [
        # User Log Features
        {'feature': 'total_songs', 'type': 'numeric', 'source': 'user_logs_v2', 'description': '30일간 총 재생 곡 수'},
        {'feature': 'total_secs', 'type': 'numeric', 'source': 'user_logs_v2', 'description': '30일간 총 청취 시간 (초)'},
        {'feature': 'num_25_sum', 'type': 'numeric', 'source': 'user_logs_v2', 'description': '25% 미만 청취 곡 수 (스킵)'},
        {'feature': 'num_50_sum', 'type': 'numeric', 'source': 'user_logs_v2', 'description': '25-50% 청취 곡 수'},
        {'feature': 'num_75_sum', 'type': 'numeric', 'source': 'user_logs_v2', 'description': '50-75% 청취 곡 수'},
        {'feature': 'num_985_sum', 'type': 'numeric', 'source': 'user_logs_v2', 'description': '75-98.5% 청취 곡 수'},
        {'feature': 'num_100_sum', 'type': 'numeric', 'source': 'user_logs_v2', 'description': '98.5%+ 완주 곡 수'},
        {'feature': 'num_unq_sum', 'type': 'numeric', 'source': 'user_logs_v2', 'description': '고유 곡 수'},
        {'feature': 'active_days', 'type': 'numeric', 'source': 'user_logs_v2', 'description': '30일 중 활동 일수'},
        {'feature': 'skip_ratio', 'type': 'numeric', 'source': 'user_logs_v2', 'description': '스킵율 (num_25/total)'},
        {'feature': 'complete_ratio', 'type': 'numeric', 'source': 'user_logs_v2', 'description': '완주율 (num_100/total)'},
        {'feature': 'partial_ratio', 'type': 'numeric', 'source': 'user_logs_v2', 'description': '부분청취율 ((num_50+num_75)/total)'},
        {'feature': 'avg_songs_per_day', 'type': 'numeric', 'source': 'user_logs_v2', 'description': '일평균 재생 곡 수'},
        {'feature': 'avg_secs_per_day', 'type': 'numeric', 'source': 'user_logs_v2', 'description': '일평균 청취 시간 (초)'},
        {'feature': 'listening_variety', 'type': 'numeric', 'source': 'user_logs_v2', 'description': '청취 다양성 (unique/total)'},
        {'feature': 'avg_song_length', 'type': 'numeric', 'source': 'user_logs_v2', 'description': '평균 곡 길이 (초)'},
        
        # Transaction Features
        {'feature': 'total_payment', 'type': 'numeric', 'source': 'transactions_v2', 'description': '총 결제 금액 (2년 누적)'},
        {'feature': 'avg_payment', 'type': 'numeric', 'source': 'transactions_v2', 'description': '평균 결제 금액'},
        {'feature': 'avg_list_price', 'type': 'numeric', 'source': 'transactions_v2', 'description': '평균 정가'},
        {'feature': 'cancel_count', 'type': 'numeric', 'source': 'transactions_v2', 'description': '취소 횟수'},
        {'feature': 'auto_renew_rate', 'type': 'numeric', 'source': 'transactions_v2', 'description': '자동 갱신 비율'},
        {'feature': 'avg_discount_rate', 'type': 'numeric', 'source': 'transactions_v2', 'description': '평균 할인율'},
        {'feature': 'transaction_count', 'type': 'numeric', 'source': 'transactions_v2', 'description': '거래 횟수 (2년 누적)'},
        {'feature': 'is_auto_renew_last', 'type': 'binary', 'source': 'transactions_v2', 'description': '최근 거래 자동갱신 여부'},
        {'feature': 'plan_days_last', 'type': 'numeric', 'source': 'transactions_v2', 'description': '최근 구독 기간 (일)'},
        {'feature': 'payment_method_last', 'type': 'categorical', 'source': 'transactions_v2', 'description': '최근 결제 수단'},
        {'feature': 'days_to_expire', 'type': 'numeric', 'source': 'transactions_v2', 'description': '만료까지 남은 일수 (T 기준)'},
        {'feature': 'has_cancelled', 'type': 'binary', 'source': 'transactions_v2', 'description': '취소 이력 여부'},
        
        # Member Features
        {'feature': 'city', 'type': 'categorical', 'source': 'members_v3', 'description': '도시 코드'},
        {'feature': 'age', 'type': 'numeric', 'source': 'members_v3', 'description': '나이 (이상치 처리됨)'},
        {'feature': 'registered_via', 'type': 'categorical', 'source': 'members_v3', 'description': '가입 경로'},
        {'feature': 'tenure_days', 'type': 'numeric', 'source': 'members_v3', 'description': '가입 후 경과 일수'},
        {'feature': 'gender_female', 'type': 'binary', 'source': 'members_v3', 'description': '성별: 여성'},
        {'feature': 'gender_male', 'type': 'binary', 'source': 'members_v3', 'description': '성별: 남성'},
        {'feature': 'gender_unknown', 'type': 'binary', 'source': 'members_v3', 'description': '성별: 미입력'},
    ]
    
    return pd.DataFrame(feature_info)


# ============================================
# 실행
# ============================================
if __name__ == "__main__":
    # 파이프라인 실행 및 저장
    df, splits = run_preprocessing_pipeline(save_dir=DATA_DIR, split_data=True)
    
    # 피처 목록 출력
    print("\n📋 생성된 피처 목록:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")
    
    # 피처 정보 저장
    feature_df = get_feature_info()
    feature_df.to_csv(DATA_DIR / 'feature_dictionary.csv', index=False)
    print(f"\n💾 피처 사전 저장: {DATA_DIR / 'feature_dictionary.csv'}")
