"""
User Logs 집계 스크립트
========================
작성자: 이도훈 (LDH)
작성일: 2025-12-16

목적: user_logs_v2.csv를 msno 기준으로 멀티 윈도우 집계
출력: user_logs_aggregated_ldh.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================
# 설정
# ============================================================
DATA_DIR = Path(__file__).resolve().parents[2] / 'data'
T = pd.Timestamp('2017-04-01')  # 예측 시점

# 멀티 윈도우 정의
WINDOWS = {
    'w7': (pd.Timestamp('2017-03-25'), pd.Timestamp('2017-03-31')),   # 최근 7일
    'w14': (pd.Timestamp('2017-03-18'), pd.Timestamp('2017-03-31')),  # 최근 14일
    'w21': (pd.Timestamp('2017-03-11'), pd.Timestamp('2017-03-31')),  # 최근 21일
    'w30': (pd.Timestamp('2017-03-01'), pd.Timestamp('2017-03-31')),  # 전체 30일
}


# ============================================================
# 이상치 처리 함수
# ============================================================
def clip_outliers(df: pd.DataFrame, column: str, lower_pct: float = 0.001, upper_pct: float = 0.999) -> pd.DataFrame:
    """
    Percentile 기반 이상치 클리핑
    
    Args:
        df: 데이터프레임
        column: 클리핑할 컬럼명
        lower_pct: 하한 퍼센타일 (기본 0.1%)
        upper_pct: 상한 퍼센타일 (기본 99.9%)
    
    Returns:
        클리핑된 데이터프레임
    """
    if column not in df.columns:
        return df
    
    lower_bound = df[column].quantile(lower_pct)
    upper_bound = df[column].quantile(upper_pct)
    
    original_min = df[column].min()
    original_max = df[column].max()
    
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    clipped_count = ((df[column] == lower_bound) | (df[column] == upper_bound)).sum()
    
    print(f"  {column}: [{original_min:.2f}, {original_max:.2f}] -> [{lower_bound:.2f}, {upper_bound:.2f}] (clipped: {clipped_count})")
    
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    user_logs 데이터의 이상치 처리
    
    처리 대상:
    - total_secs: 총 청취 시간
    - num_25, num_50, num_75, num_985, num_100: 재생 구간별 곡 수
    - num_unq: 고유 곡 수
    """
    print("\n[Outlier Handling] Percentile-based clipping (0.1% - 99.9%):")
    
    outlier_columns = ['total_secs', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq']
    
    for col in outlier_columns:
        df = clip_outliers(df, col)
    
    return df


# ============================================================
# 데이터 로드
# ============================================================
def load_user_logs(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """user_logs_v2.csv 로드 및 기본 전처리"""
    print("[1/5] Loading user_logs_v2.csv...")
    
    df = pd.read_csv(data_dir / 'user_logs_v2.csv')
    print(f"  Raw shape: {df.shape}")
    
    # 날짜 변환
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    
    # 이상치 처리
    df = handle_outliers(df)
    
    # 기본 통계
    print(f"  Date range: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  Unique users: {df['msno'].nunique():,}")
    
    return df


# ============================================================
# 단일 윈도우 집계
# ============================================================
def aggregate_single_window(df: pd.DataFrame, window_name: str, 
                            start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    단일 윈도우에 대한 집계 수행
    
    Args:
        df: user_logs 데이터프레임
        window_name: 윈도우 이름 (예: 'w7', 'w14')
        start_date: 시작일
        end_date: 종료일
    
    Returns:
        집계된 데이터프레임
    """
    # 윈도우 필터링
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    window_df = df[mask].copy()
    
    if len(window_df) == 0:
        print(f"  Warning: No data in window {window_name}")
        return pd.DataFrame()
    
    # 기본 집계
    agg = window_df.groupby('msno').agg({
        'date': 'nunique',           # 활동 일수
        'total_secs': ['sum', 'mean', 'std'],  # 청취 시간
        'num_25': 'sum',             # 25% 미만 청취 (스킵)
        'num_50': 'sum',             # 50% 미만 청취
        'num_75': 'sum',             # 75% 미만 청취
        'num_985': 'sum',            # 98.5% 미만 청취
        'num_100': 'sum',            # 100% 청취 (완주)
        'num_unq': 'sum',            # 고유 곡 수
    }).reset_index()
    
    # 컬럼명 정리
    agg.columns = ['msno', 
                   'num_days_active', 'total_secs', 'avg_secs_per_day', 'std_secs',
                   'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq']
    
    # 총 재생 곡 수 계산
    agg['num_songs'] = agg['num_25'] + agg['num_50'] + agg['num_75'] + agg['num_985'] + agg['num_100']
    
    # 일평균 곡 수
    agg['avg_songs_per_day'] = agg['num_songs'] / (agg['num_days_active'] + 1e-9)
    
    # 50% 미만 재생 (short play)
    agg['short_play'] = agg['num_25'] + agg['num_50']
    
    # 비율 계산 (epsilon 추가로 0 나눗셈 방지)
    eps = 1e-9
    agg['skip_ratio'] = agg['num_25'] / (agg['num_songs'] + eps)
    agg['completion_ratio'] = agg['num_100'] / (agg['num_songs'] + eps)
    agg['short_play_ratio'] = agg['short_play'] / (agg['num_songs'] + eps)
    agg['variety_ratio'] = agg['num_unq'] / (agg['num_songs'] + eps)
    
    # 불필요 컬럼 제거
    agg = agg.drop(['num_50', 'num_75', 'num_985'], axis=1)
    
    # 윈도우 접미사 추가
    rename_dict = {col: f"{col}_{window_name}" for col in agg.columns if col != 'msno'}
    agg = agg.rename(columns=rename_dict)
    
    # Inf 값 처리
    agg = agg.replace([np.inf, -np.inf], 0)
    
    return agg


# ============================================================
# 전체 윈도우 집계
# ============================================================
def aggregate_all_windows(df: pd.DataFrame) -> pd.DataFrame:
    """모든 윈도우에 대해 집계 수행 후 병합"""
    print("\n[2/5] Aggregating by windows...")
    
    result = None
    
    for window_name, (start_date, end_date) in WINDOWS.items():
        print(f"  Processing {window_name}: {start_date.date()} ~ {end_date.date()}")
        
        window_agg = aggregate_single_window(df, window_name, start_date, end_date)
        
        if result is None:
            result = window_agg
        else:
            result = result.merge(window_agg, on='msno', how='outer')
    
    # NaN을 0으로 채우기 (해당 윈도우에 활동이 없는 경우)
    result = result.fillna(0)
    
    print(f"  Combined shape: {result.shape}")
    
    return result


# ============================================================
# 변화량/추세 피처 생성
# ============================================================
def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """윈도우 간 변화량/추세 피처 추가"""
    print("\n[3/5] Adding trend features...")
    
    eps = 1e-9
    
    # 청취 시간 추세 (최근 / 전체)
    df['secs_trend_w7_w30'] = df['total_secs_w7'] / (df['total_secs_w30'] + eps)
    df['secs_trend_w14_w30'] = df['total_secs_w14'] / (df['total_secs_w30'] + eps)
    
    # 활동일 추세
    df['days_trend_w7_w14'] = df['num_days_active_w7'] / (df['num_days_active_w14'] + eps)
    df['days_trend_w7_w30'] = df['num_days_active_w7'] / (df['num_days_active_w30'] + eps)
    
    # 곡 수 추세
    df['songs_trend_w7_w30'] = df['num_songs_w7'] / (df['num_songs_w30'] + eps)
    df['songs_trend_w14_w30'] = df['num_songs_w14'] / (df['num_songs_w30'] + eps)
    
    # 스킵율/완주율 변화 (차이)
    df['skip_trend_w7_w30'] = df['skip_ratio_w7'] - df['skip_ratio_w30']
    df['completion_trend_w7_w30'] = df['completion_ratio_w7'] - df['completion_ratio_w30']
    
    # 최근성 비율 (recency)
    df['recency_secs_ratio'] = df['total_secs_w7'] / (df['total_secs_w30'] + eps)
    df['recency_songs_ratio'] = df['num_songs_w7'] / (df['num_songs_w30'] + eps)
    
    # Inf 값 처리
    df = df.replace([np.inf, -np.inf], 0)
    
    # 비율 값 클리핑 (0~1 범위)
    ratio_cols = [col for col in df.columns if 'trend' in col or 'ratio' in col]
    for col in ratio_cols:
        if 'trend' in col and 'skip' not in col and 'completion' not in col:
            df[col] = df[col].clip(0, 1)
    
    print(f"  Added {len(ratio_cols)} trend features")
    
    return df


# ============================================================
# 최종 검증
# ============================================================
def validate_output(df: pd.DataFrame) -> None:
    """출력 데이터 검증"""
    print("\n[4/5] Validating output...")
    
    # 결측치 확인
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("  Warning: Null values found!")
        print(null_counts[null_counts > 0])
    else:
        print("  No null values")
    
    # Inf 값 확인
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
    if inf_counts.sum() > 0:
        print("  Warning: Inf values found!")
        print(inf_counts[inf_counts > 0])
    else:
        print("  No inf values")
    
    # 기본 통계
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")


# ============================================================
# 메인 파이프라인
# ============================================================
def run_aggregation_pipeline(data_dir: Path = DATA_DIR, save: bool = True) -> pd.DataFrame:
    """전체 집계 파이프라인 실행"""
    print("=" * 60)
    print("User Logs Aggregation Pipeline")
    print("=" * 60)
    
    # 1. 데이터 로드
    df = load_user_logs(data_dir)
    
    # 2. 윈도우별 집계
    agg_df = aggregate_all_windows(df)
    
    # 3. 추세 피처 추가
    agg_df = add_trend_features(agg_df)
    
    # 4. 검증
    validate_output(agg_df)
    
    # 5. 저장
    if save:
        print("\n[5/5] Saving to parquet...")
        output_path = data_dir / 'user_logs_aggregated_ldh.parquet'
        agg_df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"  Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)
    
    # head 출력
    print("\n[Sample Data - head(5)]")
    print(agg_df.head())
    
    return agg_df


# ============================================================
# 실행
# ============================================================
if __name__ == '__main__':
    agg_df = run_aggregation_pipeline()

