import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path

pd.set_option('display.max_columns', None)

# 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed"
RAW_DATA_DIR = PROJECT_ROOT / "data/raw"

V3_PATH = DATA_DIR / "kkbox_train_feature_v3.parquet"
USER_LOGS_PATH = RAW_DATA_DIR / "user_logs_v2.csv"
OUTPUT_PATH = DATA_DIR / "kkbox_train_feature_v4.parquet"

print(f"Project Root: {PROJECT_ROOT}")
print(f"Loading V3 Data from: {V3_PATH}")

if not V3_PATH.exists():
    print(f"Error: V3 file not found at {V3_PATH}")
    # Fallback/Debug info
    if (PROJECT_ROOT / "data").exists():
        print(f"Data directory exists at {PROJECT_ROOT / 'data'}")
    else:
        print(f"Data directory MISSING at {PROJECT_ROOT / 'data'}")
    exit(1)

df_v3 = pd.read_parquet(V3_PATH)
print(f"V3 Shape: {df_v3.shape}")

print("Creating Arithmetic Derived Features...")
df_v4 = df_v3.copy()

# 1. Active Decay Rate (활동 감소율)
epsilon = 1e-6
val_decay = df_v4['num_days_active_w7'] / ((df_v4['num_days_active_w30'] / 4) + epsilon)
df_v4['active_decay_rate'] = val_decay.clip(upper=10.0)

# 2. Listening Time Velocity (청취 가속도)
df_v4['listening_time_velocity'] = df_v4['avg_secs_per_day_w7'] - df_v4['avg_secs_per_day_w14']

# 3. Discovery Index (탐색 지수)
val_disc = df_v4['num_unq_w7'] / (df_v4['num_songs_w7'] + epsilon)
df_v4['discovery_index'] = val_disc.clip(upper=1.0)

# 4. Skip Passion Index (스킵 열정도)
val_skip = df_v4['num_25_w7'] / (df_v4['num_100_w7'] + epsilon)
df_v4['skip_passion_index'] = val_skip.clip(upper=100.0)

# 5. Daily Listening Variance (Renaming existing feature)
df_v4['daily_listening_variance'] = df_v4['std_secs_w7']

# 6. Engagement Density (몰입 밀도)
df_v4['engagement_density'] = df_v4['total_secs_w7'] / (df_v4['num_days_active_w7'] + epsilon)

print("Arithmetic features created.")

# 7. Last Active Gap
print("Processing Raw User Logs for Last Active Gap...")

if not os.path.exists(USER_LOGS_PATH):
     print(f"Warning: Raw user logs not found at {USER_LOGS_PATH}. Skipping last_active_gap calculation (filling with -1).")
     df_v4['last_active_gap'] = -1
else:
    # user_logs_v2.csv is large, read necessary columns only
    chunks = pd.read_csv(USER_LOGS_PATH, usecols=['msno', 'date'], chunksize=1000000)

    max_dates = []
    print("Reading chunks...")
    for i, chunk in enumerate(chunks):
        chunk_max = chunk.groupby('msno')['date'].max()
        max_dates.append(chunk_max)
        if i % 10 == 0:
            print(f"Processed chunk {i}...")

    # Combine and find global max per user
    all_max_dates = pd.concat(max_dates)
    final_last_active = all_max_dates.groupby(level=0).max().reset_index()
    final_last_active.rename(columns={'date': 'last_active_date'}, inplace=True)

    # Convert to datetime
    final_last_active['last_active_date'] = pd.to_datetime(final_last_active['last_active_date'], format='%Y%m%d')

    print(f"Max Active Dates Calculated. Users: {len(final_last_active)}")
    
    # Determine Study Cutoff Date
    global_max_date = final_last_active['last_active_date'].max()
    print(f"Global Max Date in Logs: {global_max_date}")

    # Calculate Gap
    final_last_active['last_active_gap'] = (global_max_date - final_last_active['last_active_date']).dt.days
    
    # Merge with V4 DataFrame
    df_v4 = df_v4.merge(final_last_active[['msno', 'last_active_gap']], on='msno', how='left')

    # Fill NA for users with no logs
    max_gap_found = df_v4['last_active_gap'].max()
    df_v4['last_active_gap'] = df_v4['last_active_gap'].fillna(max_gap_found + 1)

print("Merged Last Active Gap.")

print(f"Saving V4 to {OUTPUT_PATH}...")
df_v4.to_parquet(OUTPUT_PATH, index=False)
print("Done.")

print("New Feature Statistics:")
new_cols = ['active_decay_rate', 'listening_time_velocity', 'discovery_index', 'skip_passion_index', 'last_active_gap']
print(df_v4[new_cols].describe())
