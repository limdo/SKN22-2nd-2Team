import pandas as pd

print('Loading parquet file...')
df = pd.read_parquet('data/kkbox_train_feature_v1.parquet')

print(f'Original shape: {df.shape}')
print(f'Original columns: {len(df.columns)}')

# 중복 컬럼 제거
drop_cols = ['recency_secs_ratio', 'recency_songs_ratio']
existing_drops = [c for c in drop_cols if c in df.columns]

if existing_drops:
    df = df.drop(columns=existing_drops)
    print(f'Dropped columns: {existing_drops}')
else:
    print('No columns to drop (already removed)')

print(f'New shape: {df.shape}')
print(f'New columns: {len(df.columns)}')

# 저장
df.to_parquet('data/kkbox_train_feature_v1.parquet', index=False)
print('Saved successfully!')

