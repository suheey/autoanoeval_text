import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

def load_dataset(dataset_name):
    """CSV 데이터셋 로드 및 전처리"""
    csv_path = f'/lab-di/nfsdata/home/suhee.yoon/autoanoeval/data/adbench_column/{dataset_name}.csv'    
    print(f"📥 CSV 데이터셋 로드: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ 파일이 존재하지 않습니다: {csv_path}")
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"📊 로드된 데이터 형태: {df.shape}")

    y = df['label'].values
    df = df.drop(columns=['label'])

    # 숫자 / 문자열 feature 자동 구분
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    string_cols = df.select_dtypes(include=['object', 'string']).columns
    
    print(f"📈 숫자 컬럼: {len(numeric_cols)}개, 문자열 컬럼: {len(string_cols)}개")

    # 숫자 feature normalize
    scaler = MinMaxScaler()
    X_numeric = scaler.fit_transform(df[numeric_cols].values)

    # 문자열 feature → integer encoding
    X_strings = []
    for col in string_cols:
        unique_vals = np.unique(df[col])
        val_to_int = {val: idx for idx, val in enumerate(unique_vals)}
        encoded_col = np.vectorize(val_to_int.get)(df[col].values).reshape(-1, 1)
        X_strings.append(encoded_col)
        print(f"   🔤 {col}: {len(unique_vals)}개 고유값 → 정수 인코딩")

    # 숫자 + 문자열 feature 결합
    if X_strings:
        X_strings = np.hstack(X_strings)
        X = np.hstack([X_numeric, X_strings])
        print(f"📊 최종 feature 차원: {X_numeric.shape[1]} (숫자) + {X_strings.shape[1]} (문자열) = {X.shape[1]}")
    else:
        X = X_numeric
        print(f"📊 최종 feature 차원: {X.shape[1]} (숫자만)")

    print(f"🏷️ 레이블 분포: 정상 {np.sum(y == 0):,}개, 이상 {np.sum(y == 1):,}개")
    
    return X, y