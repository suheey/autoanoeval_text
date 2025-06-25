import numpy as np
from sklearn.model_selection import train_test_split
from config.settings import RANDOM_SEED, MAX_NORMAL_SAMPLES, MAX_ANOMALY_SAMPLES

def prepare_dataset_splits(X_original, y_original):
    """데이터셋 분할"""
    print(f"📊 원본 데이터: {X_original.shape}")
    print(f"📊 클래스 분포 - 정상: {np.sum(y_original == 0):,}, 이상: {np.sum(y_original == 1):,}")

    # 대용량 데이터 최적화
    max_normal = MAX_NORMAL_SAMPLES
    max_anomaly = MAX_ANOMALY_SAMPLES
    
    if np.sum(y_original == 0) > max_normal * 2:
        print(f"⚡ 대용량 정상 데이터 감지. {max_normal:,}개로 제한")
    if np.sum(y_original == 1) > max_anomaly * 2:
        print(f"⚡ 대용량 이상 데이터 감지. {max_anomaly:,}개로 제한")

    # 데이터 제한 및 분리
    X_normal = X_original[y_original == 0][:max_normal]
    X_anomaly = X_original[y_original == 1][:max_anomaly]
    
    # 데이터 분할
    X_normal_train, X_normal_holdout = train_test_split(
        X_normal, test_size=0.4, random_state=RANDOM_SEED
    )
    X_anomaly_val, X_anomaly_test = train_test_split(
        X_anomaly, test_size=0.7, random_state=RANDOM_SEED
    )
    X_normal_val, X_normal_test = train_test_split(
        X_normal_holdout, test_size=0.5, random_state=RANDOM_SEED
    )
    
    # 최종 데이터셋 구성
    X_val_real = np.vstack([X_normal_val, X_anomaly_val])
    y_val_real = np.concatenate([np.zeros(len(X_normal_val)), np.ones(len(X_anomaly_val))])
    
    X_test = np.vstack([X_normal_test, X_anomaly_test])
    y_test = np.concatenate([np.zeros(len(X_normal_test)), np.ones(len(X_anomaly_test))])
    
    # 데이터 셔플
    idx = np.random.RandomState(RANDOM_SEED).permutation(len(y_val_real))
    X_val_real, y_val_real = X_val_real[idx], y_val_real[idx]
    
    idx = np.random.RandomState(RANDOM_SEED).permutation(len(y_test))
    X_test, y_test = X_test[idx], y_test[idx]
    
    print(f"\n📋 데이터셋 분할 완료:")
    print(f"   Train (정상만): {X_normal_train.shape}")
    print(f"   Real Validation: {X_val_real.shape} (정상: {np.sum(y_val_real == 0):,}, 이상: {np.sum(y_val_real == 1):,})")
    print(f"   Test: {X_test.shape} (정상: {np.sum(y_test == 0):,}, 이상: {np.sum(y_test == 1):,})")
    
    return X_normal_train, X_normal_val, X_val_real, y_val_real, X_test, y_test, X_anomaly_val