import numpy as np
import os
from datetime import datetime
import urllib.request
from sklearn.model_selection import train_test_split
from data_generator import SimpleDataGenerator
from visualization import visualize_tsne
from autoanoeval.ADBench.autoanoeval.model_selection_org import run_model_selection_experiment
import argparse
import sys

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 설정
RANDOM_SEED = 42
ANOMALY_TYPES = ['local', 'cluster', 'global','discrepancy']  # 원하는 3가지 이상치 유형


class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # 버퍼 바로 비우기
    def flush(self):
        for f in self.files:
            f.flush()
            
            
# 데이터셋 다운로드 함수
def download_dataset(url, filename):
    if not os.path.exists(filename):
        print(f"{filename} 다운로드 중...")
        urllib.request.urlretrieve(url, filename)
        print(f"{filename} 다운로드 완료!")
    else:
        print(f"{filename}이 이미 존재합니다.")
        
        
def load_dataset(dataset_name):
    csv_path = f'/lab-di/nfsdata/home/suhee.yoon/autoanoeval/data/adbench_column/{dataset_name}.csv'    
    print(f"📥 csv dataset 로드: {csv_path}")
    df = pd.read_csv(csv_path)

    y = df['label'].values
    df = df.drop(columns=['label'])  # label 제거

    # ✅ 숫자 / 문자열 feature 자동 구분
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    string_cols = df.select_dtypes(include=['object', 'string']).columns

    # ✅ 숫자 feature normalize
    scaler = MinMaxScaler()
    X_numeric = scaler.fit_transform(df[numeric_cols].values)

    # ✅ 문자열 feature → integer encoding
    X_strings = []
    for col in string_cols:
        unique_vals = np.unique(df[col])
        val_to_int = {val: idx for idx, val in enumerate(unique_vals)}
        encoded_col = np.vectorize(val_to_int.get)(df[col].values).reshape(-1, 1)
        X_strings.append(encoded_col)

    # ✅ 숫자 + 문자열 feature 결합
    if X_strings:
        X_strings = np.hstack(X_strings)
        X = np.hstack([X_numeric, X_strings])
    else:
        X = X_numeric

    return X, y


            
def main(args):
    # 결과 저장 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{args.dataset_name}_experiment_results_{timestamp}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # ✅ log.txt로 + 터미널 동시에 출력
    log_file = open(os.path.join(results_dir, "log.txt"), "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    
    # 데이터셋 다운로드
    # dataset_path = f'/lab-di/nfsdata/home/suhee.yoon/autoanoeval/ADBench/adbench/datasets/Classical/{args.dataset_name}.npz'
    # dataset_url = f"https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/{args.dataset_name}.npz"
    # download_dataset(dataset_url, dataset_path)


    # 데이터셋 로드
    print("데이터셋 로드 중...")
    # data = np.load(dataset_path, allow_pickle=True)
    # X_original, y_original = data['X'], data['y']
    X_original, y_original = load_dataset(args.dataset_name)

    print(f"원본 데이터셋 shape: {X_original.shape}")
    print(f"원본 클래스 분포 - 정상: {np.sum(y_original == 0)}, 이상: {np.sum(y_original == 1)}")

    # 데이터셋 준비: 정상 데이터와 이상 데이터 분리
    X_normal = X_original[y_original == 0]
    X_anomaly = X_original[y_original == 1]
    
    # 1. 정상 데이터 분할: 학습용(train)과 검증/테스트용(holdout)
    X_normal_train, X_normal_holdout = train_test_split(X_normal, test_size=0.4, random_state=RANDOM_SEED)
    
    # 2. 실제 이상 데이터 분할: 검증용(50%)과 테스트용(50%)
    X_anomaly_val, X_anomaly_test = train_test_split(X_anomaly, test_size=0.7, random_state=RANDOM_SEED)

    # 3. 검증 세트와 테스트 세트에 정상 데이터 추가 (holdout 데이터의 절반씩)
    X_normal_val, X_normal_test = train_test_split(X_normal_holdout, test_size=0.5, random_state=RANDOM_SEED)
    
    # Real anomaly validation set과 test set 생성
    X_val_real = np.vstack([X_normal_val, X_anomaly_val])
    y_val_real = np.concatenate([np.zeros(len(X_normal_val)), np.ones(len(X_anomaly_val))])
    
    X_test = np.vstack([X_normal_test, X_anomaly_test])
    y_test = np.concatenate([np.zeros(len(X_normal_test)), np.ones(len(X_anomaly_test))])
    
    # 각 데이터셋 셔플
    def shuffle_data(X, y):
        idx = np.random.RandomState(RANDOM_SEED).permutation(len(y))
        return X[idx], y[idx]
    
    X_val_real, y_val_real = shuffle_data(X_val_real, y_val_real)
    X_test, y_test = shuffle_data(X_test, y_test)
    
    print(f"\n데이터셋 분할 완료:")
    print(f"Train set (정상만): {X_normal_train.shape}")
    print(f"Real validation set: {X_val_real.shape}, 정상: {np.sum(y_val_real == 0)}, 이상: {np.sum(y_val_real == 1)}")
    print(f"Test set: {X_test.shape}, 정상: {np.sum(y_test == 0)}, 이상: {np.sum(y_test == 1)}")
    
    # 여러 유형의 Synthetic Anomaly 생성
    data_generator = SimpleDataGenerator(seed=RANDOM_SEED)
    synthetic_val_sets = {}
    
    for anomaly_type in ANOMALY_TYPES:
        print(f"\n{anomaly_type} 유형의 합성 이상치로 검증 세트 생성 중...")
        
        # 합성 이상치 생성 (실제 이상치와 동일한 개수로)
        synthetic_anomalies = data_generator.generate_anomalies(
            X=X_original,
            y=y_original,
            anomaly_type=anomaly_type,
            alpha=5,  # local, cluster 이상치 강도
            percentage=0.2,  # global 이상치 범위
            anomaly_count=len(X_anomaly_val)
        )
        
        # 합성 이상치로 검증 세트 생성
        X_val_synthetic = np.vstack([X_normal_val, synthetic_anomalies])
        y_val_synthetic = np.concatenate([np.zeros(len(X_normal_val)), np.ones(len(synthetic_anomalies))])
        X_val_synthetic, y_val_synthetic = shuffle_data(X_val_synthetic, y_val_synthetic)
        
        synthetic_val_sets[anomaly_type] = (X_val_synthetic, y_val_synthetic)
        
        print(f"{anomaly_type} 검증 세트: {X_val_synthetic.shape}, 정상: {np.sum(y_val_synthetic == 0)}, 이상: {np.sum(y_val_synthetic == 1)}")
    
    # t-SNE 시각화
    print("\nt-SNE 시각화 생성 중...")
    
    # 실제 이상치에 대한 시각화
    visualize_tsne(
        X_test, y_test, None,
        title='Real Anomalies t-SNE Visualization',
        filename=os.path.join(results_dir, 'real_anomalies_tsne.png'),
        anomaly_types=None
    )
    
    # 모든 합성 이상치 유형과 정상 데이터 결합
    X_all_synthetic = X_normal_test.copy()
    y_all_synthetic = np.zeros(len(X_normal_test))
    y_types = np.zeros(len(X_normal_test))
    
    # 각 합성 이상치 유형 추가
    for i, anomaly_type in enumerate(synthetic_val_sets.keys(), 1):
        X_val, y_val = synthetic_val_sets[anomaly_type]
        synthetic_anomalies = X_val[y_val == 1]
        
        X_all_synthetic = np.vstack([X_all_synthetic, synthetic_anomalies])
        y_all_synthetic = np.concatenate([y_all_synthetic, np.ones(len(synthetic_anomalies))])
        y_types = np.concatenate([y_types, np.full(len(synthetic_anomalies), i)])
    
    # 합성 이상치 유형별 시각화
    visualize_tsne(
        X_all_synthetic, y_all_synthetic, y_types,
        title='Synthetic Anomalies t-SNE Visualization',
        filename=os.path.join(results_dir, 'synthetic_anomalies_tsne.png'),
        anomaly_types=ANOMALY_TYPES
    )
    
    # PyOD 모델 선택 실험 실행
    print("\nPyOD 모델 선택 실험 실행 중...")
    run_model_selection_experiment(
        X_normal_train=X_normal_train,
        X_val_real=X_val_real, y_val_real=y_val_real,
        synthetic_val_sets=synthetic_val_sets,
        X_test=X_test, y_test=y_test,
        results_dir=results_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    main(args)