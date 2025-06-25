import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import urllib.request

# 결과 저장 디렉토리 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"tsne_visualizations_{timestamp}"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 설정
RANDOM_SEED = 42
ANOMALY_TYPES = ['local', 'cluster', 'global']  # 원하는 3가지 이상치 유형

# 데이터셋 다운로드 함수
def download_dataset(url, filename):
    if not os.path.exists(filename):
        print(f"{filename} 다운로드 중...")
        urllib.request.urlretrieve(url, filename)
        print(f"{filename} 다운로드 완료!")
    else:
        print(f"{filename}이 이미 존재합니다.")

# 데이터셋 다운로드
dataset_path = '9_census.npz'
dataset_url = "https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/6_cardio.npz"
download_dataset(dataset_url, dataset_path)

# 데이터셋 로드
print("데이터셋 로드 중...")
data = np.load(dataset_path, allow_pickle=True)
X_original, y_original = data['X'], data['y']

print(f"원본 데이터셋 shape: {X_original.shape}")
print(f"원본 클래스 분포 - 정상: {np.sum(y_original == 0)}, 이상: {np.sum(y_original == 1)}")

# ADBench의 DataGenerator에서 필요한 부분만 가져온 간소화 버전
class SimpleDataGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_realistic_synthetic(self, X, y, realistic_synthetic_mode, alpha=5, percentage=0.1):
        """
        합성 이상치 생성 함수
        
        Parameters:
        - X: 입력 데이터
        - y: 레이블 (0: 정상, 1: 이상)
        - realistic_synthetic_mode: 이상치 유형 ('local', 'cluster', 'global')
        - alpha: local, cluster 이상치의 스케일링 파라미터
        - percentage: global 이상치의 범위 파라미터
        
        Returns:
        - X_synthetic: 합성 이상치가 포함된 특성
        - y_synthetic: 합성 이상치가 포함된 레이블
        """
        from sklearn.mixture import GaussianMixture
        
        # 정상 데이터와 이상 데이터 개수 계산
        pts_n = len(np.where(y == 0)[0])
        pts_a = len(np.where(y == 1)[0])
        
        # 정상 데이터만 사용
        X_normal = X[y == 0]
        
        # 정상 데이터로 모델 학습 및 합성 정상 데이터 생성
        if realistic_synthetic_mode in ['local', 'cluster', 'global']:
            # GMM의 최적 컴포넌트 수 찾기
            metric_list = []
            n_components_list = list(range(1, min(10, len(X_normal) // 50 + 1)))
            
            for n_components in n_components_list:
                gm = GaussianMixture(n_components=n_components, random_state=self.seed).fit(X_normal)
                metric_list.append(gm.bic(X_normal))
            
            best_n_components = n_components_list[np.argmin(metric_list)]
            print(f"최적 GMM 컴포넌트 수: {best_n_components}")
            
            # 최적 컴포넌트로 GMM 학습
            gm = GaussianMixture(n_components=best_n_components, random_state=self.seed).fit(X_normal)
            
            # 합성 정상 데이터 생성
            X_synthetic_normal = gm.sample(pts_n)[0]
        
        # 합성 이상 데이터 생성
        if realistic_synthetic_mode == 'local':
            # local 이상치: 동일한 위치, 더 큰 공분산
            gm_anomaly = GaussianMixture(n_components=best_n_components, random_state=self.seed)
            gm_anomaly.weights_ = gm.weights_
            gm_anomaly.means_ = gm.means_
            gm_anomaly.covariances_ = gm.covariances_ * alpha
            gm_anomaly.precisions_cholesky_ = None  # 재계산하도록 None 설정
            
            X_synthetic_anomalies = gm_anomaly.sample(pts_a)[0]
            
        elif realistic_synthetic_mode == 'cluster':
            # cluster 이상치: 다른 위치(평균), 동일한 공분산
            gm_anomaly = GaussianMixture(n_components=best_n_components, random_state=self.seed)
            gm_anomaly.weights_ = gm.weights_
            gm_anomaly.means_ = gm.means_ * alpha
            gm_anomaly.covariances_ = gm.covariances_
            gm_anomaly.precisions_cholesky_ = None  # 재계산하도록 None 설정
            
            X_synthetic_anomalies = gm_anomaly.sample(pts_a)[0]
            
        elif realistic_synthetic_mode == 'global':
            # global 이상치: 정상 데이터 범위를 벗어난 균일 분포
            X_synthetic_anomalies = []
            
            for i in range(X_synthetic_normal.shape[1]):
                low = np.min(X_synthetic_normal[:, i]) * (1 + percentage)
                high = np.max(X_synthetic_normal[:, i]) * (1 + percentage)
                
                # 값의 범위가 너무 작은 경우 확장
                if high - low < 1e-5:
                    low -= 0.1
                    high += 0.1
                
                X_synthetic_anomalies.append(np.random.uniform(low=low, high=high, size=pts_a))
            
            X_synthetic_anomalies = np.array(X_synthetic_anomalies).T
        
        # 합성 데이터 결합
        X_synthetic = np.vstack([X_synthetic_normal, X_synthetic_anomalies])
        y_synthetic = np.concatenate([np.zeros(X_synthetic_normal.shape[0]), np.ones(X_synthetic_anomalies.shape[0])])
        
        # 데이터 셔플
        idx = np.random.permutation(len(y_synthetic))
        X_synthetic = X_synthetic[idx]
        y_synthetic = y_synthetic[idx]
        
        return X_synthetic, y_synthetic, X_synthetic_anomalies

# 각 이상치 유형별 테스트 데이터 생성
def generate_anomalies(anomaly_type, alpha=5, percentage=0.2, anomaly_count=None):
    print(f"\n{'='*50}\n{anomaly_type} 유형의 이상치 생성 중...\n{'='*50}")
    
    # 원본 데이터 복사
    X = X_original.copy()
    y = y_original.copy()
    
    # 이상치 개수가 지정되지 않은 경우, 원본 데이터의 이상치 개수 사용
    if anomaly_count is None:
        anomaly_count = np.sum(y == 1)
    
    # 합성 이상치 생성
    generator = SimpleDataGenerator(seed=RANDOM_SEED)
    try:
        # 추가 파라미터로 X_synthetic_anomalies를 받아옴
        _, _, X_anomalies = generator.generate_realistic_synthetic(
            X=X, 
            y=y, 
            realistic_synthetic_mode=anomaly_type,
            alpha=alpha, 
            percentage=percentage
        )
        
        # 필요한 개수만큼만 이상치 추출
        if len(X_anomalies) > anomaly_count:
            indices = np.random.choice(len(X_anomalies), anomaly_count, replace=False)
            X_anomalies = X_anomalies[indices]
        
        print(f"{anomaly_type} 이상치 생성 완료: {len(X_anomalies)} 개")
        return X_anomalies
    
    except Exception as e:
        print(f"이상치 생성 중 오류 발생: {e}")
        return None

# 원본 정상 데이터와 여러 유형의 이상치를 결합하여 테스트 세트 생성
def create_combined_test_set():
    # 원본 데이터에서 정상 데이터만 추출
    X_normal = X_original[y_original == 0]
    
    # 각 유형별 이상치 개수 계산 (원본 이상치 개수의 1/3씩)
    original_anomaly_count = np.sum(y_original == 1)
    anomaly_count_per_type = max(1, original_anomaly_count // len(ANOMALY_TYPES))
    
    print(f"각 유형별 이상치 개수: {anomaly_count_per_type}")
    
    # 각 유형별 이상치 생성
    anomalies_dict = {}
    all_anomalies = []
    
    for anomaly_type in ANOMALY_TYPES:
        anomalies = generate_anomalies(
            anomaly_type=anomaly_type,
            alpha=5,             # local, cluster 이상치 강도
            percentage=0.2,      # global 이상치 범위
            anomaly_count=anomaly_count_per_type
        )
        
        if anomalies is not None:
            anomalies_dict[anomaly_type] = anomalies
            all_anomalies.append(anomalies)
    
    if not all_anomalies:
        print("이상치 생성에 실패했습니다.")
        return None, None, None
    
    # 모든 이상치 결합
    X_all_anomalies = np.vstack(all_anomalies)
    
    # 정상 데이터와 이상치 결합
    X_test = np.vstack([X_normal, X_all_anomalies])
    y_test = np.concatenate([
        np.zeros(len(X_normal)),
        np.ones(len(X_all_anomalies))
    ])
    
    # 각 이상치 유형에 대한 레이블 생성 (정상: 0, local: 1, cluster: 2, global: 3 등)
    y_types = np.zeros(len(X_test))
    
    start_idx = len(X_normal)
    for i, anomaly_type in enumerate(ANOMALY_TYPES, 1):
        if anomaly_type in anomalies_dict:
            end_idx = start_idx + len(anomalies_dict[anomaly_type])
            y_types[start_idx:end_idx] = i
            start_idx = end_idx
    
    # 데이터 셔플
    idx = np.random.permutation(len(y_test))
    X_test = X_test[idx]
    y_test = y_test[idx]
    y_types = y_types[idx]
    
    print(f"결합된 테스트 세트 생성 완료 - 크기: {X_test.shape}")
    print(f"클래스 분포 - 정상: {np.sum(y_test == 0)}, 이상: {np.sum(y_test == 1)}")
    
    return X_test, y_test, y_types

# 통합 테스트 세트 생성
X_test, y_test, y_types = create_combined_test_set()

if X_test is not None:
    # 여러 유형의 이상치가 포함된 테스트 세트에 대한 t-SNE 시각화
    print("\nt-SNE 시각화 생성 중...")
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)
    X_test_tsne = tsne.fit_transform(X_test)
    
    # 이상치 유형별 t-SNE 시각화 (정상 vs 여러 이상치 유형)
    plt.figure(figsize=(14, 10))
    
    # 색상 및 라벨 설정
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
    markers = ['o', '^', 's', 'D', '*']
    labels = ['Normal'] + ANOMALY_TYPES
    
    # 정상 데이터 그리기
    plt.scatter(
        X_test_tsne[y_types == 0, 0], X_test_tsne[y_types == 0, 1],
        c=colors[0], label=labels[0], alpha=0.5, s=10, marker=markers[0]
    )
    
    # 각 이상치 유형 그리기
    for i, anomaly_type in enumerate(ANOMALY_TYPES, 1):
        plt.scatter(
            X_test_tsne[y_types == i, 0], X_test_tsne[y_types == i, 1],
            c=colors[i], label=f'{anomaly_type.capitalize()} Anomaly', alpha=0.8, s=20, marker=markers[min(i, len(markers)-1)]
        )
    
    plt.title('test t-SNE', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 시각화 저장
    combined_tsne_filename = os.path.join(results_dir, 'combined_anomaly_types_tsne.png')
    plt.savefig(combined_tsne_filename, dpi=300)
    plt.close()
    print(f"통합 이상치 시각화가 {combined_tsne_filename}에 저장되었습니다")
    
    # 정상/이상치 구분만 보여주는 시각화 (이상치 유형 구분 없이)
    plt.figure(figsize=(14, 10))
    
    plt.scatter(
        X_test_tsne[y_test == 0, 0], X_test_tsne[y_test == 0, 1],
        c='blue', label='Normal', alpha=0.5, s=10
    )
    plt.scatter(
        X_test_tsne[y_test == 1, 0], X_test_tsne[y_test == 1, 1],
        c='red', label='Anomaly', alpha=0.8, s=20
    )
    
    plt.title('test t-SNE', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 시각화 저장
    binary_tsne_filename = os.path.join(results_dir, 'binary_anomaly_tsne.png')
    plt.savefig(binary_tsne_filename, dpi=300)
    plt.close()
    print(f"이진 이상치 시각화가 {binary_tsne_filename}에 저장되었습니다")

print(f"\n모든 시각화가 {results_dir} 디렉토리에 저장되었습니다")