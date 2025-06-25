import numpy as np
from sklearn.mixture import GaussianMixture

class SimpleDataGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        print("🐌 CPU 모드로 데이터 생성기 초기화")
    
    def generate_realistic_synthetic(self, X, y, realistic_synthetic_mode, alpha=5, percentage=0.1):
        """합성 이상치 생성 함수"""
        print(f"🔬 {realistic_synthetic_mode} 모드로 합성 데이터 생성")
        
        # 정상 데이터와 이상 데이터 개수 계산
        pts_n = len(np.where(y == 0)[0])
        pts_a = len(np.where(y == 1)[0])
        
        # 정상 데이터만 사용
        X_normal = X[y == 0]
        
        # 대용량 데이터 처리 최적화
        if len(X_normal) > 50000:
            print(f"⚡ 대용량 데이터 감지 ({len(X_normal):,}개). 샘플링 적용")
            indices = np.random.choice(len(X_normal), 50000, replace=False)
            X_normal = X_normal[indices]
        
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
            print("🔬 Local 이상치 생성...")
            X_synthetic_anomalies = self._generate_local_anomalies(gm, pts_a, alpha)
            
        elif realistic_synthetic_mode == 'cluster':
            print("🔬 Cluster 이상치 생성...")
            X_synthetic_anomalies = self._generate_cluster_anomalies(gm, pts_a, alpha)
            
        elif realistic_synthetic_mode == 'global':
            print("🔬 Global 이상치 생성...")
            X_synthetic_anomalies = self._generate_global_anomalies(X_normal, pts_a, percentage)
        
        # 합성 데이터 결합
        X_synthetic = np.vstack([X_synthetic_normal, X_synthetic_anomalies])
        y_synthetic = np.concatenate([np.zeros(X_synthetic_normal.shape[0]), np.ones(X_synthetic_anomalies.shape[0])])
        
        # 데이터 셔플
        idx = np.random.permutation(len(y_synthetic))
        X_synthetic = X_synthetic[idx]
        y_synthetic = y_synthetic[idx]
        
        return X_synthetic, y_synthetic, X_synthetic_anomalies
    
    def _generate_local_anomalies(self, gm, pts_a, alpha):
        """Local 이상치 생성 - 동일한 위치, 더 큰 공분산"""
        gm_anomaly = GaussianMixture(n_components=gm.n_components, random_state=self.seed)
        gm_anomaly.weights_ = gm.weights_
        gm_anomaly.means_ = gm.means_
        gm_anomaly.covariances_ = gm.covariances_ * alpha
        gm_anomaly.precisions_cholesky_ = None
        
        X_anomalies = gm_anomaly.sample(pts_a)[0]
        return X_anomalies
    
    def _generate_cluster_anomalies(self, gm, pts_a, alpha):
        """Cluster 이상치 생성 - 다른 위치(평균), 동일한 공분산"""
        gm_anomaly = GaussianMixture(n_components=gm.n_components, random_state=self.seed)
        gm_anomaly.weights_ = gm.weights_
        gm_anomaly.means_ = gm.means_ * alpha
        gm_anomaly.covariances_ = gm.covariances_
        gm_anomaly.precisions_cholesky_ = None
        
        X_anomalies = gm_anomaly.sample(pts_a)[0]
        return X_anomalies
    
    def _generate_global_anomalies(self, X_normal, pts_a, percentage):
        """Global 이상치 생성 - 정상 데이터 범위를 벗어난 균일 분포"""
        X_anomalies = []
        
        for i in range(X_normal.shape[1]):
            low = np.min(X_normal[:, i]) * (1 + percentage)
            high = np.max(X_normal[:, i]) * (1 + percentage)
            
            if high - low < 1e-5:
                low -= 0.1
                high += 0.1
            
            X_anomalies.append(np.random.uniform(low=low, high=high, size=pts_a))
        
        return np.array(X_anomalies).T
    
    def _generate_discrepancy_anomalies(self, X_normal, anomaly_count):
        """Discrepancy 이상치 생성"""
        normal_mean = np.mean(X_normal, axis=0)
        random_indices = np.random.choice(len(X_normal), anomaly_count, replace=False)
        X_anomalies = 2 * normal_mean - X_normal[random_indices]
        return X_anomalies
    
    def generate_anomalies(self, X, y, anomaly_type, alpha=5, percentage=0.2, anomaly_count=None):
        """특정 유형의 이상치 생성"""
        print(f"\n🔬 {anomaly_type} 유형의 이상치 생성 중...")
        
        # 이상치 개수가 지정되지 않은 경우, 원본 데이터의 이상치 개수 사용
        if anomaly_count is None:
            anomaly_count = np.sum(y == 1)
            
        print(f"생성할 이상치 개수: {anomaly_count:,}")
        
        try:
            if anomaly_type == 'discrepancy':
                # Discrepancy 이상치
                X_normal = X[y == 0]
                
                # 대용량 데이터 최적화
                if len(X_normal) > 50000:
                    print(f"⚡ 대용량 정상 데이터 감지 ({len(X_normal):,}개). 샘플링 적용")
                    indices = np.random.choice(len(X_normal), 50000, replace=False)
                    X_normal = X_normal[indices]
                
                X_anomalies = self._generate_discrepancy_anomalies(X_normal, anomaly_count)
                
            else:
                # 기존 방법 사용
                _, _, X_anomalies = self.generate_realistic_synthetic(
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
                
            print(f"✅ {anomaly_type} 이상치 생성 완료: {len(X_anomalies):,}개")
            print(f"📊 이상치 차원: {X_anomalies.shape}")
            
            # 차원 검증
            if X_anomalies.shape[1] != X.shape[1]:
                print(f"⚠️ 차원 불일치 감지. 조정 중...")
                if len(X_anomalies.shape) == 1:
                    X_anomalies = X_anomalies.reshape(-1, X.shape[1])
                
            return X_anomalies
        
        except Exception as e:
            print(f"❌ 이상치 생성 중 오류 발생: {e}")
            return self._generate_fallback_anomalies(X, y, anomaly_count)
    
    def _generate_fallback_anomalies(self, X, y, anomaly_count):
        """오류 시 폴백 이상치 생성"""
        X_fallback = X[y == 1]
        if len(X_fallback) > 0:
            if len(X_fallback) >= anomaly_count:
                return X_fallback[:anomaly_count]
            else:
                indices = np.random.choice(len(X_fallback), anomaly_count, replace=True)
                return X_fallback[indices]
        else:
            # 정상 데이터에 노이즈 추가
            X_normal = X[y == 0]
            indices = np.random.choice(len(X_normal), anomaly_count, replace=False)
            X_noise = X_normal[indices] + np.random.normal(0, np.std(X_normal) * 2, (anomaly_count, X.shape[1]))
            return X_noise