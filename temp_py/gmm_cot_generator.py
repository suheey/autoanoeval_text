# gmm_cot_generator.py
# GMM 기반 Local Anomaly 생성 + LLM Chain-of-Thought 필터링

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class GMM_CoT_AnomalyGenerator:
    """
    GMM 기반 Local Anomaly 생성 + LLM Chain-of-Thought 필터링 클래스
    
    핵심 아이디어:
    1. 정상 데이터로 GMM 학습
    2. Covariance 확장(α배)하여 경계 외곽 샘플 생성
    3. LLM CoT로 도출한 의학적 이상 조건으로 필터링
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        self.scaler = StandardScaler()
        self.gmm_normal = None
        self.feature_names = None
        self.cot_rules = None
        
    def set_feature_names(self, feature_names: List[str]):
        """특성 이름 설정 (CoT 규칙 생성을 위해)"""
        self.feature_names = feature_names
        
    def generate_cot_rules(self) -> Dict[str, Any]:
        """
        LLM Chain-of-Thought 기반 의학적 이상 조건 생성
        실제 연구에서는 LLM API를 호출하지만, 여기서는 의학적 지식 기반 규칙을 하드코딩
        
        Cardiotocography 도메인 지식:
        - 태아 심박수(LB): 정상 110-160 bpm
        - 가속(AC): 높을수록 좋음 (태아 건강)
        - 자궁수축(UC): 너무 빈번하면 위험
        - 변동성(ASTV): 너무 높거나 낮으면 위험
        - 감속(DL, DS, DP): 존재하면 위험신호
        """
        
        # Chain-of-Thought 추론 기반 이상 조건들
        cot_rules = {
            "bradycardia_with_low_variability": {
                "description": "서맥 + 낮은 변동성 → 태아 저산소증 의심",
                "conditions": [
                    ("LB", "<", 110),  # 서맥
                    ("ASTV", "<", 20), # 낮은 단기 변동성
                ],
                "reasoning": "심박수가 낮으면서 변동성도 낮으면 태아가 스트레스 상태일 가능성"
            },
            
            "tachycardia_with_decelerations": {
                "description": "빈맥 + 감속 → 태아 곤란증",
                "conditions": [
                    ("LB", ">", 160),  # 빈맥
                    ("DL", ">", 2),    # 경한 감속 존재
                ],
                "reasoning": "심박수가 높으면서 감속이 발생하면 태아에게 산소 공급 문제 가능성"
            },
            
            "excessive_contractions_low_acceleration": {
                "description": "과도한 자궁수축 + 낮은 가속 → 자궁 과활동",
                "conditions": [
                    ("UC", ">", 8),   # 과도한 자궁수축
                    ("AC", "<", 2),   # 낮은 가속
                ],
                "reasoning": "자궁수축이 너무 빈번하면서 태아 반응(가속)이 적으면 태아 스트레스"
            },
            
            "high_variability_with_movements": {
                "description": "과도한 변동성 + 태아 움직임 → 과활동성 태아",
                "conditions": [
                    ("ASTV", ">", 80),  # 높은 단기 변동성
                    ("FM", ">", 100),   # 과도한 태아 움직임
                ],
                "reasoning": "변동성과 움직임이 모두 과도하면 태아가 과자극 상태"
            },
            
            "prolonged_decelerations": {
                "description": "지속적 감속 → 급성 태아 곤란증",
                "conditions": [
                    ("DP", ">", 1),    # 지속적 감속 존재
                    ("ASTV", "<", 30), # 변동성 감소
                ],
                "reasoning": "지속적인 감속은 태아에게 심각한 위험 신호"
            },
            
            "abnormal_histogram_pattern": {
                "description": "비정상적 히스토그램 패턴 → 심박수 불안정",
                "conditions": [
                    ("Width", ">", 100),  # 넓은 히스토그램
                    ("Variance", ">", 50), # 높은 분산
                    ("Nmax", ">", 10),     # 많은 peak
                ],
                "reasoning": "히스토그램이 불안정하면 심박수 패턴의 일관성 부족"
            }
        }
        
        self.cot_rules = cot_rules
        return cot_rules
    
    def fit_gmm_normal(self, X_normal: np.ndarray, max_components: int = 10):
        """
        정상 데이터에 GMM 학습
        
        Parameters:
        - X_normal: 정상 데이터
        - max_components: 최대 컴포넌트 수
        """
        print(f"정상 데이터 GMM 학습 중... 샘플 수: {len(X_normal)}")
        
        # 데이터 정규화
        X_scaled = self.scaler.fit_transform(X_normal)
        
        # 최적 컴포넌트 수 찾기 (BIC 기준)
        n_components_list = list(range(1, min(max_components + 1, len(X_normal) // 50 + 1)))
        bic_scores = []
        
        for n_components in n_components_list:
            try:
                gmm = GaussianMixture(
                    n_components=n_components, 
                    random_state=self.seed,
                    covariance_type='full'
                ).fit(X_scaled)
                bic_scores.append(gmm.bic(X_scaled))
            except:
                # 수렴하지 않는 경우 매우 큰 BIC 점수 할당
                bic_scores.append(1e10)
        
        # 최적 컴포넌트 선택
        best_n_components = n_components_list[np.argmin(bic_scores)]
        print(f"최적 GMM 컴포넌트 수: {best_n_components}")
        
        # 최종 GMM 학습
        self.gmm_normal = GaussianMixture(
            n_components=best_n_components,
            random_state=self.seed,
            covariance_type='full',
            max_iter=200  # 수렴 안정성을 위해 반복 횟수 증가
        ).fit(X_scaled)
        
        print(f"GMM 학습 완료 - 컴포넌트: {best_n_components}, BIC: {self.gmm_normal.bic(X_scaled):.2f}")
        
    def generate_gmm_samples(self, n_samples: int, alpha: float = 5.0) -> np.ndarray:
        """
        GMM에서 covariance 확장된 샘플 생성
        
        Parameters:
        - n_samples: 생성할 샘플 수
        - alpha: covariance 확장 배율
        
        Returns:
        - 생성된 샘플 (원본 스케일)
        """
        if self.gmm_normal is None:
            raise ValueError("GMM이 학습되지 않았습니다. fit_gmm_normal()을 먼저 호출하세요.")
        
        print(f"GMM 샘플 생성 중... 요청 샘플 수: {n_samples}, alpha: {alpha}")
        
        try:
            # 확장된 covariance로 새로운 GMM 생성
            gmm_expanded = GaussianMixture(
                n_components=self.gmm_normal.n_components,
                random_state=self.seed,
                covariance_type='full'
            )
            
            # 기존 GMM 파라미터 복사
            gmm_expanded.weights_ = self.gmm_normal.weights_.copy()
            gmm_expanded.means_ = self.gmm_normal.means_.copy()
            gmm_expanded.covariances_ = self.gmm_normal.covariances_ * alpha  # covariance 확장
            gmm_expanded.precisions_cholesky_ = None  # 재계산하도록 None 설정
            
            # 내부 파라미터 재계산
            try:
                gmm_expanded._check_parameters(gmm_expanded.weights_.reshape(-1, 1))
            except:
                # 파라미터 체크 실패 시 대안 방법
                gmm_expanded.converged_ = True
                gmm_expanded.n_iter_ = 1
            
            # 샘플 생성
            samples_scaled, _ = gmm_expanded.sample(n_samples)
            
            # 원본 스케일로 변환
            samples_original = self.scaler.inverse_transform(samples_scaled)
            
            print(f"GMM 샘플 생성 완료: {samples_original.shape}")
            return samples_original
            
        except Exception as e:
            print(f"GMM 샘플 생성 중 오류: {e}")
            # 대안: 기존 GMM에서 직접 샘플링 후 노이즈 추가
            samples_scaled, _ = self.gmm_normal.sample(n_samples)
            noise = np.random.normal(0, alpha * 0.1, samples_scaled.shape)
            samples_scaled_noisy = samples_scaled + noise
            samples_original = self.scaler.inverse_transform(samples_scaled_noisy)
            print(f"대안 방법으로 샘플 생성 완료: {samples_original.shape}")
            return samples_original
    
    def apply_cot_filter(self, samples: np.ndarray, rule_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        CoT 규칙 기반 샘플 필터링
        
        Parameters:
        - samples: 필터링할 샘플들
        - rule_name: 적용할 CoT 규칙 이름
        
        Returns:
        - filtered_samples: 필터링된 샘플
        - mask: 필터링 마스크
        """
        if self.cot_rules is None:
            self.generate_cot_rules()
        
        if rule_name not in self.cot_rules:
            raise ValueError(f"알 수 없는 규칙: {rule_name}")
        
        rule = self.cot_rules[rule_name]
        print(f"\nCoT 규칙 적용: {rule_name}")
        print(f"설명: {rule['description']}")
        print(f"추론: {rule['reasoning']}")
        
        # 샘플을 DataFrame으로 변환 (필터링을 위해)
        if self.feature_names is None:
            raise ValueError("feature_names가 설정되지 않았습니다.")
        
        df_samples = pd.DataFrame(samples, columns=self.feature_names)
        
        # 조건 적용
        mask = np.ones(len(df_samples), dtype=bool)
        applied_conditions = 0
        
        for feature, operator, threshold in rule['conditions']:
            if feature not in df_samples.columns:
                print(f"경고: 특성 '{feature}'를 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            if operator == ">":
                condition = df_samples[feature] > threshold
            elif operator == "<":
                condition = df_samples[feature] < threshold
            elif operator == ">=":
                condition = df_samples[feature] >= threshold
            elif operator == "<=":
                condition = df_samples[feature] <= threshold
            else:
                print(f"지원하지 않는 연산자: {operator}. 건너뜁니다.")
                continue
            
            mask = mask & condition
            applied_conditions += 1
            print(f"  조건 '{feature} {operator} {threshold}': {condition.sum()} 샘플 만족")
        
        filtered_samples = samples[mask]
        print(f"필터링 결과: {len(samples)} → {len(filtered_samples)} 샘플 (적용된 조건: {applied_conditions}개)")
        
        return filtered_samples, mask
    
    def generate_cot_filtered_anomalies(self, 
                                       target_count: int,
                                       rule_name: str,
                                       alpha: float = 5.0,
                                       max_attempts: int = 10) -> np.ndarray:
        """
        CoT 필터링 기반 이상치 생성 (목표 개수까지)
        
        Parameters:
        - target_count: 목표 이상치 개수
        - rule_name: 적용할 CoT 규칙
        - alpha: GMM covariance 확장 배율
        - max_attempts: 최대 시도 횟수
        
        Returns:
        - 생성된 이상치 샘플
        """
        print(f"\n=== CoT 기반 이상치 생성 시작 ===")
        print(f"목표 개수: {target_count}, 규칙: {rule_name}")
        
        all_anomalies = []
        
        for attempt in range(max_attempts):
            print(f"\n--- 시도 {attempt + 1}/{max_attempts} ---")
            
            # 목표보다 많이 생성 (필터링으로 줄어들 것을 고려)
            # 지수적 증가 대신 선형 증가로 메모리 효율성 개선
            generation_count = target_count * (5 + attempt * 5)  # 5배씩 점진적 증가
            generation_count = min(generation_count, target_count * 50)  # 최대 50배로 제한
            
            try:
                # GMM 샘플 생성
                samples = self.generate_gmm_samples(generation_count, alpha)
                
                # CoT 필터링 적용
                filtered_samples, _ = self.apply_cot_filter(samples, rule_name)
                
                if len(filtered_samples) > 0:
                    all_anomalies.append(filtered_samples)
                    
                    # 현재까지 수집된 이상치 개수
                    total_collected = sum(len(anomalies) for anomalies in all_anomalies)
                    print(f"현재까지 수집된 이상치: {total_collected}")
                    
                    # 목표 달성 시 종료
                    if total_collected >= target_count:
                        break
                else:
                    print("필터링 결과가 없습니다. 다음 시도...")
                    
            except Exception as e:
                print(f"시도 {attempt + 1} 중 오류 발생: {e}")
                continue
        
        # 모든 이상치 결합
        if all_anomalies:
            final_anomalies = np.vstack(all_anomalies)
            
            # 목표 개수만큼만 반환
            if len(final_anomalies) > target_count:
                indices = np.random.choice(len(final_anomalies), target_count, replace=False)
                final_anomalies = final_anomalies[indices]
            
            print(f"\n=== 최종 결과 ===")
            print(f"요청 개수: {target_count}, 생성된 개수: {len(final_anomalies)}")
            return final_anomalies
        else:
            print("이상치 생성에 실패했습니다. 빈 배열을 반환합니다.")
            # 원본 스케일의 빈 배열 반환
            return np.array([]).reshape(0, len(self.feature_names))
    
    def analyze_generated_anomalies(self, rule_anomalies: Dict[str, np.ndarray]):
        """생성된 이상치들의 통계적 특성 분석"""
        print(f"\n=== 생성된 이상치 분석 ===")
        
        # 규칙별 통계
        for rule_name, anomalies in rule_anomalies.items():
            if len(anomalies) == 0:
                print(f"{rule_name}: 생성된 이상치 없음")
                continue
                
            print(f"\n{rule_name}:")
            print(f"  개수: {len(anomalies)}")
            
            # 주요 특성별 통계 (존재하는 특성만)
            df_anomalies = pd.DataFrame(anomalies, columns=self.feature_names)
            key_features = ['LB', 'AC', 'UC', 'ASTV', 'DL', 'Width', 'Variance']
            
            for feature in key_features:
                if feature in df_anomalies.columns:
                    mean_val = df_anomalies[feature].mean()
                    std_val = df_anomalies[feature].std()
                    print(f"  {feature}: 평균={mean_val:.2f}, 표준편차={std_val:.2f}")
    
    def visualize_anomaly_distribution(self, 
                                     X_normal: np.ndarray,
                                     rule_anomalies: Dict[str, np.ndarray],
                                     features_to_plot: List[str] = None):
        """
        정상 데이터와 생성된 이상치들의 분포 시각화
        
        Parameters:
        - X_normal: 정상 데이터
        - rule_anomalies: 규칙별 이상치 딕셔너리
        - features_to_plot: 시각화할 특성들 (None이면 처음 4개)
        
        Returns:
        - fig: matplotlib figure 객체 (저장용)
        """
        if features_to_plot is None:
            features_to_plot = self.feature_names[:4]
        
        # 정상 데이터를 DataFrame으로 변환
        df_normal = pd.DataFrame(X_normal, columns=self.feature_names)
        
        # 시각화할 규칙이 없으면 종료
        valid_rules = {k: v for k, v in rule_anomalies.items() if len(v) > 0}
        if not valid_rules:
            print("시각화할 규칙 기반 이상치가 없습니다.")
            return None
        
        # 시각화
        n_features = len(features_to_plot)
        n_rules = len(valid_rules)
        
        fig, axes = plt.subplots(n_features, n_rules + 1, 
                                figsize=(4 * (n_rules + 1), 3 * n_features))
        
        # 1차원 배열 처리
        if n_features == 1:
            axes = axes.reshape(1, -1)
        if n_rules == 0:
            return None
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i, feature in enumerate(features_to_plot):
            if feature not in self.feature_names:
                continue
                
            # 정상 데이터 분포
            try:
                axes[i, 0].hist(df_normal[feature].dropna(), bins=30, alpha=0.7, 
                               color='lightblue', label='Normal', density=True)
                axes[i, 0].set_title(f'Normal - {feature}')
                axes[i, 0].set_ylabel('Density')
                axes[i, 0].legend()
            except:
                axes[i, 0].text(0.5, 0.5, f'Error plotting {feature}', 
                               transform=axes[i, 0].transAxes, ha='center', va='center')
            
            # 각 규칙별 이상치 분포
            for j, (rule_name, anomalies) in enumerate(valid_rules.items()):
                try:
                    df_anomalies = pd.DataFrame(anomalies, columns=self.feature_names)
                    
                    # 정상 데이터 (배경)
                    axes[i, j + 1].hist(df_normal[feature].dropna(), bins=30, alpha=0.3, 
                                       color='lightblue', label='Normal', density=True)
                    
                    # 이상치 데이터
                    axes[i, j + 1].hist(df_anomalies[feature].dropna(), bins=30, alpha=0.7, 
                                       color=colors[j % len(colors)], 
                                       label=f'Anomaly', density=True)
                    
                    axes[i, j + 1].set_title(f'{rule_name[:15]} - {feature}')  # 제목 길이 제한
                    axes[i, j + 1].legend()
                    
                except Exception as e:
                    axes[i, j + 1].text(0.5, 0.5, f'Error: {str(e)[:20]}...', 
                                       transform=axes[i, j + 1].transAxes, 
                                       ha='center', va='center')
        
        plt.tight_layout()
        return fig