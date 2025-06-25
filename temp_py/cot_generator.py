import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from typing import List, Dict, Tuple, Any
import json
import random


class CoTAnoEvalGenerator:
    """
    CoT(Chain of Thought) 기반 Dataset-specific AnoEval 생성기
    다양한 프롬프트 템플릿을 통해 도메인 전문가의 지식을 활용하여
    현실적이고 다양한 이상치를 생성합니다.
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # 태아 심박수 데이터셋을 위한 도메인 지식 기반 CoT 프롬프트 템플릿
        self.cot_prompt_templates = {
            "physiological_state": {
                "description": "Given this fetal heart rate dataset, what are abnormal physiological states?",
                "conditions": [
                    "Severe bradycardia with minimal variability",
                    "Tachycardia with high variability",
                    "Prolonged deceleration patterns",
                    "Absence of accelerations with high baseline"
                ]
            },
            "temporal_deterioration": {
                "description": "What combinations may indicate deterioration over time?",
                "conditions": [
                    "Progressively decreasing variability",
                    "Increasing baseline with decreasing accelerations",
                    "Prolonged periods of minimal variability",
                    "Late decelerations with reduced recovery"
                ]
            },
            "sensor_malfunction": {
                "description": "What conditions could result from sensor failure or noise?",
                "conditions": [
                    "Extremely high ASTV with zero MSTV",
                    "Impossible physiological combinations",
                    "Zero values in multiple critical parameters",
                    "Extreme outliers in histogram features"
                ]
            },
            "emergency_situations": {
                "description": "What patterns may appear in emergency situations like fetal distress?",
                "conditions": [
                    "Severe bradycardia with absent variability",
                    "Recurrent severe decelerations",
                    "Persistent minimal variability",
                    "Sinusoidal pattern indicators"
                ]
            },
            "statistical_anomalies": {
                "description": "What statistical combinations might seem unlikely or rare?",
                "conditions": [
                    "High baseline with low variability",
                    "Normal baseline with extreme histogram skewness",
                    "High accelerations with low movements",
                    "Contradictory trend and pattern indicators"
                ]
            }
        }
        
        # 특성 이름과 정상 범위 정의 (도메인 지식 기반)
        self.feature_info = {
            'LB': {'name': 'Baseline', 'normal_range': (110, 160), 'unit': 'bpm'},
            'AC': {'name': 'Accelerations', 'normal_range': (0, 10), 'unit': 'count'},
            'FM': {'name': 'Fetal Movements', 'normal_range': (0, 500), 'unit': 'count'},
            'UC': {'name': 'Uterine Contractions', 'normal_range': (0, 10), 'unit': 'count'},
            'ASTV': {'name': 'Short Term Variability', 'normal_range': (20, 80), 'unit': 'ms'},
            'MSTV': {'name': 'Mean Short Term Variability', 'normal_range': (0.5, 7), 'unit': 'ms'},
            'ALTV': {'name': 'Long Term Variability', 'normal_range': (0, 50), 'unit': 'ms'},
            'MLTV': {'name': 'Mean Long Term Variability', 'normal_range': (0, 15), 'unit': 'ms'},
            'Variance': {'name': 'Variance', 'normal_range': (0, 100), 'unit': 'ms²'}
        }
    
    def generate_cot_conditions(self, X_normal: np.ndarray, feature_names: List[str]) -> List[Dict]:
        """
        CoT 기반으로 이상치 조건들을 생성합니다.
        
        Parameters:
        - X_normal: 정상 데이터
        - feature_names: 특성 이름 리스트
        
        Returns:
        - List of condition dictionaries
        """
        print("🧠 CoT 기반 이상치 조건 생성 중...")
        
        all_conditions = []
        
        # 각 프롬프트 템플릿에 대해 조건 생성
        for template_name, template_info in self.cot_prompt_templates.items():
            print(f"  📋 {template_name} 템플릿 처리 중...")
            
            for condition_desc in template_info["conditions"]:
                # 각 조건을 구체적인 수치 조건으로 변환
                specific_conditions = self._convert_to_specific_conditions(
                    condition_desc, X_normal, feature_names
                )
                
                for specific_condition in specific_conditions:
                    condition_dict = {
                        'template': template_name,
                        'description': condition_desc,
                        'specific_condition': specific_condition,
                        'feature_constraints': specific_condition
                    }
                    all_conditions.append(condition_dict)
        
        print(f"✅ 총 {len(all_conditions)}개의 CoT 조건 생성 완료")
        return all_conditions
    
    def _convert_to_specific_conditions(self, condition_desc: str, X_normal: np.ndarray, 
                                      feature_names: List[str]) -> List[Dict]:
        """
        텍스트 조건을 구체적인 수치 조건으로 변환합니다.
        """
        conditions = []
        
        # 정상 데이터의 통계 정보 계산
        normal_stats = {}
        for i, feature in enumerate(feature_names):
            if i < X_normal.shape[1]:
                normal_stats[feature] = {
                    'mean': np.mean(X_normal[:, i]),
                    'std': np.std(X_normal[:, i]),
                    'min': np.min(X_normal[:, i]),
                    'max': np.max(X_normal[:, i]),
                    'q25': np.percentile(X_normal[:, i], 25),
                    'q75': np.percentile(X_normal[:, i], 75)
                }
        
        # 조건별 구체적 수치 생성
        if "bradycardia" in condition_desc.lower():
            # 서맥: 낮은 기저선
            if 'LB' in normal_stats:
                conditions.append({
                    'LB': ('below', normal_stats['LB']['q25'] - normal_stats['LB']['std']),
                    'ASTV': ('below', normal_stats.get('ASTV', {}).get('q25', 30))
                })
        
        elif "tachycardia" in condition_desc.lower():
            # 빈맥: 높은 기저선
            if 'LB' in normal_stats:
                conditions.append({
                    'LB': ('above', normal_stats['LB']['q75'] + normal_stats['LB']['std']),
                    'ASTV': ('above', normal_stats.get('ASTV', {}).get('q75', 70))
                })
        
        elif "minimal variability" in condition_desc.lower():
            # 최소 변동성
            conditions.append({
                'ASTV': ('below', 20),
                'MSTV': ('below', 1.0)
            })
        
        elif "extreme outliers" in condition_desc.lower():
            # 극단적 이상값
            for feature in ['ASTV', 'MSTV', 'Variance']:
                if feature in normal_stats:
                    conditions.append({
                        feature: ('above', normal_stats[feature]['max'] * 2)
                    })
        
        elif "zero values" in condition_desc.lower():
            # 영값 조건
            conditions.append({
                'ASTV': ('equal', 0),
                'AC': ('equal', 0),
                'FM': ('equal', 0)
            })
        
        elif "high baseline" in condition_desc.lower():
            # 높은 기저선 조건
            if 'LB' in normal_stats:
                conditions.append({
                    'LB': ('above', 150),
                    'ASTV': ('below', normal_stats.get('ASTV', {}).get('mean', 50))
                })
        
        # 기본 조건이 없는 경우 랜덤 조건 생성
        if not conditions:
            conditions.append(self._generate_random_condition(normal_stats))
        
        return conditions
    
    def _generate_random_condition(self, normal_stats: Dict) -> Dict:
        """랜덤한 이상치 조건을 생성합니다."""
        condition = {}
        
        # 랜덤하게 1-3개의 특성 선택
        selected_features = random.sample(list(normal_stats.keys()), 
                                        min(3, random.randint(1, len(normal_stats))))
        
        for feature in selected_features:
            stats = normal_stats[feature]
            
            # 랜덤하게 조건 유형 선택
            condition_type = random.choice(['above', 'below'])
            
            if condition_type == 'above':
                threshold = stats['q75'] + random.uniform(0.5, 2.0) * stats['std']
            else:
                threshold = stats['q25'] - random.uniform(0.5, 2.0) * stats['std']
                threshold = max(0, threshold)  # 음수 방지
            
            condition[feature] = (condition_type, threshold)
        
        return condition
    
    def generate_soft_variants(self, base_condition: Dict, num_variants: int = 5) -> List[Dict]:
        """
        하나의 기본 조건에서 soft variant들을 생성합니다.
        
        Parameters:
        - base_condition: 기본 조건
        - num_variants: 생성할 변형 수
        
        Returns:
        - List of variant conditions
        """
        variants = []
        
        for _ in range(num_variants):
            variant = {}
            
            for feature, (operator, threshold) in base_condition.items():
                # 임계값에 노이즈 추가
                noise_factor = random.uniform(0.8, 1.2)
                new_threshold = threshold * noise_factor
                
                # 추가적인 변동성 적용
                if random.random() < 0.3:  # 30% 확률로 연산자 변경
                    if operator == 'above':
                        new_operator = 'below' if random.random() < 0.5 else 'above'
                    elif operator == 'below':
                        new_operator = 'above' if random.random() < 0.5 else 'below'
                    else:
                        new_operator = operator
                else:
                    new_operator = operator
                
                variant[feature] = (new_operator, new_threshold)
            
            variants.append(variant)
        
        return variants
    
    def generate_combination_conditions(self, conditions: List[Dict], 
                                      num_combinations: int = 10) -> List[Dict]:
        """
        두 개 이상의 조건을 조합하여 새로운 조건들을 생성합니다.
        
        Parameters:
        - conditions: 기본 조건들
        - num_combinations: 생성할 조합 수
        
        Returns:
        - List of combined conditions
        """
        combinations = []
        
        for _ in range(num_combinations):
            # 랜덤하게 2-3개의 조건 선택
            selected_conditions = random.sample(conditions, 
                                              min(len(conditions), random.randint(2, 3)))
            
            # 조건들을 조합
            combined_condition = {}
            
            for condition in selected_conditions:
                specific_cond = condition['specific_condition']
                
                for feature, constraint in specific_cond.items():
                    if feature not in combined_condition:
                        combined_condition[feature] = constraint
                    else:
                        # 기존 조건과 충돌하는 경우, 더 극단적인 조건 선택
                        existing_op, existing_val = combined_condition[feature]
                        new_op, new_val = constraint
                        
                        if existing_op == new_op:
                            # 같은 연산자인 경우 더 극단적인 값 선택
                            if existing_op == 'above':
                                combined_condition[feature] = (existing_op, max(existing_val, new_val))
                            elif existing_op == 'below':
                                combined_condition[feature] = (existing_op, min(existing_val, new_val))
                        else:
                            # 다른 연산자인 경우 랜덤 선택
                            combined_condition[feature] = random.choice([
                                (existing_op, existing_val), 
                                (new_op, new_val)
                            ])
            
            combination_dict = {
                'template': 'combination',
                'description': f'Combination of {len(selected_conditions)} conditions',
                'specific_condition': combined_condition,
                'feature_constraints': combined_condition
            }
            
            combinations.append(combination_dict)
        
        return combinations
    
    def apply_semantic_perturbation(self, X_normal: np.ndarray, 
                                   num_samples: int = 100) -> np.ndarray:
        """
        의미적 perturbation을 적용하여 이상치를 생성합니다.
        
        Parameters:
        - X_normal: 정상 샘플들
        - num_samples: 생성할 이상치 수
        
        Returns:
        - 생성된 이상치들
        """
        print("🔄 의미적 perturbation 적용 중...")
        
        # 정상 샘플들 중 랜덤 선택
        selected_indices = np.random.choice(len(X_normal), 
                                          min(num_samples, len(X_normal)), 
                                          replace=False)
        selected_samples = X_normal[selected_indices]
        
        perturbed_samples = []
        
        for sample in selected_samples:
            # 각 특성에 대해 도메인 지식 기반 perturbation 적용
            perturbed_sample = sample.copy()
            
            # 랜덤하게 1-3개의 특성을 선택하여 변형
            num_features_to_perturb = random.randint(1, min(3, len(sample)))
            features_to_perturb = random.sample(range(len(sample)), num_features_to_perturb)
            
            for feature_idx in features_to_perturb:
                original_value = sample[feature_idx]
                
                # 특성별 perturbation 전략
                perturbation_strategies = [
                    lambda x: x * random.uniform(1.5, 3.0),  # 증가
                    lambda x: x * random.uniform(0.1, 0.5),  # 감소
                    lambda x: x + random.uniform(50, 100),   # 상수 추가
                    lambda x: max(0, x - random.uniform(20, 50)),  # 상수 감소
                    lambda x: 0 if random.random() < 0.3 else x,  # 가끔 0으로 설정
                ]
                
                strategy = random.choice(perturbation_strategies)
                perturbed_sample[feature_idx] = strategy(original_value)
            
            perturbed_samples.append(perturbed_sample)
        
        return np.array(perturbed_samples)
    
    def generate_cot_based_anomalies(self, X_normal: np.ndarray, feature_names: List[str],
                                    num_anomalies: int = 1000) -> Tuple[np.ndarray, List[Dict]]:
        """
        CoT 기반으로 다양한 이상치들을 생성합니다.
        
        Parameters:
        - X_normal: 정상 데이터
        - feature_names: 특성 이름들
        - num_anomalies: 생성할 이상치 수
        
        Returns:
        - 생성된 이상치들과 사용된 조건들
        """
        print("🎯 CoT 기반 이상치 생성 시작...")
        
        # 1. CoT 조건들 생성
        cot_conditions = self.generate_cot_conditions(X_normal, feature_names)
        
        # 2. Soft variants 생성
        print("🎨 Soft variants 생성 중...")
        all_conditions = cot_conditions.copy()
        
        for condition in cot_conditions[:10]:  # 처음 10개 조건에 대해서만
            variants = self.generate_soft_variants(condition['specific_condition'], 3)
            for variant in variants:
                variant_dict = {
                    'template': f"{condition['template']}_variant",
                    'description': f"Variant of {condition['description']}",
                    'specific_condition': variant,
                    'feature_constraints': variant
                }
                all_conditions.append(variant_dict)
        
        # 3. 조합 조건들 생성
        print("🧬 조합 조건들 생성 중...")
        combination_conditions = self.generate_combination_conditions(cot_conditions, 15)
        all_conditions.extend(combination_conditions)
        
        # 4. 의미적 perturbation 적용
        print("✏️ 의미적 perturbation 적용 중...")
        semantic_anomalies = self.apply_semantic_perturbation(X_normal, num_anomalies // 4)
        
        # 5. 조건 기반 이상치 생성
        print("🔧 조건 기반 이상치 생성 중...")
        condition_based_anomalies = []
        
        remaining_anomalies = num_anomalies - len(semantic_anomalies)
        conditions_per_anomaly = max(1, len(all_conditions) // remaining_anomalies)
        
        for i in range(remaining_anomalies):
            condition_idx = i % len(all_conditions)
            condition = all_conditions[condition_idx]
            
            # 조건에 맞는 이상치 생성
            anomaly = self._generate_anomaly_from_condition(
                condition['specific_condition'], X_normal, feature_names
            )
            
            if anomaly is not None:
                condition_based_anomalies.append(anomaly)
        
        # 6. 모든 이상치 결합
        all_anomalies = []
        
        if len(semantic_anomalies) > 0:
            all_anomalies.extend(semantic_anomalies)
        
        if len(condition_based_anomalies) > 0:
            all_anomalies.extend(condition_based_anomalies)
        
        # 부족한 경우 추가 생성
        while len(all_anomalies) < num_anomalies:
            # 랜덤 조건으로 추가 이상치 생성
            random_condition = random.choice(all_conditions)
            anomaly = self._generate_anomaly_from_condition(
                random_condition['specific_condition'], X_normal, feature_names
            )
            if anomaly is not None:
                all_anomalies.append(anomaly)
        
        # 요청된 수만큼만 반환
        final_anomalies = np.array(all_anomalies[:num_anomalies])
        
        print(f"✅ CoT 기반 이상치 생성 완료: {len(final_anomalies)}개")
        print(f"📊 사용된 조건 수: {len(all_conditions)}개")
        
        return final_anomalies, all_conditions
    
    def _generate_anomaly_from_condition(self, condition: Dict, X_normal: np.ndarray,
                                       feature_names: List[str]) -> np.ndarray:
        """
        주어진 조건에 맞는 이상치를 생성합니다.
        """
        try:
            # 정상 샘플 중 하나를 기반으로 시작
            base_sample = X_normal[np.random.randint(0, len(X_normal))].copy()
            
            # 조건에 따라 특성값 수정
            for feature, (operator, threshold) in condition.items():
                if feature in feature_names:
                    feature_idx = feature_names.index(feature)
                    
                    if operator == 'above':
                        base_sample[feature_idx] = threshold + random.uniform(0, threshold * 0.1)
                    elif operator == 'below':
                        base_sample[feature_idx] = max(0, threshold - random.uniform(0, threshold * 0.1))
                    elif operator == 'equal':
                        base_sample[feature_idx] = threshold
            
            return base_sample
            
        except Exception as e:
            print(f"⚠️ 조건 기반 이상치 생성 중 오류: {e}")
            return None


class EnhancedDataGenerator(SimpleDataGenerator):
    """
    CoT 기반 이상치 생성을 포함한 확장된 데이터 생성기
    """
    
    def __init__(self, seed=42):
        super().__init__(seed)
        self.cot_generator = CoTAnoEvalGenerator(seed)
    
    def generate_anomalies(self, X, y, anomaly_type, alpha=5, percentage=0.2, anomaly_count=None):
        """
        확장된 이상치 생성 함수 (CoT 기반 이상치 포함)
        """
        if anomaly_type == 'cot_based':
            # CoT 기반 이상치 생성
            X_normal = X[y == 0]
            
            # 특성 이름 생성 (실제 데이터셋에 맞게 조정 필요)
            feature_names = ['LB', 'AC', 'FM', 'UC', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 
                           'DL', 'DS', 'DP', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 
                           'Mode', 'Mean', 'Median', 'Variance', 'Tendency']
            
            # 데이터 차원에 맞게 특성 이름 조정
            feature_names = feature_names[:X.shape[1]]
            
            if anomaly_count is None:
                anomaly_count = np.sum(y == 1)
            
            cot_anomalies, conditions_used = self.cot_generator.generate_cot_based_anomalies(
                X_normal, feature_names, anomaly_count
            )
            
            print(f"🎯 CoT 기반 이상치 생성 완료: {len(cot_anomalies)}개")
            print(f"📋 사용된 조건 유형: {len(set([c['template'] for c in conditions_used]))}개")
            
            return cot_anomalies
        
        else:
            # 기존 방식으로 이상치 생성
            return super().generate_anomalies(X, y, anomaly_type, alpha, percentage, anomaly_count)