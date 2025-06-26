import numpy as np
import pandas as pd
import json
import re
import os
from typing import List, Dict, Any, Tuple, Callable
from .base_generator import BaseAnomalyGenerator

class RuleBasedAnomalyGenerator(BaseAnomalyGenerator):
    """규칙 기반 합성 이상치 생성기 (수학적 조건을 직접 코드로 구현)"""
    
    def __init__(self, seed: int = 42):
        super().__init__(seed)
        print(f"🔧 규칙 기반 이상치 생성기 초기화")
        print(f"   💡 모드: 수학적 조건을 코드로 직접 구현")
    
    def parse_mathematical_condition(self, condition: str, feature_names: List[str]) -> Callable:
        """수학적 조건 문자열을 Python 함수로 변환"""
        
        # 조건 정리
        condition = condition.strip()
        condition = re.sub(r'\s+', ' ', condition)
        
        # AND/OR 연산자 처리
        condition = condition.replace(' AND ', ' and ')
        condition = condition.replace(' OR ', ' or ')
        
        # 특성 이름을 data[index] 형태로 변환
        for i, name in enumerate(feature_names):
            # 정확한 매칭을 위해 단어 경계 사용
            pattern = r'\b' + re.escape(name) + r'\b'
            condition = re.sub(pattern, f'data[{i}]', condition)
        
        try:
            # 안전한 함수 생성
            def condition_func(data):
                # 로컬 변수로 안전하게 실행
                return eval(condition, {"__builtins__": {}}, {"data": data, "abs": abs, "min": min, "max": max})
            
            # 테스트 실행
            test_data = np.random.random(len(feature_names))
            condition_func(test_data)
            
            print(f"✅ 조건 파싱 성공: {condition}")
            return condition_func
            
        except Exception as e:
            print(f"❌ 조건 파싱 실패: {condition}")
            print(f"   오류: {e}")
            return None
    
    def extract_condition_bounds(self, condition: str, feature_names: List[str]) -> Dict[str, Tuple[float, float]]:
        """조건에서 각 특성의 범위 추출"""
        bounds = {}
        
        # 각 특성에 대한 조건 추출
        for name in feature_names:
            bounds[name] = (float('-inf'), float('inf'))  # 기본값
        
        # 간단한 패턴 매칭으로 범위 추출
        # 예: "LB > 160" → LB의 하한을 160으로 설정
        patterns = [
            (r'(\w+)\s*>\s*([\d.]+)', lambda m: (float(m.group(2)), float('inf'))),
            (r'(\w+)\s*>=\s*([\d.]+)', lambda m: (float(m.group(2)), float('inf'))),
            (r'(\w+)\s*<\s*([\d.]+)', lambda m: (float('-inf'), float(m.group(2)))),
            (r'(\w+)\s*<=\s*([\d.]+)', lambda m: (float('-inf'), float(m.group(2)))),
            (r'(\w+)\s*==\s*([\d.]+)', lambda m: (float(m.group(2)), float(m.group(2)))),
        ]
        
        for pattern, bound_func in patterns:
            matches = re.finditer(pattern, condition)
            for match in matches:
                feature_name = match.group(1)
                if feature_name in feature_names:
                    new_bounds = bound_func(match)
                    current_bounds = bounds[feature_name]
                    # 교집합 계산
                    bounds[feature_name] = (
                        max(current_bounds[0], new_bounds[0]),
                        min(current_bounds[1], new_bounds[1])
                    )
        
        return bounds
    
    def generate_sample_for_condition(self, condition_func: Callable, feature_stats: Dict[str, Dict],
                                    feature_names: List[str], bounds: Dict[str, Tuple[float, float]],
                                    max_attempts: int = 1000) -> np.ndarray:
        """특정 조건을 만족하는 샘플 하나 생성"""
        
        for attempt in range(max_attempts):
            sample = np.zeros(len(feature_names))
            
            for i, name in enumerate(feature_names):
                stats = feature_stats[name]
                lower_bound, upper_bound = bounds[name]
                
                # 통계 기반 범위와 조건 범위의 교집합
                min_val = max(stats['min'], lower_bound)
                max_val = min(stats['max'], upper_bound)
                
                if min_val > max_val:
                    # 불가능한 조건
                    sample[i] = stats['mean']
                elif lower_bound == upper_bound and lower_bound != float('-inf'):
                    # 정확한 값 지정
                    sample[i] = lower_bound
                else:
                    # 범위 내에서 랜덤 생성
                    if min_val == float('-inf'):
                        min_val = stats['min']
                    if max_val == float('inf'):
                        max_val = stats['max']
                    
                    # 정규분포 기반 생성 (범위 제한)
                    if name in ['LB', 'AC', 'ASTV', 'MSTV', 'ALTV', 'MLTV']:  # 연속형
                        sample[i] = np.random.uniform(min_val, max_val)
                    else:  # 정수형
                        sample[i] = np.random.randint(max(0, int(min_val)), int(max_val) + 1)
            
            # 조건 확인
            try:
                if condition_func(sample):
                    return sample
            except:
                continue
        
        # 최대 시도 후에도 실패하면 제약 조건만 만족하는 샘플 반환
        print(f"⚠️ 조건 만족 샘플 생성 실패, 근사 샘플 반환")
        return sample
    
    def generate_anomalies_from_conditions(self, X: np.ndarray, y: np.ndarray,
                                         anomaly_patterns: Dict[str, Any],
                                         anomaly_count: int = None,
                                         feature_names: List[str] = None) -> np.ndarray:
        """수학적 조건들을 직접 구현하여 이상치 생성"""
        
        if anomaly_count is None:
            anomaly_count = np.sum(y == 1)
        
        print(f"🔧 규칙 기반 이상치 생성 중... ({anomaly_count:,}개)")
        
        # 정상 데이터 통계
        X_normal = X[y == 0]
        n_features = X_normal.shape[1]
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        # 특성별 통계 계산
        feature_stats = {}
        for i, name in enumerate(feature_names):
            feature_stats[name] = {
                'mean': float(np.mean(X_normal[:, i])),
                'std': float(np.std(X_normal[:, i])),
                'min': float(np.min(X_normal[:, i])),
                'max': float(np.max(X_normal[:, i]))
            }
        
        # 조건들 파싱
        conditions = anomaly_patterns.get('anomaly_conditions', [])
        parsed_conditions = []
        
        for i, condition_info in enumerate(conditions):
            condition_str = condition_info.get('condition', '')
            print(f"🔍 조건 {i+1} 파싱: {condition_str}")
            
            condition_func = self.parse_mathematical_condition(condition_str, feature_names)
            bounds = self.extract_condition_bounds(condition_str, feature_names)
            
            if condition_func is not None:
                parsed_conditions.append({
                    'func': condition_func,
                    'bounds': bounds,
                    'info': condition_info
                })
            else:
                print(f"⚠️ 조건 {i+1} 건너뛰기")
        
        if not parsed_conditions:
            print("❌ 파싱 가능한 조건이 없습니다")
            return np.array([])
        
        # 각 조건별로 균등하게 샘플 생성
        samples_per_condition = anomaly_count // len(parsed_conditions)
        remaining_samples = anomaly_count % len(parsed_conditions)
        
        all_anomalies = []
        
        for i, condition_info in enumerate(parsed_conditions):
            # 이 조건에서 생성할 샘플 수
            current_count = samples_per_condition
            if i < remaining_samples:
                current_count += 1
            
            print(f"📊 조건 {i+1}에서 {current_count}개 샘플 생성 중...")
            
            condition_samples = []
            for j in range(current_count):
                sample = self.generate_sample_for_condition(
                    condition_info['func'],
                    feature_stats,
                    feature_names,
                    condition_info['bounds']
                )
                condition_samples.append(sample)
            
            if condition_samples:
                all_anomalies.extend(condition_samples)
                print(f"✅ 조건 {i+1}: {len(condition_samples)}개 생성 완료")
        
        if all_anomalies:
            anomaly_data = np.array(all_anomalies)
            print(f"✅ 규칙 기반 이상치 생성 완료: {len(anomaly_data):,}개")
            
            # 검증: 생성된 샘플들이 실제로 조건을 만족하는지 확인
            validation_count = 0
            for sample in anomaly_data:
                for condition_info in parsed_conditions:
                    try:
                        if condition_info['func'](sample):
                            validation_count += 1
                            break
                    except:
                        continue
            
            print(f"🔍 검증: {validation_count}/{len(anomaly_data)}개 샘플이 조건 만족 ({validation_count/len(anomaly_data)*100:.1f}%)")
            
            return anomaly_data
        else:
            print("❌ 생성된 이상치 없음")
            return np.array([])
    
    def generate_anomalies(self, X: np.ndarray, y: np.ndarray,
                          anomaly_patterns: Dict[str, Any],
                          anomaly_count: int = None,
                          feature_names: List[str] = None,
                          dataset_name: str = "Unknown") -> np.ndarray:
        """규칙 기반 이상치 생성 메인 함수"""
        
        print(f"🔧 규칙 기반 이상치 생성 시작")
        print(f"📊 대상 조건 개수: {len(anomaly_patterns.get('anomaly_conditions', []))}개")
        
        return self.generate_anomalies_from_conditions(
            X, y, anomaly_patterns, anomaly_count, feature_names
        )

