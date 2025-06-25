import numpy as np
import pandas as pd
import openai
import json
import re
from typing import List, Dict, Any
from .base_generator import BaseAnomalyGenerator

class LLMAnomalyGenerator(BaseAnomalyGenerator):
    """LLM 기반 합성 이상치 생성기"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", seed: int = 42):
        super().__init__(seed)
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        print(f"🤖 LLM 이상치 생성기 초기화: {model}")
    
    def analyze_anomaly_patterns(self, X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str] = None, 
                                dataset_name: str = "Unknown") -> Dict[str, Any]:
        """LLM을 사용하여 이상치 패턴 분석"""
        
        print(f"🤖 LLM 이상치 패턴 분석 시작: {dataset_name}")
        
        # 데이터셋 기본 정보
        n_samples, n_features = X.shape
        n_normal = np.sum(y == 0)
        n_anomaly = np.sum(y == 1)
        
        # 정상 데이터 통계
        X_normal = X[y == 0]
        
        # 특성 이름 설정
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        # 정상 샘플 예시 생성 (3개)
        normal_samples = []
        for i in range(min(3, len(X_normal))):
            sample = {}
            for j, feature_name in enumerate(feature_names):
                sample[feature_name] = round(X_normal[i, j], 3)
            normal_samples.append(sample)
        
        prompt = f"""Your objective is to predict what combinations of values may indicate a plausible fault or anomaly scenario.

Consider the following dataset description:
• Dataset: {dataset_name}
• Total samples: {n_samples:,} (Normal: {n_normal:,}, Anomaly: {n_anomaly:,})
• Features: {n_features}

Consider the following features:
{', '.join(feature_names)}

Refer the normal sample examples:
{json.dumps(normal_samples, indent=2)}

Explain step-by-step the realistic anomaly pattern:

1️⃣ 일반적인 관계 파악
Normally, [describe typical relationships between features]

2️⃣ 비정상 조건 도출 ①
If [condition], this might be due to [reason].

3️⃣ 비정상 조건 도출 ②
Similarly, [another condition] may suggest [another reason].

Then provide mathematical conditions for anomalies:

📌 이상치 조건 예시:
• [Feature] > [threshold] AND [Feature] < [threshold] → 🔍 [explanation]
• [Feature] > [threshold] AND [Feature] < [threshold] → 🔍 [explanation]

Provide your response in JSON format:
{{
    "normal_relationships": "Description of typical feature relationships",
    "anomaly_conditions": [
        {{
            "condition": "mathematical condition (e.g., Temperature > 90 AND Vibration < 0.2)",
            "explanation": "reason why this is anomalous",
            "scenario": "real-world scenario"
        }}
    ]
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in anomaly detection and domain analysis. Provide step-by-step reasoning for anomaly patterns in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group())
                print(f"✅ LLM 패턴 분석 완료")
                self._print_analysis_result(result)
                return result
            else:
                print(f"❌ JSON 형식 응답 없음")
                return {"error": "No JSON response"}
                
        except Exception as e:
            print(f"❌ LLM 패턴 분석 실패: {e}")
            return {"error": str(e)}
    
    def generate_anomalies_from_patterns(self, X: np.ndarray, y: np.ndarray,
                                       anomaly_patterns: Dict[str, Any],
                                       anomaly_count: int = None,
                                       feature_names: List[str] = None) -> np.ndarray:
        """패턴 분석 결과를 바탕으로 이상치 생성"""
        
        if anomaly_count is None:
            anomaly_count = np.sum(y == 1)
        
        print(f"🎯 패턴 기반 이상치 생성: {anomaly_count:,}개")
        
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
        
        prompt = f"""Generate {anomaly_count} synthetic anomaly data points based on the following anomaly patterns:

Normal Relationships:
{anomaly_patterns.get('normal_relationships', 'Not provided')}

Anomaly Conditions:
"""
        
        for i, condition in enumerate(anomaly_patterns.get('anomaly_conditions', [])):
            prompt += f"{i+1}. {condition.get('condition', '')} → {condition.get('explanation', '')}\n"
        
        prompt += f"""

Feature Statistics (Normal Data):
{json.dumps(feature_stats, indent=2)}

Generate realistic anomaly data points that satisfy the above anomaly conditions.
Ensure the generated data:
1. Follows the mathematical conditions identified
2. Maintains realistic value ranges
3. Represents diverse anomaly scenarios

Provide response in JSON format:
{{
    "anomaly_data": [
        {{{', '.join([f'"{name}": value' for name in feature_names])}}},
        {{{', '.join([f'"{name}": value' for name in feature_names])}}}
    ],
    "pattern_usage": "Which patterns were used to generate each anomaly"
}}

Generate exactly {min(anomaly_count, 100)} data points."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in synthetic anomaly generation. Create realistic anomaly data based on identified patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group())
                anomaly_data_dicts = result.get('anomaly_data', [])
                
                # 딕셔너리를 numpy 배열로 변환
                anomaly_data = []
                for data_dict in anomaly_data_dicts:
                    row = [data_dict.get(name, 0) for name in feature_names]
                    anomaly_data.append(row)
                
                anomaly_data = np.array(anomaly_data)
                
                if len(anomaly_data) > 0:
                    # 필요한 개수만큼 조정
                    if len(anomaly_data) > anomaly_count:
                        indices = np.random.choice(len(anomaly_data), anomaly_count, replace=False)
                        anomaly_data = anomaly_data[indices]
                    elif len(anomaly_data) < anomaly_count:
                        # 부족한 경우 복제
                        needed = anomaly_count - len(anomaly_data)
                        indices = np.random.choice(len(anomaly_data), needed, replace=True)
                        additional = anomaly_data[indices]
                        anomaly_data = np.vstack([anomaly_data, additional])
                    
                    print(f"✅ 패턴 기반 이상치 생성 완료: {len(anomaly_data):,}개")
                    return anomaly_data
                else:
                    print(f"❌ 생성된 데이터 없음")
                    return np.array([])
            else:
                print(f"❌ JSON 형식 응답 없음")
                return np.array([])
                
        except Exception as e:
            print(f"❌ 패턴 기반 이상치 생성 실패: {e}")
            return np.array([])
    
    def generate_anomalies(self, X: np.ndarray, y: np.ndarray, 
                          anomaly_type: str = "pattern_based",
                          anomaly_count: int = None,
                          feature_names: List[str] = None,
                          dataset_name: str = "Unknown") -> np.ndarray:
        """통합 이상치 생성 함수"""
        
        print(f"🚀 LLM 기반 이상치 생성 시작")
        
        # 1단계: 이상치 패턴 분석
        anomaly_patterns = self.analyze_anomaly_patterns(
            X, y, feature_names, dataset_name
        )
        
        if "error" in anomaly_patterns:
            print(f"❌ 패턴 분석 실패")
            return np.array([])
        
        # 2단계: 패턴 기반 이상치 생성
        anomalies = self.generate_anomalies_from_patterns(
            X, y, anomaly_patterns, anomaly_count, feature_names
        )
        
        return anomalies
    
    def _print_analysis_result(self, result: Dict[str, Any]):
        """분석 결과 출력"""
        print("\n📋 이상치 패턴 분석 결과:")
        
        if "normal_relationships" in result:
            print(f"\n1️⃣ 일반적인 관계:")
            print(f"   {result['normal_relationships']}")
        
        if "anomaly_conditions" in result:
            print(f"\n📌 이상치 조건:")
            for i, condition in enumerate(result['anomaly_conditions']):
                print(f"   {i+1}. {condition.get('condition', '')} → 🔍 {condition.get('explanation', '')}")
                if condition.get('scenario'):
                    print(f"      시나리오: {condition['scenario']}")
        
        print()  # 빈 줄 추가