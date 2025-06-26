import numpy as np
import pandas as pd
import openai
import json
import re
from typing import List, Dict, Any
from .base_generator import BaseAnomalyGenerator

class LLMAnomalyGenerator(BaseAnomalyGenerator):
    """LLM 기반 합성 이상치 생성기"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", seed: int = 42, num_anomaly_conditions: int = 5):
        super().__init__(seed)
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.num_anomaly_conditions = num_anomaly_conditions
        print(f"🤖 LLM 이상치 생성기 초기화: {model}")
        print(f"   📊 이상치 조건 개수: {num_anomaly_conditions}개")
    
    def analyze_anomaly_patterns(self, X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str] = None, 
                                dataset_name: str = "Unknown",
                                num_conditions: int = None) -> Dict[str, Any]:
        """LLM을 사용하여 이상치 패턴 분석"""
        
        # 조건 개수 설정 (파라미터 > 인스턴스 설정 > 기본값)
        if num_conditions is None:
            num_conditions = self.num_anomaly_conditions
        
        print(f"🤖 LLM 이상치 패턴 분석 시작: {dataset_name}")
        print(f"   📊 요청된 이상치 조건 개수: {num_conditions}개")
        
        # 데이터셋 기본 정보
        n_samples, n_features = X.shape
        n_normal = np.sum(y == 0)
        n_anomaly = np.sum(y == 1)
        
        # 정상 데이터 통계
        X_normal = X[y == 0]
        
        # 특성 이름 설정
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        # 정상 샘플 예시 생성 (10개) - 텍스트 형태로 변환
        normal_samples_text = []
        num_samples = min(10, len(X_normal))  # 최대 10개까지
        
        for i in range(num_samples):
            sample_parts = []
            for j, feature_name in enumerate(feature_names):
                value = round(X_normal[i, j], 3)
                # 정수값인 경우 .0 제거
                if value == int(value):
                    value = int(value)
                sample_parts.append(f"{feature_name} is {value}")
            
            # 더 읽기 쉽게 포맷팅
            sample_text = " , ".join(sample_parts)
            normal_samples_text.append(sample_text)
        
        print(f"정상 샘플 {num_samples}개 생성됨")
        print("첫 번째 예시:", normal_samples_text[0] if normal_samples_text else "없음")
        
        # 프롬프트에 넣을 정상 샘플 예시들 생성
        normal_samples_section = ""
        for i, sample_text in enumerate(normal_samples_text, 1):
            normal_samples_section += f"Normal Sample {i}: {sample_text}\n\n"

        prompt = f"""**Respond with valid JSON only — no prose, no bullet points.**  
        Your objective is to predict what combinations of values may indicate a plausible fault or anomaly scenario.

Consider the following dataset description:
• Dataset: {dataset_name}
• Total samples: {n_samples:,} (Normal: {n_normal:,}, Anomaly: {n_anomaly:,})
• Features: {n_features}

Consider the following features:
{', '.join(feature_names)}

Here are examples of normal samples from the dataset:

{normal_samples_section}Based on these {len(normal_samples_text)} normal examples, explain step-by-step what would constitute realistic anomaly patterns:

1️⃣ Identify typical feature relationships
Normally, [describe typical relationships between features based on the examples above]

2️⃣ Derive anomaly conditions 1
If [condition], this might be due to [reason].

3️⃣ Derive anomaly conditions 2
Similarly, [another condition] may suggest [another reason].

Then provide mathematical conditions for anomalies:

📌 Anomaly Condition Examples (always combine **at least two** different features):
• [Feature_A] > [threshold_A] AND [Feature_B] < [threshold_B] → 🔍 [explanation]
• [Feature_C] > [threshold_C] AND [Feature_D] / [Feature_E] > [ratio] → 🔍 [explanation]
• [Feature_F] = [value_F] AND [Feature_G] < [threshold_G] AND [Feature_H] > [threshold_H] → 🔍 [explanation]

📌 Provide {num_conditions} anomaly conditions in JSON format, each with a `condition`, `explanation`, and `scenario`:

{{
    "normal_relationships": "Description of typical feature relationships observed in the normal samples",
    "anomaly_conditions": [
        {{
            "condition": "mathematical condition (e.g., LB > 200 AND AC < 1)",
            "explanation": "reason why this combination is anomalous",
            "scenario": "real-world scenario that could cause this anomaly"
        }}
    ]
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in anomaly detection and domain analysis. Provide step-by-step reasoning for anomaly patterns. Output ONLY a JSON object** matching the schema exactly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group())
                
                # 이상치 조건 개수 검증
                actual_conditions = len(result.get('anomaly_conditions', []))
                print(f"✅ LLM 패턴 분석 완료 (조건 {actual_conditions}개/{num_conditions}개 요청)")
                
                if actual_conditions != num_conditions:
                    print(f"⚠️ 요청한 조건 개수({num_conditions})와 생성된 개수({actual_conditions})가 다릅니다.")
                
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
                          dataset_name: str = "Unknown",
                          num_conditions: int = None) -> np.ndarray:
        """통합 이상치 생성 함수"""
        
        print(f"🚀 LLM 기반 이상치 생성 시작")
        
        # 1단계: 이상치 패턴 분석
        anomaly_patterns = self.analyze_anomaly_patterns(
            X, y, feature_names, dataset_name, num_conditions
        )
        
        if "error" in anomaly_patterns:
            print(f"❌ 패턴 분석 실패")
            return np.array([])
        
        # 2단계: 패턴 기반 이상치 생성
        anomalies = self.generate_anomalies_from_patterns(
            X, y, anomaly_patterns, anomaly_count, feature_names
        )
        
        return anomalies
    
    def set_anomaly_conditions_count(self, count: int):
        """이상치 조건 개수 설정"""
        self.num_anomaly_conditions = count
        print(f"📊 이상치 조건 개수가 {count}개로 설정되었습니다.")
    
    def get_anomaly_conditions_count(self) -> int:
        """현재 설정된 이상치 조건 개수 반환"""
        return self.num_anomaly_conditions
    
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