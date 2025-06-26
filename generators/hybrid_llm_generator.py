import numpy as np
import pandas as pd
import openai
import json
import re
import os
from typing import List, Dict, Any
from .base_generator import BaseAnomalyGenerator

class HybridLLMAnomalyGenerator(BaseAnomalyGenerator):
    """하이브리드 LLM 기반 합성 이상치 생성기 (분석: 수동, 생성: 자동)"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", seed: int = 42, num_anomaly_conditions: int = 5):
        super().__init__(seed)
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.num_anomaly_conditions = num_anomaly_conditions
        print(f"🔄 하이브리드 LLM 이상치 생성기 초기화: {model}")
        print(f"   📊 이상치 조건 개수: {num_anomaly_conditions}개")
        print(f"   💡 모드: 분석(수동) + 생성(자동)")
    
    def create_analysis_prompt(self, X: np.ndarray, y: np.ndarray, 
                              feature_names: List[str] = None, 
                              dataset_name: str = "Unknown",
                              num_conditions: int = None,
                              save_path: str = "./prompts") -> str:
        """1단계: 패턴 분석용 프롬프트 생성 및 저장 (수동)"""
        
        # 조건 개수 설정
        if num_conditions is None:
            num_conditions = self.num_anomaly_conditions
        
        print(f"📝 1단계: 패턴 분석 프롬프트 생성 중... (수동)")
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
        
        # 정상 샘플 예시 생성 (10개)
        normal_samples_text = []
        num_samples = min(10, len(X_normal))
        
        for i in range(num_samples):
            sample_parts = []
            for j, feature_name in enumerate(feature_names):
                value = round(X_normal[i, j], 3)
                if value == int(value):
                    value = int(value)
                sample_parts.append(f"{feature_name} is {value}")
            
            sample_text = " , ".join(sample_parts)
            normal_samples_text.append(sample_text)
        
        # 정상 샘플 섹션 생성
        normal_samples_section = ""
        for i, sample_text in enumerate(normal_samples_text, 1):
            normal_samples_section += f"Normal Sample {i}: {sample_text}\n\n"

        # 프롬프트 생성
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
        
        # 프롬프트 저장
        os.makedirs(save_path, exist_ok=True)
        prompt_file = os.path.join(save_path, f"{dataset_name}_analysis_prompt.txt")
        
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        print(f"✅ 패턴 분석 프롬프트 저장: {prompt_file}")
        print(f"📋 다음 단계:")
        print(f"   1. 위 파일의 프롬프트를 웹 LLM에 입력")
        print(f"   2. 결과를 '{save_path}/{dataset_name}_analysis_result.json'에 저장")
        print(f"   3. continue_with_auto_generation() 호출하여 자동 진행")
        
        return prompt_file
    
    def load_analysis_result(self, dataset_name: str, json_path: str = None,
                            save_path: str = "./prompts") -> Dict[str, Any]:
        """분석 결과 로드 및 검증"""
        
        if json_path is None:
            json_path = os.path.join('/lab-di/nfsdata/home/suhee.yoon/autoanoeval/ADBench/autoanoeval_text/prompts', f"{dataset_name}_analysis_result.json")
        
        if not os.path.exists(json_path):
            print(f"❌ JSON 파일이 없습니다: {json_path}")
            print(f"   웹 LLM 결과를 위 경로에 저장해주세요.")
            return {"error": "JSON file not found"}
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            # 결과 검증
            if "anomaly_conditions" not in result:
                print(f"❌ JSON 형식 오류: 'anomaly_conditions' 키가 없습니다.")
                return {"error": "Invalid JSON format"}
            
            actual_conditions = len(result.get('anomaly_conditions', []))
            print(f"✅ 패턴 분석 결과 로드 완료 (조건 {actual_conditions}개)")
            
            if actual_conditions != self.num_anomaly_conditions:
                print(f"⚠️ 요청한 조건 개수({self.num_anomaly_conditions})와 생성된 개수({actual_conditions})가 다릅니다.")
            
            self._print_analysis_result(result)
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 오류: {e}")
            return {"error": f"JSON parsing error: {e}"}
        except Exception as e:
            print(f"❌ 파일 로드 오류: {e}")
            return {"error": f"File load error: {e}"}
    
    def generate_anomalies_from_patterns_auto(self, X: np.ndarray, y: np.ndarray,
                                             anomaly_patterns: Dict[str, Any],
                                             anomaly_count: int = None,
                                             feature_names: List[str] = None) -> np.ndarray:
        """2단계: 패턴 기반 자동 이상치 생성 (API 사용)"""
        
        if anomaly_count is None:
            anomaly_count = np.sum(y == 1)
        
        print(f"🤖 2단계: API 기반 자동 이상치 생성 중... ({anomaly_count:,}개)")
        
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
        
        prompt = f"""**Respond with valid JSON only — no prose, no bullet points.**  
        Generate {anomaly_count} synthetic anomaly data points based on the following anomaly patterns:

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
            print(f"🔄 API 호출 중...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in synthetic anomaly generation. Create realistic anomaly data based on identified patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            print(f"🔍 API 응답 미리보기 (첫 500자):")
            print(content[:500])
            print("...")
            
            # JSON 추출 시도 - 여러 패턴으로 시도
            json_patterns = [
                r'\{.*?\}(?=\s*$)',  # 마지막 JSON 객체
                r'\{.*\}',           # 첫 번째 JSON 객체
                r'"anomaly_data"\s*:\s*\[.*?\]',  # anomaly_data 배열만
            ]
            
            result = None
            for pattern in json_patterns:
                json_match = re.search(pattern, content, re.DOTALL)
                if json_match:
                    try:
                        matched_text = json_match.group()
                        
                        # anomaly_data 배열만 매칭된 경우 완전한 JSON으로 변환
                        if matched_text.startswith('"anomaly_data"'):
                            matched_text = "{" + matched_text + "}"
                        
                        result = json.loads(matched_text)
                        print(f"✅ JSON 파싱 성공 (패턴: {pattern[:20]}...)")
                        break
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON 파싱 실패 (패턴: {pattern[:20]}...): {e}")
                        continue
            
            if result is None:
                print(f"❌ 모든 JSON 파싱 패턴 실패")
                print(f"🔍 전체 응답:")
                print(content)
                return np.array([])
            
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
                
                print(f"✅ API 기반 이상치 생성 완료: {len(anomaly_data):,}개")
                return anomaly_data
            else:
                print(f"❌ 생성된 데이터 없음")
                return np.array([])

                
        except Exception as e:
            print(f"❌ API 기반 이상치 생성 실패: {e}")
            return np.array([])
    
    def generate_anomalies(self, X: np.ndarray, y: np.ndarray, 
                          anomaly_type: str = "pattern_based",
                          anomaly_count: int = None,
                          feature_names: List[str] = None,
                          dataset_name: str = "Unknown",
                          num_conditions: int = None,
                          save_path: str = "./prompts",
                          hybrid_step: str = "start") -> np.ndarray:
        """하이브리드 이상치 생성 함수"""
        
        print(f"🔄 하이브리드 LLM 기반 이상치 생성 시작")
        
        if anomaly_count is None:
            anomaly_count = np.sum(y == 1)
        
        if hybrid_step == "start":
            # 1단계: 분석 프롬프트 생성 (수동)
            analysis_prompt_file = self.create_analysis_prompt(
                X, y, feature_names, dataset_name, num_conditions, save_path
            )
            
            print(f"\n⏸️ 수동 개입 필요:")
            print(f"   1. {analysis_prompt_file} 내용을 웹 LLM에 입력")
            print(f"   2. 결과를 {save_path}/{dataset_name}_analysis_result.json에 저장")
            print(f"   3. hybrid_step='continue'로 다시 실행")
            
            return np.array([])  # 수동 개입 대기
            
        elif hybrid_step == "continue":
            # 1단계 결과 로드
            anomaly_patterns = self.load_analysis_result(dataset_name, save_path=save_path)
            
            if "error" in anomaly_patterns:
                print(f"❌ 분석 결과 로드 실패")
                return np.array([])
            
            # 2단계: 자동 이상치 생성 (API)
            anomalies = self.generate_anomalies_from_patterns_auto(
                X, y, anomaly_patterns, anomaly_count, feature_names
            )
            
            return anomalies
            
        else:
            print(f"❌ 잘못된 hybrid_step 값: {hybrid_step}")
            return np.array([])
    
    def continue_with_auto_generation(self, X: np.ndarray, y: np.ndarray,
                                     feature_names: List[str] = None,
                                     dataset_name: str = "Unknown",
                                     anomaly_count: int = None,
                                     save_path: str = "./prompts") -> np.ndarray:
        """분석 완료 후 자동 생성 진행"""
        
        print(f"🚀 분석 완료 후 자동 생성 시작...")
        
        return self.generate_anomalies(
            X, y, anomaly_count=anomaly_count, feature_names=feature_names,
            dataset_name=dataset_name, save_path=save_path, hybrid_step="continue"
        )
    
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
        