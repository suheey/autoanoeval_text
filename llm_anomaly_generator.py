import numpy as np
import pandas as pd
import openai
import json
import re
from typing import List, Dict, Any

class LLMAnomalyGenerator:
    """LLM 기반 합성 이상치 생성기 (Enhanced)"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        print(f"🤖 LLM 이상치 생성기 초기화: {model}")
    
    def get_feature_mapping(self, dataset_name: str) -> Dict[str, str]:
        """데이터셋별 실제 feature 이름 매핑"""
        
        if "cardiotocography" in dataset_name.lower() or "cardio" in dataset_name.lower():
            return {
                'Feature_0': 'LB (Baseline Fetal Heart Rate)',
                'Feature_1': 'AC (Accelerations)',
                'Feature_2': 'FM (Fetal Movements)',
                'Feature_3': 'UC (Uterine Contractions)',
                'Feature_4': 'DL (Light Decelerations)',
                'Feature_5': 'DS (Severe Decelerations)',
                'Feature_6': 'DP (Prolonged Decelerations)',
                'Feature_7': 'ASTV (Abnormal Short Term Variability)',
                'Feature_8': 'MSTV (Mean Short Term Variability)',
                'Feature_9': 'ALTV (Abnormal Long Term Variability)',
                'Feature_10': 'MLTV (Mean Long Term Variability)',
                'Feature_11': 'Width (Histogram Width)',
                'Feature_12': 'Min (Histogram Min)',
                'Feature_13': 'Max (Histogram Max)',
                'Feature_14': 'Nmax (Number of Histogram Peaks)',
                'Feature_15': 'Nzeros (Number of Histogram Zeros)',
                'Feature_16': 'Mode (Histogram Mode)',
                'Feature_17': 'Mean (Histogram Mean)',
                'Feature_18': 'Median (Histogram Median)',
                'Feature_19': 'Variance (Histogram Variance)',
                'Feature_20': 'Tendency (Tendency)',
                'Feature_21': 'CLASS (Fetal State Class)'
            }
        else:
            # 기본 매핑
            return {f'Feature_{i}': f'Feature_{i}' for i in range(22)}
    
    def analyze_anomaly_patterns(self, X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str] = None, 
                                dataset_name: str = "Unknown") -> Dict[str, Any]:
        """LLM을 사용하여 이상치 패턴 분석 (Enhanced)"""
        
        print(f"🤖 LLM 이상치 패턴 분석 시작: {dataset_name}")
        
        # 데이터셋 기본 정보
        n_samples, n_features = X.shape
        n_normal = np.sum(y == 0)
        n_anomaly = np.sum(y == 1)
        
        # 정상 데이터 통계
        X_normal = X[y == 0]
        X_anomaly = X[y == 1] if n_anomaly > 0 else None
        
        # Feature 매핑 가져오기
        feature_mapping = self.get_feature_mapping(dataset_name)
        
        # 실제 feature 이름 설정
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        # 실제 의미 있는 feature 이름들
        meaningful_features = []
        for i, fname in enumerate(feature_names):
            real_name = feature_mapping.get(fname, fname)
            meaningful_features.append(f"{fname} ({real_name})")
        
        # 정상 샘플 통계
        normal_stats = []
        for i, fname in enumerate(feature_names):
            real_name = feature_mapping.get(fname, fname)
            stats = {
                'feature': f"{fname} ({real_name})",
                'mean': round(np.mean(X_normal[:, i]), 3),
                'std': round(np.std(X_normal[:, i]), 3),
                'min': round(np.min(X_normal[:, i]), 3),
                'max': round(np.max(X_normal[:, i]), 3),
                'median': round(np.median(X_normal[:, i]), 3)
            }
            normal_stats.append(stats)
        
        # 이상 샘플 통계 (있는 경우)
        anomaly_stats = []
        if X_anomaly is not None and len(X_anomaly) > 0:
            for i, fname in enumerate(feature_names):
                real_name = feature_mapping.get(fname, fname)
                stats = {
                    'feature': f"{fname} ({real_name})",
                    'mean': round(np.mean(X_anomaly[:, i]), 3),
                    'std': round(np.std(X_anomaly[:, i]), 3),
                    'min': round(np.min(X_anomaly[:, i]), 3),
                    'max': round(np.max(X_anomaly[:, i]), 3),
                    'median': round(np.median(X_anomaly[:, i]), 3)
                }
                anomaly_stats.append(stats)
        
        # 상관관계 분석 (정상 데이터)
        correlation_insights = []
        for i in range(min(5, n_features)):
            for j in range(i+1, min(5, n_features)):
                corr = np.corrcoef(X_normal[:, i], X_normal[:, j])[0, 1]
                if abs(corr) > 0.3:  # 유의미한 상관관계만
                    feat_i = feature_mapping.get(f"Feature_{i}", f"Feature_{i}")
                    feat_j = feature_mapping.get(f"Feature_{j}", f"Feature_{j}")
                    correlation_insights.append({
                        'features': f"{feat_i} & {feat_j}",
                        'correlation': round(corr, 3),
                        'type': 'positive' if corr > 0 else 'negative'
                    })
        
        prompt = f"""You are analyzing a Cardiotocography (CTG) dataset for fetal health monitoring. This dataset contains various measurements from cardiotocographic examinations used to assess fetal well-being during pregnancy.

DATASET INFORMATION:
• Dataset: {dataset_name} (Cardiotocography/Fetal Health Monitoring)
• Total samples: {n_samples:,} (Normal: {n_normal:,}, Abnormal: {n_anomaly:,})
• Features: {n_features} (fetal heart rate, uterine contractions, movements, etc.)

DOMAIN CONTEXT:
Cardiotocography monitors:
- Fetal Heart Rate (FHR) patterns
- Uterine contractions
- Fetal movements
- Various statistical measures of heart rate variability

FEATURE MEANINGS:
{chr(10).join([f"• {feat}" for feat in meaningful_features[:10]])}
{chr(10).join([f"• {feat}" for feat in meaningful_features[10:]]) if len(meaningful_features) > 10 else ""}

NORMAL DATA STATISTICS (first 10 features):
{json.dumps(normal_stats[:10], indent=2)}

{"ABNORMAL DATA STATISTICS (first 10 features):" if anomaly_stats else ""}
{json.dumps(anomaly_stats[:10], indent=2) if anomaly_stats else ""}

CORRELATION INSIGHTS:
{chr(10).join([f"• {insight['features']}: {insight['correlation']} ({insight['type']} correlation)" for insight in correlation_insights[:5]])}

TASK: Identify diverse anomaly patterns for fetal health monitoring. Consider medical scenarios where CTG readings become abnormal.

Generate 6-8 different anomaly conditions covering:
1. Abnormal fetal heart rate patterns (bradycardia, tachycardia)
2. Poor heart rate variability (reduced short/long term variability)
3. Concerning deceleration patterns
4. Abnormal baseline characteristics
5. Inconsistent statistical measurements
6. Combinations of multiple abnormal parameters

Provide response in JSON format:
{{
    "normal_relationships": "Description of typical relationships in healthy CTG readings",
    "anomaly_conditions": [
        {{
            "condition": "mathematical condition using actual feature names",
            "explanation": "medical reason why this indicates fetal distress",
            "severity": "mild/moderate/severe",
            "clinical_scenario": "real clinical situation"
        }}
    ]
}}

Make sure each condition uses specific feature names and realistic threshold values based on the statistics provided."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical expert specializing in fetal monitoring and cardiotocography interpretation. Provide clinically relevant anomaly patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            # JSON 추출 개선
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    print(f"✅ LLM 패턴 분석 완료")
                    self._print_analysis_result(result)
                    return result
                except json.JSONDecodeError as e:
                    print(f"❌ JSON 파싱 오류: {e}")
                    print(f"응답 내용: {content}")
                    return {"error": f"JSON parsing error: {e}"}
            else:
                print(f"❌ JSON 형식 응답 없음")
                print(f"응답 내용: {content}")
                return {"error": "No JSON response"}
                
        except Exception as e:
            print(f"❌ LLM 패턴 분석 실패: {e}")
            return {"error": str(e)}
    
    def generate_anomalies_from_patterns(self, X: np.ndarray, y: np.ndarray,
                                       anomaly_patterns: Dict[str, Any],
                                       anomaly_count: int = None,
                                       feature_names: List[str] = None) -> np.ndarray:
        """패턴 분석 결과를 바탕으로 이상치 생성 (Enhanced)"""
        
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
                'max': float(np.max(X_normal[:, i])),
                'q25': float(np.percentile(X_normal[:, i], 25)),
                'q75': float(np.percentile(X_normal[:, i], 75))
            }
        
        # 이상치 조건들 정리
        conditions_text = ""
        for i, condition in enumerate(anomaly_patterns.get('anomaly_conditions', [])):
            conditions_text += f"{i+1}. {condition.get('condition', '')} → {condition.get('explanation', '')}\n"
            conditions_text += f"   Severity: {condition.get('severity', 'unknown')}\n"
            conditions_text += f"   Scenario: {condition.get('clinical_scenario', '')}\n\n"

        prompt = f"""Generate {min(anomaly_count, 200)} synthetic anomaly data points for Cardiotocography dataset.

NORMAL RELATIONSHIPS:
{anomaly_patterns.get('normal_relationships', 'Not provided')}

ANOMALY CONDITIONS TO IMPLEMENT:
{conditions_text}

FEATURE STATISTICS (Normal Data):
{json.dumps(feature_stats, indent=2)}

GENERATION REQUIREMENTS:
1. Create data points that satisfy the anomaly conditions above
2. Distribute anomalies across different condition types
3. Ensure values stay within realistic ranges for CTG data
4. Add some random variation to make data more realistic
5. Some data points can combine multiple conditions for severe cases

Generate exactly {min(anomaly_count, 200)} anomaly data points.

IMPORTANT: Respond with ONLY a valid JSON object in this exact format:
{{
    "anomaly_data": [
        {{{', '.join([f'"{name}": 0.0' for name in feature_names[:3]])}}},
        {{{', '.join([f'"{name}": 0.0' for name in feature_names[:3]])}}}
    ]
}}

Do not include any text before or after the JSON object."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data generator for medical anomaly detection. Generate ONLY valid JSON without any additional text or formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content.strip()
            
            # JSON 추출 및 정리
            content = content.replace('```json', '').replace('```', '').strip()
            
            try:
                result = json.loads(content)
                anomaly_data_dicts = result.get('anomaly_data', [])
                
                if not anomaly_data_dicts:
                    print(f"❌ 생성된 이상치 데이터가 없습니다")
                    return np.array([])
                
                # 딕셔너리를 numpy 배열로 변환
                anomaly_data = []
                for data_dict in anomaly_data_dicts:
                    row = []
                    for name in feature_names:
                        value = data_dict.get(name, 0.0)
                        # 숫자가 아닌 경우 0으로 대체
                        try:
                            row.append(float(value))
                        except (ValueError, TypeError):
                            row.append(0.0)
                    anomaly_data.append(row)
                
                anomaly_data = np.array(anomaly_data)
                
                if len(anomaly_data) > 0 and anomaly_data.shape[1] == len(feature_names):
                    # 필요한 개수만큼 조정
                    if len(anomaly_data) > anomaly_count:
                        indices = np.random.choice(len(anomaly_data), anomaly_count, replace=False)
                        anomaly_data = anomaly_data[indices]
                    elif len(anomaly_data) < anomaly_count:
                        # 부족한 경우 복제 및 노이즈 추가
                        needed = anomaly_count - len(anomaly_data)
                        indices = np.random.choice(len(anomaly_data), needed, replace=True)
                        additional = anomaly_data[indices]
                        
                        # 약간의 노이즈 추가 (5% 변동)
                        noise = np.random.normal(0, 0.05, additional.shape)
                        additional = additional * (1 + noise)
                        
                        anomaly_data = np.vstack([anomaly_data, additional])
                    
                    print(f"✅ 패턴 기반 이상치 생성 완료: {len(anomaly_data):,}개")
                    print(f"📊 이상치 차원: {anomaly_data.shape}")
                    return anomaly_data
                else:
                    print(f"❌ 생성된 데이터 차원 오류: {anomaly_data.shape if len(anomaly_data) > 0 else 'empty'}")
                    return np.array([])
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 오류: {e}")
                print(f"응답 내용 (첫 500자): {content[:500]}")
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
            print(f"❌ 패턴 분석 실패: {anomaly_patterns['error']}")
            return np.array([])
        
        # 2단계: 패턴 기반 이상치 생성
        anomalies = self.generate_anomalies_from_patterns(
            X, y, anomaly_patterns, anomaly_count, feature_names
        )
        
        return anomalies
    
    def _print_analysis_result(self, result: Dict[str, Any]):
        """분석 결과 출력 (Enhanced)"""
        print("\n📋 이상치 패턴 분석 결과:")
        
        if "normal_relationships" in result:
            print(f"\n1️⃣ 정상 CTG 패턴:")
            print(f"   {result['normal_relationships']}")
        
        if "anomaly_conditions" in result:
            print(f"\n📌 이상치 조건들 ({len(result['anomaly_conditions'])}개):")
            for i, condition in enumerate(result['anomaly_conditions']):
                severity = condition.get('severity', 'unknown')
                emoji = "🔴" if severity == "severe" else "🟡" if severity == "moderate" else "🟢"
                
                print(f"   {i+1}. {emoji} {condition.get('condition', '')}")
                print(f"      → 🔍 {condition.get('explanation', '')}")
                if condition.get('clinical_scenario'):
                    print(f"      → 🏥 {condition['clinical_scenario']}")
                if condition.get('severity'):
                    print(f"      → ⚡ 심각도: {condition['severity']}")
                print()
        
        print()  # 빈 줄 추가