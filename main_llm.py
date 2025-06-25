import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from data_generator import SimpleDataGenerator
from llm_anomaly_generator import LLMAnomalyGenerator
from model_selection_enhanced import run_model_selection_experiment
import argparse
import sys
import time

# 설정
RANDOM_SEED = 42
ANOMALY_TYPES = ['local', 'cluster', 'global', 'discrepancy']

class Tee(object):
    """터미널과 파일 동시 출력을 위한 클래스"""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def load_dataset(dataset_name):
    """CSV 데이터셋 로드 및 전처리"""
    csv_path = f'/lab-di/nfsdata/home/suhee.yoon/autoanoeval/data/adbench_column/{dataset_name}.csv'    
    print(f"📥 CSV 데이터셋 로드: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ 파일이 존재하지 않습니다: {csv_path}")
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"📊 로드된 데이터 형태: {df.shape}")

    y = df['label'].values
    df = df.drop(columns=['label'])

    # 숫자 / 문자열 feature 자동 구분
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    string_cols = df.select_dtypes(include=['object', 'string']).columns
    
    print(f"📈 숫자 컬럼: {len(numeric_cols)}개, 문자열 컬럼: {len(string_cols)}개")

    # 숫자 feature normalize
    scaler = MinMaxScaler()
    X_numeric = scaler.fit_transform(df[numeric_cols].values)

    # 문자열 feature → integer encoding
    X_strings = []
    for col in string_cols:
        unique_vals = np.unique(df[col])
        val_to_int = {val: idx for idx, val in enumerate(unique_vals)}
        encoded_col = np.vectorize(val_to_int.get)(df[col].values).reshape(-1, 1)
        X_strings.append(encoded_col)
        print(f"   🔤 {col}: {len(unique_vals)}개 고유값 → 정수 인코딩")

    # 숫자 + 문자열 feature 결합
    if X_strings:
        X_strings = np.hstack(X_strings)
        X = np.hstack([X_numeric, X_strings])
        print(f"📊 최종 feature 차원: {X_numeric.shape[1]} (숫자) + {X_strings.shape[1]} (문자열) = {X.shape[1]}")
    else:
        X = X_numeric
        print(f"📊 최종 feature 차원: {X.shape[1]} (숫자만)")

    print(f"🏷️ 레이블 분포: 정상 {np.sum(y == 0):,}개, 이상 {np.sum(y == 1):,}개")
    
    return X, y

def prepare_dataset_splits(X_original, y_original):
    """데이터셋 분할"""
    print(f"📊 원본 데이터: {X_original.shape}")
    print(f"📊 클래스 분포 - 정상: {np.sum(y_original == 0):,}, 이상: {np.sum(y_original == 1):,}")

    # 대용량 데이터 최적화
    max_normal = 3000
    max_anomaly = 500
    
    if np.sum(y_original == 0) > max_normal * 2:
        print(f"⚡ 대용량 정상 데이터 감지. {max_normal:,}개로 제한")
    if np.sum(y_original == 1) > max_anomaly * 2:
        print(f"⚡ 대용량 이상 데이터 감지. {max_anomaly:,}개로 제한")

    # 데이터 제한 및 분리
    X_normal = X_original[y_original == 0][:max_normal]
    X_anomaly = X_original[y_original == 1][:max_anomaly]
    
    # 데이터 분할
    X_normal_train, X_normal_holdout = train_test_split(
        X_normal, test_size=0.4, random_state=RANDOM_SEED
    )
    X_anomaly_val, X_anomaly_test = train_test_split(
        X_anomaly, test_size=0.7, random_state=RANDOM_SEED
    )
    X_normal_val, X_normal_test = train_test_split(
        X_normal_holdout, test_size=0.5, random_state=RANDOM_SEED
    )
    
    # 최종 데이터셋 구성
    X_val_real = np.vstack([X_normal_val, X_anomaly_val])
    y_val_real = np.concatenate([np.zeros(len(X_normal_val)), np.ones(len(X_anomaly_val))])
    
    X_test = np.vstack([X_normal_test, X_anomaly_test])
    y_test = np.concatenate([np.zeros(len(X_normal_test)), np.ones(len(X_anomaly_test))])
    
    # 데이터 셔플
    idx = np.random.RandomState(RANDOM_SEED).permutation(len(y_val_real))
    X_val_real, y_val_real = X_val_real[idx], y_val_real[idx]
    
    idx = np.random.RandomState(RANDOM_SEED).permutation(len(y_test))
    X_test, y_test = X_test[idx], y_test[idx]
    
    print(f"\n📋 데이터셋 분할 완료:")
    print(f"   Train (정상만): {X_normal_train.shape}")
    print(f"   Real Validation: {X_val_real.shape} (정상: {np.sum(y_val_real == 0):,}, 이상: {np.sum(y_val_real == 1):,})")
    print(f"   Test: {X_test.shape} (정상: {np.sum(y_test == 0):,}, 이상: {np.sum(y_test == 1):,})")
    
    return X_normal_train, X_normal_val, X_val_real, y_val_real, X_test, y_test, X_anomaly_val

def generate_llm_validation_sets(X_original, y_original, X_normal_val, X_anomaly_val, 
                                feature_names=None, dataset_name="Unknown", 
                                openai_api_key=None):
    """LLM 기반 합성 이상치 검증 세트 생성"""
    print(f"\n🧪 LLM 기반 이상치 패턴 분석 및 생성...")
    
    if not openai_api_key:
        print("❌ OpenAI API 키가 없습니다. 통계적 방법을 사용하세요.")
        return {}
    
    # LLM 생성기 초기화
    llm_generator = LLMAnomalyGenerator(api_key=openai_api_key)
    
    try:
        # LLM 기반 이상치 생성
        synthetic_anomalies = llm_generator.generate_anomalies(
            X=X_original,
            y=y_original,
            anomaly_count=len(X_anomaly_val),
            feature_names=feature_names,
            dataset_name=dataset_name
        )
        
        if len(synthetic_anomalies) > 0:
            # 검증 세트 구성
            X_val_synthetic = np.vstack([X_normal_val, synthetic_anomalies])
            y_val_synthetic = np.concatenate([
                np.zeros(len(X_normal_val)), 
                np.ones(len(synthetic_anomalies))
            ])
            
            # 데이터 셔플
            idx = np.random.RandomState(RANDOM_SEED).permutation(len(y_val_synthetic))
            X_val_synthetic, y_val_synthetic = X_val_synthetic[idx], y_val_synthetic[idx]
            
            print(f"✅ LLM 기반 검증 세트 생성 완료: {X_val_synthetic.shape}")
            print(f"   정상: {np.sum(y_val_synthetic == 0):,}, 이상: {np.sum(y_val_synthetic == 1):,}")
            
            return {"llm_patterns": (X_val_synthetic, y_val_synthetic)}
        else:
            print("❌ LLM 이상치 생성 실패")
            return {}
            
    except Exception as e:
        print(f"❌ LLM 생성 중 오류: {e}")
        return {}

def generate_synthetic_validation_sets(X_original, y_original, X_normal_val, X_anomaly_val):
    """기존 통계적 합성 이상치 검증 세트 생성 (폴백용)"""
    print(f"\n🧪 통계적 Synthetic Anomaly 검증 세트 생성...")
    
    # 데이터 생성기 초기화
    data_generator = SimpleDataGenerator(seed=RANDOM_SEED)
    synthetic_val_sets = {}
    
    # 각 유형별 합성 이상치 생성
    for anomaly_type in ANOMALY_TYPES:
        print(f"   🔬 {anomaly_type} 유형 생성 중...")
        
        try:
            synthetic_anomalies = data_generator.generate_anomalies(
                X=X_original,
                y=y_original,
                anomaly_type=anomaly_type,
                alpha=5,
                percentage=0.2,
                anomaly_count=len(X_anomaly_val)
            )
            
            # 검증 세트 구성
            X_val_synthetic = np.vstack([X_normal_val, synthetic_anomalies])
            y_val_synthetic = np.concatenate([np.zeros(len(X_normal_val)), np.ones(len(synthetic_anomalies))])
            
            # 데이터 셔플
            idx = np.random.RandomState(RANDOM_SEED).permutation(len(y_val_synthetic))
            X_val_synthetic, y_val_synthetic = X_val_synthetic[idx], y_val_synthetic[idx]
            
            synthetic_val_sets[anomaly_type] = (X_val_synthetic, y_val_synthetic)
            
            print(f"      ✅ {anomaly_type}: {X_val_synthetic.shape} "
                  f"(정상: {np.sum(y_val_synthetic == 0):,}, 이상: {np.sum(y_val_synthetic == 1):,})")
            
        except Exception as e:
            print(f"      ❌ {anomaly_type} 생성 실패: {e}")
    
    return synthetic_val_sets

def main(args):
    """메인 실험 실행 함수"""
    print("🔬 LLM 기반 이상치 패턴 분석 실험 시작")
    print("=" * 80)
    
    # 결과 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./result_metric/{args.dataset_name}_experiment_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # 로그 파일 설정
    log_file = open(os.path.join(results_dir, "experiment_log.txt"), "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    
    try:
        # 1. 데이터셋 준비 (CSV 로드)
        print(f"\n📊 데이터셋 로드: {args.dataset_name}")
        X_original, y_original = load_dataset(args.dataset_name)
        
        # 2. 데이터셋 분할
        experiment_start_time = time.time()
        
        X_normal_train, X_normal_val, X_val_real, y_val_real, X_test, y_test, X_anomaly_val = prepare_dataset_splits(
            X_original, y_original
        )
        
        # 특성 이름 생성 (실제 데이터에 맞게 수정 필요)
        feature_names = [f"Feature_{i}" for i in range(X_original.shape[1])]
        
        # 3. LLM 기반 이상치 패턴 분석 및 생성
        llm_val_sets = generate_llm_validation_sets(
            X_original=X_original, 
            y_original=y_original, 
            X_normal_val=X_normal_val, 
            X_anomaly_val=X_anomaly_val,
            feature_names=feature_names,
            dataset_name=args.dataset_name,
            openai_api_key=getattr(args, 'openai_api_key', None)
        )
        
        # 4. 기존 통계적 방법과 병행 사용도 가능
        if not llm_val_sets and hasattr(args, 'use_statistical_fallback') and args.use_statistical_fallback:
            print("\n🔄 통계적 방법으로 폴백...")
            synthetic_val_sets = generate_synthetic_validation_sets(
                X_original, y_original, X_normal_val, X_anomaly_val
            )
        else:
            synthetic_val_sets = llm_val_sets
        
        print(f"\n📋 생성된 검증 세트: {list(synthetic_val_sets.keys())}")
        
        # 5. 모델 선택 실험 실행 (모듈이 있는 경우)
        try:
            print(f"\n🚀 모델 선택 실험 실행...")
            model_start_time = time.time()
            
            all_results, best_models, evaluation_metrics = run_model_selection_experiment(
                X_normal_train=X_normal_train,
                X_val_real=X_val_real,
                y_val_real=y_val_real,
                synthetic_val_sets=synthetic_val_sets,
                X_test=X_test,
                y_test=y_test,
                results_dir=results_dir
            )
            
            model_time = time.time() - model_start_time
            total_time = time.time() - experiment_start_time
            
            # 성능 요약
            print(f"\n⏱️ 실험 시간 요약:")
            print(f"   데이터 준비: {model_start_time - experiment_start_time:.2f}s")
            print(f"   모델 학습/평가: {model_time:.2f}s")
            print(f"   전체 실험 시간: {total_time:.2f}s")
            
        except ImportError:
            print(f"⚠️ model_selection_enhanced 모듈이 없습니다. 패턴 분석까지만 수행됩니다.")
        except Exception as e:
            print(f"❌ 모델 선택 실험 실패: {e}")
        
        print(f"\n🎉 실험 성공적으로 완료!")
        print(f"📁 결과 위치: {results_dir}")
        
    except Exception as e:
        print(f"\n❌ 실험 중 오류 발생: {e}")
        raise
    finally:
        log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 기반 이상치 패턴 분석 실험")
    parser.add_argument("--dataset_name", type=str, required=True, 
                       help="데이터셋 이름")
    parser.add_argument("--openai_api_key", type=str, required=True,
                       help="OpenAI API 키")
    parser.add_argument("--use_statistical_fallback", action="store_true", default=False,
                       help="LLM 실패 시 통계적 방법 사용")
    
    args = parser.parse_args()
    main(args)