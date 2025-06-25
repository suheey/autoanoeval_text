import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import os
from datetime import datetime
import urllib.request
from sklearn.model_selection import train_test_split
from data_generator import SimpleDataGenerator
from model_selection_enhanced import run_model_selection_experiment
import argparse
import sys
import time

# GPU 가속 확인
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"🚀 GPU 가속 사용 가능!")
        print(f"📊 GPU: {cp.cuda.Device().mem_info}")
        print(f"💾 GPU 메모리: {cp.cuda.Device().mem_info[1] / 1e9:.1f} GB")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ CuPy가 설치되지 않음. CPU 모드로 실행합니다.")
    print("GPU 가속을 원하시면: pip install cupy-cuda12x")

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
    df = df.drop(columns=['label'])  # label 제거

    # ✅ 숫자 / 문자열 feature 자동 구분
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    string_cols = df.select_dtypes(include=['object', 'string']).columns
    
    print(f"📈 숫자 컬럼: {len(numeric_cols)}개, 문자열 컬럼: {len(string_cols)}개")

    # ✅ 숫자 feature normalize
    scaler = MinMaxScaler()
    X_numeric = scaler.fit_transform(df[numeric_cols].values)

    # ✅ 문자열 feature → integer encoding
    X_strings = []
    for col in string_cols:
        unique_vals = np.unique(df[col])
        val_to_int = {val: idx for idx, val in enumerate(unique_vals)}
        encoded_col = np.vectorize(val_to_int.get)(df[col].values).reshape(-1, 1)
        X_strings.append(encoded_col)
        print(f"   🔤 {col}: {len(unique_vals)}개 고유값 → 정수 인코딩")

    # ✅ 숫자 + 문자열 feature 결합
    if X_strings:
        X_strings = np.hstack(X_strings)
        X = np.hstack([X_numeric, X_strings])
        print(f"📊 최종 feature 차원: {X_numeric.shape[1]} (숫자) + {X_strings.shape[1]} (문자열) = {X.shape[1]}")
    else:
        X = X_numeric
        print(f"📊 최종 feature 차원: {X.shape[1]} (숫자만)")

    print(f"🏷️ 레이블 분포: 정상 {np.sum(y == 0):,}개, 이상 {np.sum(y == 1):,}개")
    
    return X, y

def download_dataset(url, filename):
    """데이터셋 다운로드 (CSV 사용 시 사용되지 않음)"""
    print(f"ℹ️ CSV 파일 직접 로드 모드. 다운로드 생략.")
    pass

def prepare_dataset_splits(X_original, y_original):
    """GPU 가속 데이터셋 분할"""
    print(f"📊 원본 데이터: {X_original.shape}")
    print(f"📊 클래스 분포 - 정상: {np.sum(y_original == 0):,}, 이상: {np.sum(y_original == 1):,}")

    # 대용량 데이터 최적화
    max_normal = 3000
    max_anomaly = 500
    
    if np.sum(y_original == 0) > max_normal * 2:  # 충분한 여유가 있는 경우만 제한
        print(f"⚡ 대용량 정상 데이터 감지. {max_normal:,}개로 제한")
    if np.sum(y_original == 1) > max_anomaly * 2:
        print(f"⚡ 대용량 이상 데이터 감지. {max_anomaly:,}개로 제한")

    # 데이터 제한 및 분리
    X_normal = X_original[y_original == 0][:max_normal]
    X_anomaly = X_original[y_original == 1][:max_anomaly]
    
    # GPU 가속 데이터 분할 (가능한 경우)
    if GPU_AVAILABLE and len(X_normal) > 10000:
        print("🚀 GPU 가속 데이터 분할 적용")
        X_normal, X_normal_holdout = gpu_train_test_split(X_normal, test_size=0.4)
        X_anomaly_val, X_anomaly_test = gpu_train_test_split(X_anomaly, test_size=0.7)
        X_normal_val, X_normal_test = gpu_train_test_split(X_normal_holdout, test_size=0.5)
    else:
        # 기존 CPU 방식
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
    
    # GPU 가속 데이터 셔플
    X_val_real, y_val_real = gpu_shuffle_data(X_val_real, y_val_real)
    X_test, y_test = gpu_shuffle_data(X_test, y_test)
    
    print(f"\n📋 데이터셋 분할 완료:")
    print(f"   Train (정상만): {X_normal_train.shape}")
    print(f"   Real Validation: {X_val_real.shape} (정상: {np.sum(y_val_real == 0):,}, 이상: {np.sum(y_val_real == 1):,})")
    print(f"   Test: {X_test.shape} (정상: {np.sum(y_test == 0):,}, 이상: {np.sum(y_test == 1):,})")
    
    return X_normal_train, X_normal_val, X_val_real, y_val_real, X_test, y_test, X_anomaly_val

def gpu_train_test_split(X, test_size=0.3):
    """GPU 가속 train-test split"""
    if GPU_AVAILABLE:
        X_gpu = cp.asarray(X)
        n_samples = len(X_gpu)
        n_test = int(n_samples * test_size)
        
        # GPU에서 랜덤 인덱스 생성
        indices = cp.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        # GPU에서 분할
        X_train_gpu = X_gpu[train_indices]
        X_test_gpu = X_gpu[test_indices]
        
        # CPU로 이동
        return cp.asnumpy(X_train_gpu), cp.asnumpy(X_test_gpu)
    else:
        # CPU 폴백
        return train_test_split(X, test_size=test_size, random_state=RANDOM_SEED)

def gpu_shuffle_data(X, y):
    """GPU 가속 데이터 셔플"""
    if GPU_AVAILABLE and len(X) > 1000:
        # GPU에서 셔플
        idx_gpu = cp.random.RandomState(RANDOM_SEED).permutation(len(y))
        idx = cp.asnumpy(idx_gpu)
    else:
        # CPU 셔플
        idx = np.random.RandomState(RANDOM_SEED).permutation(len(y))
    
    return X[idx], y[idx]

def generate_synthetic_validation_sets(X_original, y_original, X_normal_val, X_anomaly_val, use_gpu=True):
    """GPU 가속 합성 이상치 검증 세트 생성"""
    print(f"\n🧪 Synthetic Anomaly 검증 세트 생성... ({'GPU' if use_gpu and GPU_AVAILABLE else 'CPU'})")
    
    # GPU 사용 여부 결정
    actual_use_gpu = use_gpu and GPU_AVAILABLE
    
    if actual_use_gpu:
        print(f"🚀 GPU 가속 모드 활성화")
        # PyTorch CUDA 메모리 정보는 제한적이므로 간단히 표시
        props = torch.cuda.get_device_properties(0)
        print(f"💾 GPU 총 메모리: {props.total_memory / 1e9:.1f} GB")
        
        # 데이터 크기 기반 간단한 검사
        data_size_gb = X_original.nbytes / 1e9
        if data_size_gb > 2.0:  # 2GB 이상 시 주의
            print(f"⚠️ 데이터 크기({data_size_gb:.1f}GB)가 큼. 메모리 사용량 주의")
    
    # 데이터 생성기 초기화 (PyTorch 버전은 SimpleDataGenerator에 맞춰 수정 필요)
    data_generator = SimpleDataGenerator(seed=RANDOM_SEED, use_gpu=actual_use_gpu)
    synthetic_val_sets = {}
    synthetic_anomalies_by_type = {}
    
    # 각 유형별 시간 측정
    generation_times = {}
    
    # 각 유형별 합성 이상치 생성
    for anomaly_type in ANOMALY_TYPES:
        print(f"   🔬 {anomaly_type} 유형 생성 중...")
        
        # 시간 측정 시작
        start_time = time.time()
        
        try:
            synthetic_anomalies = data_generator.generate_anomalies(
                X=X_original,
                y=y_original,
                anomaly_type=anomaly_type,
                alpha=5,
                percentage=0.2,
                anomaly_count=len(X_anomaly_val)
            )
            
            generation_time = time.time() - start_time
            generation_times[anomaly_type] = generation_time
            
            synthetic_anomalies_by_type[anomaly_type] = synthetic_anomalies
            
            # 검증 세트 구성
            X_val_synthetic = np.vstack([X_normal_val, synthetic_anomalies])
            y_val_synthetic = np.concatenate([np.zeros(len(X_normal_val)), np.ones(len(synthetic_anomalies))])
            
            # GPU 가속 셔플
            X_val_synthetic, y_val_synthetic = gpu_shuffle_data(X_val_synthetic, y_val_synthetic)
            
            synthetic_val_sets[anomaly_type] = (X_val_synthetic, y_val_synthetic)
            
            print(f"      ✅ {anomaly_type}: {X_val_synthetic.shape} "
                  f"(정상: {np.sum(y_val_synthetic == 0):,}, 이상: {np.sum(y_val_synthetic == 1):,}) "
                  f"[{generation_time:.2f}s]")
            
        except Exception as e:
            print(f"      ❌ {anomaly_type} 생성 실패: {e}")
            generation_times[anomaly_type] = None
    
    # 성능 요약 출력
    print(f"\n⚡ 생성 시간 요약 ({'GPU' if actual_use_gpu else 'CPU'}):")
    total_time = 0
    for anomaly_type, gen_time in generation_times.items():
        if gen_time is not None:
            print(f"   {anomaly_type:12s}: {gen_time:6.2f}s")
            total_time += gen_time
        else:
            print(f"   {anomaly_type:12s}: Failed")
    
    if total_time > 0:
        print(f"   {'Total':12s}: {total_time:6.2f}s")
        print(f"   {'Average':12s}: {total_time/len([t for t in generation_times.values() if t is not None]):6.2f}s")
    
    return synthetic_val_sets

def check_gpu_requirements(X_original, y_original):
    """GPU 사용 가능성 및 권장사항 검사"""
    print(f"\n🔍 GPU 가속 요구사항 분석:")
    
    # 데이터 크기 분석
    data_size_mb = X_original.nbytes / 1e6
    n_samples, n_features = X_original.shape
    n_normal = np.sum(y_original == 0)
    n_anomaly = np.sum(y_original == 1)
    
    print(f"   📊 데이터 크기: {data_size_mb:.1f} MB ({n_samples:,} x {n_features})")
    print(f"   📊 정상/이상: {n_normal:,} / {n_anomaly:,}")
    
    # GPU 사용 권장사항
    if not GPU_AVAILABLE:
        print(f"   ❌ GPU 사용 불가 (CuPy 미설치)")
        return False
    
    # GPU 메모리 확인
    free_mem_gb = cp.cuda.Device().mem_info[0] / 1e9
    total_mem_gb = cp.cuda.Device().mem_info[1] / 1e9
    data_size_gb = data_size_mb / 1000
    
    print(f"   💾 GPU 메모리: {free_mem_gb:.1f}/{total_mem_gb:.1f} GB 사용가능")
    
    # 성능 예측
    if n_samples < 5000:
        speedup_estimate = "1-2x"
        recommendation = "CPU 권장 (작은 데이터)"
        use_gpu = False
    elif n_samples < 20000:
        speedup_estimate = "2-5x"
        recommendation = "GPU 권장"
        use_gpu = True
    else:
        speedup_estimate = "5-20x"
        recommendation = "GPU 강력 권장"
        use_gpu = True
    
    # 메모리 충분성 검사
    if data_size_gb > free_mem_gb * 0.7:
        print(f"   ⚠️ GPU 메모리 부족 위험. CPU 사용 권장")
        use_gpu = False
    
    print(f"   🚀 예상 성능 향상: {speedup_estimate}")
    print(f"   💡 권장사항: {recommendation}")
    
    return use_gpu

def main(args):
    """메인 실험 실행 함수"""
    print("🔬 Synthetic Anomaly 기반 모델 선택 실용성 검증 실험 시작")
    print("=" * 80)
    
    # GPU 상태 확인
    if GPU_AVAILABLE:
        print(f"🚀 GPU 가속 활성화: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("🐌 CPU 모드로 실행")
    
    # 결과 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_suffix = "_gpu" if GPU_AVAILABLE else "_cpu"
    results_dir = f"./result_metric/{args.dataset_name}_experiment_results_{timestamp}{gpu_suffix}"
    os.makedirs(results_dir, exist_ok=True)

    # 로그 파일 설정
    log_file = open(os.path.join(results_dir, "experiment_log.txt"), "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    
    try:
        # 1. 데이터셋 준비 (CSV 로드)
        print(f"\n📊 데이터셋 로드: {args.dataset_name}")
        X_original, y_original = load_dataset(args.dataset_name)
        
        # 2. GPU 사용 가능성 검사
        use_gpu_recommended = check_gpu_requirements(X_original, y_original)
        
        # 사용자 GPU 설정 반영
        use_gpu = args.use_gpu and use_gpu_recommended if hasattr(args, 'use_gpu') else use_gpu_recommended
        
        # 3. 데이터셋 분할
        # 3. 데이터셋 분할
        experiment_start_time = time.time()
        
        X_normal_train, X_normal_val, X_val_real, y_val_real, X_test, y_test, X_anomaly_val = prepare_dataset_splits(
            X_original, y_original
        )
        
        # 4. 합성 이상치 검증 세트 생성 (GPU 가속)
        synthetic_val_sets = generate_synthetic_validation_sets(
            X_original, y_original, X_normal_val, X_anomaly_val, use_gpu=use_gpu
        )
        
        # 5. 모델 선택 실험 실행
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
        
        # 6. 성능 요약
        print(f"\n⏱️ 실험 시간 요약:")
        print(f"   합성 데이터 생성: {model_start_time - experiment_start_time:.2f}s")
        print(f"   모델 학습/평가: {model_time:.2f}s")
        print(f"   전체 실험 시간: {total_time:.2f}s")
        
        # GPU 사용 통계 저장 (PyTorch 기반)
        if GPU_AVAILABLE:
            gpu_stats = {
                'gpu_used': use_gpu,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'data_size_mb': X_original.nbytes / 1e6,
                'total_time': total_time
            }
            
            # GPU 통계를 파일로 저장
            import json
            with open(os.path.join(results_dir, 'gpu_performance.json'), 'w') as f:
                json.dump(gpu_stats, f, indent=2)
        
        print(f"\n🎉 실험 성공적으로 완료!")
        print(f"📁 결과 위치: {results_dir}")
        
        if GPU_AVAILABLE and use_gpu:
            print(f"🚀 GPU 가속이 적용되었습니다!")
        
    except Exception as e:
        print(f"\n❌ 실험 중 오류 발생: {e}")
        
        # GPU 메모리 정리 (PyTorch)
        if GPU_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                print("🧹 GPU 메모리 정리 완료")
            except:
                pass
        
        raise
    finally:
        log_file.close()
        
        # GPU 메모리 정리 (PyTorch)
        if GPU_AVAILABLE:
            try:
                torch.cuda.empty_cache()
            except:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Anomaly 기반 모델 선택 실용성 검증 실험")
    parser.add_argument("--dataset_name", type=str, required=True, 
                       help="ADBench 데이터셋 이름 (예: cardio, satellite)")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                       help="GPU 가속 사용 여부 (기본값: True)")
    parser.add_argument("--no_gpu", action="store_true", default=False,
                       help="GPU 사용 강제 비활성화")
    
    args = parser.parse_args()
    
    # GPU 사용 설정 처리
    if args.no_gpu:
        args.use_gpu = False
    
    # 시작 전 GPU 상태 출력
    if GPU_AVAILABLE and args.use_gpu:
        print("🚀 GPU 가속 모드로 실행 준비 완료")
    else:
        print("🐌 CPU 모드로 실행 준비 완료")
    
    main(args)