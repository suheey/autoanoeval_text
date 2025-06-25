import numpy as np
import os
from datetime import datetime
import urllib.request
from sklearn.model_selection import train_test_split
from data_generator import SimpleDataGenerator
from gmm_cot_generator import GMM_CoT_AnomalyGenerator  # 새로운 GMM CoT 모듈 임포트
from visualization_cot import visualize_tsne
from model_selection import run_model_selection_experiment
import argparse
import sys

# 설정
RANDOM_SEED = 42
ANOMALY_TYPES = ['local', 'cluster', 'global', 'discrepancy']  # 기존 4가지 이상치 유형

# Cardiotocography 및 의료 데이터 관련 특성 이름 정의
MEDICAL_FEATURE_NAMES = {
    # Cardiotocography 데이터셋 (태아 심박동 모니터링)
    'cardiotocography': ['LB', 'AC', 'FM', 'UC', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 
                        'DL', 'DS', 'DP', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 
                        'Mode', 'Mean', 'Median', 'Variance', 'Tendency'],
    'cardio': ['LB', 'AC', 'FM', 'UC', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 
               'DL', 'DS', 'DP', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 
               'Mode', 'Mean', 'Median', 'Variance'],
    'ctg': ['LB', 'AC', 'FM', 'UC', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 
            'DL', 'DS', 'DP', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 
            'Mode', 'Mean', 'Median', 'Variance']
}

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # 버퍼 바로 비우기
    def flush(self):
        for f in self.files:
            f.flush()

# 데이터셋 다운로드 함수
def download_dataset(url, filename):
    if not os.path.exists(filename):
        print(f"{filename} 다운로드 중...")
        urllib.request.urlretrieve(url, filename)
        print(f"{filename} 다운로드 완료!")
    else:
        print(f"{filename}이 이미 존재합니다.")

def is_medical_dataset(dataset_name):
    """의료 관련 데이터셋인지 확인"""
    medical_keywords = ['cardio', 'ctg', 'thyroid', 'heart', 'ecg', 'medical', 'health']
    return any(keyword in dataset_name.lower() for keyword in medical_keywords)

def get_feature_names_for_dataset(dataset_name, actual_feature_count):
    """데이터셋에 맞는 특성 이름 반환"""
    dataset_lower = dataset_name.lower()
    
    # 의료 데이터셋인 경우 특화된 특성 이름 사용
    for key, feature_names in MEDICAL_FEATURE_NAMES.items():
        if key in dataset_lower:
            # 실제 특성 개수에 맞춰 조정
            if len(feature_names) >= actual_feature_count:
                return feature_names[:actual_feature_count]
            else:
                # 부족하면 generic 이름으로 확장
                extended_names = feature_names + [f'feature_{i}' for i in range(len(feature_names), actual_feature_count)]
                return extended_names
    
    # 일반 데이터셋인 경우 generic 특성 이름 사용
    return [f'feature_{i}' for i in range(actual_feature_count)]

def generate_gmm_cot_validation_sets(X_normal_val, X_anomaly_val, dataset_name, results_dir):
    """
    GMM CoT 기반 합성 검증 세트 생성
    
    Parameters:
    - X_normal_val: 정상 검증 데이터
    - X_anomaly_val: 이상 검증 데이터 (개수 참조용)
    - dataset_name: 데이터셋 이름
    - results_dir: 결과 저장 디렉토리
    
    Returns:
    - cot_validation_sets: CoT 기반 검증 세트 딕셔너리
    """
    print(f"\n{'='*60}")
    print(f"🧠 GMM CoT 기반 합성 이상치 생성 시작")
    print(f"{'='*60}")
    
    # GMM CoT 생성기 초기화
    cot_generator = GMM_CoT_AnomalyGenerator(seed=RANDOM_SEED)
    
    # 특성 이름 설정
    actual_feature_count = X_normal_val.shape[1]
    feature_names = get_feature_names_for_dataset(dataset_name, actual_feature_count)
    cot_generator.set_feature_names(feature_names)
    
    is_medical = is_medical_dataset(dataset_name)
    print(f"데이터셋: {dataset_name}")
    print(f"의료 데이터셋 여부: {is_medical}")
    print(f"특성 수: {actual_feature_count}")
    print(f"사용할 특성 이름: {feature_names[:5]}..." if len(feature_names) > 5 else f"사용할 특성 이름: {feature_names}")
    
    # GMM 학습
    try:
        print(f"\n🔧 GMM 학습 중...")
        cot_generator.fit_gmm_normal(X_normal_val, max_components=8)  # 컴포넌트 수 제한
    except Exception as e:
        print(f"❌ GMM 학습 실패: {e}")
        print("GMM CoT 생성을 건너뜁니다.")
        return {}
    
    # CoT 규칙 생성
    cot_rules = cot_generator.generate_cot_rules()
    
    # 각 CoT 규칙별로 합성 검증 세트 생성
    cot_validation_sets = {}
    target_anomaly_count = len(X_anomaly_val)
    
    if is_medical:
        # 의료 데이터셋인 경우 더 많은 의학적 규칙 사용
        selected_rules = list(cot_rules.keys())[:4]  # 처음 4개 규칙 사용
        alpha = 3.5  # 의학 데이터에 적합한 확장 계수 (보수적)
        max_attempts = 3  # 시간 절약
    else:
        # 일반 데이터셋인 경우 일부 규칙만 사용 (적응적으로)
        selected_rules = ['abnormal_histogram_pattern', 'bradycardia_with_low_variability']  # 일반적으로 적용 가능한 규칙
        alpha = 5.0  # 일반적인 확장 계수
        max_attempts = 2  # 더 적은 시도
    
    print(f"\n📋 사용할 CoT 규칙 ({len(selected_rules)}개): {selected_rules}")
    print(f"목표 이상치 개수: {target_anomaly_count}")
    print(f"GMM 확장 계수 (alpha): {alpha}")
    
    successful_rules = 0
    
    for i, rule_name in enumerate(selected_rules, 1):
        try:
            print(f"\n--- [{i}/{len(selected_rules)}] {rule_name} 규칙 처리 중 ---")
            
            # CoT 필터링 기반 이상치 생성
            synthetic_anomalies = cot_generator.generate_cot_filtered_anomalies(
                target_count=target_anomaly_count,
                rule_name=rule_name,
                alpha=alpha,
                max_attempts=max_attempts
            )
            
            if len(synthetic_anomalies) > 0:
                # 정상 데이터와 결합하여 검증 세트 구성
                actual_normal_count = min(len(X_normal_val), len(synthetic_anomalies))
                X_val_cot = np.vstack([X_normal_val[:actual_normal_count], synthetic_anomalies])
                y_val_cot = np.concatenate([np.zeros(actual_normal_count), 
                                           np.ones(len(synthetic_anomalies))])
                
                # 데이터 셔플
                idx = np.random.RandomState(RANDOM_SEED).permutation(len(y_val_cot))
                X_val_cot = X_val_cot[idx]
                y_val_cot = y_val_cot[idx]
                
                # 짧은 이름으로 저장 (시각화 시 가독성을 위해)
                short_rule_name = rule_name.replace('_with_', '_').replace('_low_', '_')[:20]
                cot_validation_sets[f'cot_{short_rule_name}'] = (X_val_cot, y_val_cot)
                
                successful_rules += 1
                print(f"✅ {rule_name} 검증 세트 생성 완료")
                print(f"   검증 세트 크기: {X_val_cot.shape}")
                print(f"   정상: {np.sum(y_val_cot == 0)}, 이상: {np.sum(y_val_cot == 1)}")
            else:
                print(f"❌ {rule_name} 규칙으로 이상치 생성 실패 (필터링 결과 없음)")
                
        except Exception as e:
            print(f"❌ {rule_name} 규칙 처리 중 오류: {e}")
            continue
    
    # 생성 결과 요약
    print(f"\n🎯 GMM CoT 검증 세트 생성 완료")
    print(f"성공한 규칙: {successful_rules}/{len(selected_rules)}")
    print(f"생성된 CoT 검증 세트: {len(cot_validation_sets)}개")
    
    # 생성된 이상치 분석 및 시각화 (의료 데이터셋이고 성공한 규칙이 있는 경우)
    if is_medical and cot_validation_sets:
        try:
            print(f"\n📊 CoT 이상치 통계 분석 중...")
            
            # 분석용 이상치 딕셔너리 생성
            rule_anomalies = {}
            for cot_key, (X_val, y_val) in cot_validation_sets.items():
                rule_name = cot_key.replace('cot_', '')
                anomalies = X_val[y_val == 1]
                rule_anomalies[rule_name] = anomalies
            
            # 통계 분석
            cot_generator.analyze_generated_anomalies(rule_anomalies)
            
            # 시각화 저장 (의료 데이터인 경우만)
            if len(rule_anomalies) > 0:
                print(f"\n📈 CoT 이상치 분포 시각화 중...")
                
                import matplotlib
                matplotlib.use('Agg')  # GUI 없는 환경을 위해
                import matplotlib.pyplot as plt
                plt.ioff()  # 인터랙티브 모드 끄기
                
                # 주요 의료 특성만 시각화
                medical_features = ['LB', 'AC', 'ASTV', 'DL'] if is_medical else feature_names[:4]
                available_features = [f for f in medical_features if f in feature_names]
                
                if available_features:
                    fig = cot_generator.visualize_anomaly_distribution(
                        X_normal_val, 
                        rule_anomalies, 
                        features_to_plot=available_features[:4]  # 최대 4개 특성
                    )
                    
                    if fig is not None:
                        # 그래프 저장
                        viz_filename = os.path.join(results_dir, 'cot_anomalies_distribution.png')
                        fig.savefig(viz_filename, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        print(f"✅ CoT 이상치 분포 시각화가 {viz_filename}에 저장되었습니다")
                
        except Exception as e:
            print(f"⚠️ CoT 이상치 분석/시각화 중 오류 (계속 진행): {e}")
    
    return cot_validation_sets

def main(args):
    # 결과 저장 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{args.dataset_name}_experiment_results_{timestamp}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # ✅ log.txt로 + 터미널 동시에 출력
    log_file = open(os.path.join(results_dir, "log.txt"), "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    
    print(f"🚀 실험 시작: {args.dataset_name}")
    print(f"📁 결과 저장 디렉토리: {results_dir}")
    
    # 데이터셋 다운로드
    dataset_path = f'/lab-di/nfsdata/home/suhee.yoon/autoanoeval/ADBench/adbench/datasets/Classical/{args.dataset_name}.npz'
    dataset_url = f"https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/{args.dataset_name}.npz"
    download_dataset(dataset_url, dataset_path)

    # 데이터셋 로드
    print("📥 데이터셋 로드 중...")
    data = np.load(dataset_path, allow_pickle=True)
    X_original, y_original = data['X'], data['y']

    print(f"원본 데이터셋 shape: {X_original.shape}")
    print(f"원본 클래스 분포 - 정상: {np.sum(y_original == 0)}, 이상: {np.sum(y_original == 1)}")

    # 데이터셋 준비: 정상 데이터와 이상 데이터 분리
    X_normal = X_original[y_original == 0][:5000]
    X_anomaly = X_original[y_original == 1][:1000]
    
    # 1. 정상 데이터 분할: 학습용(train)과 검증/테스트용(holdout)
    X_normal_train, X_normal_holdout = train_test_split(X_normal, test_size=0.4, random_state=RANDOM_SEED)
    
    # 2. 실제 이상 데이터 분할: 검증용(30%)과 테스트용(70%)
    X_anomaly_val, X_anomaly_test = train_test_split(X_anomaly, test_size=0.7, random_state=RANDOM_SEED)
    
    # 3. 검증 세트와 테스트 세트에 정상 데이터 추가 (holdout 데이터의 절반씩)
    X_normal_val, X_normal_test = train_test_split(X_normal_holdout, test_size=0.5, random_state=RANDOM_SEED)
    
    # Real anomaly validation set과 test set 생성
    X_val_real = np.vstack([X_normal_val, X_anomaly_val])
    y_val_real = np.concatenate([np.zeros(len(X_normal_val)), np.ones(len(X_anomaly_val))])
    
    X_test = np.vstack([X_normal_test, X_anomaly_test])
    y_test = np.concatenate([np.zeros(len(X_normal_test)), np.ones(len(X_anomaly_test))])
    
    # 각 데이터셋 셔플
    def shuffle_data(X, y):
        idx = np.random.RandomState(RANDOM_SEED).permutation(len(y))
        return X[idx], y[idx]
    
    X_val_real, y_val_real = shuffle_data(X_val_real, y_val_real)
    X_test, y_test = shuffle_data(X_test, y_test)
    
    print(f"\n📊 데이터셋 분할 완료:")
    print(f"Train set (정상만): {X_normal_train.shape}")
    print(f"Real validation set: {X_val_real.shape}, 정상: {np.sum(y_val_real == 0)}, 이상: {np.sum(y_val_real == 1)}")
    print(f"Test set: {X_test.shape}, 정상: {np.sum(y_test == 0)}, 이상: {np.sum(y_test == 1)}")
    
    # ===== 기존 방식: 여러 유형의 Synthetic Anomaly 생성 =====
    print(f"\n{'='*60}")
    print(f"🔧 기존 방식 합성 이상치 생성")
    print(f"{'='*60}")
    
    data_generator = SimpleDataGenerator(seed=RANDOM_SEED)
    synthetic_val_sets = {}
    synthetic_anomalies_by_type = {}  # 각 유형별 이상치 저장할 딕셔너리 추가
    
    for anomaly_type in ANOMALY_TYPES:
        print(f"\n{anomaly_type} 유형의 합성 이상치로 검증 세트 생성 중...")
        
        # 합성 이상치 생성 (실제 이상치와 동일한 개수로)
        synthetic_anomalies = data_generator.generate_anomalies(
            X=X_original,
            y=y_original,
            anomaly_type=anomaly_type,
            alpha=5,  # local, cluster 이상치 강도
            percentage=0.2,  # global 이상치 범위
            anomaly_count=len(X_anomaly_val)
        )
        
        # 생성한 이상치 저장
        synthetic_anomalies_by_type[anomaly_type] = synthetic_anomalies
        
        # 합성 이상치로 검증 세트 생성
        X_val_synthetic = np.vstack([X_normal_val, synthetic_anomalies])
        y_val_synthetic = np.concatenate([np.zeros(len(X_normal_val)), np.ones(len(synthetic_anomalies))])
        X_val_synthetic, y_val_synthetic = shuffle_data(X_val_synthetic, y_val_synthetic)
        
        synthetic_val_sets[anomaly_type] = (X_val_synthetic, y_val_synthetic)
        
        print(f"{anomaly_type} 검증 세트: {X_val_synthetic.shape}, 정상: {np.sum(y_val_synthetic == 0)}, 이상: {np.sum(y_val_synthetic == 1)}")
    
    # 각 유형에서 1/4씩 이상치 샘플링하여 혼합 데이터셋 생성
    print("\n혼합 이상치(mixed) 검증 세트 생성 중...")
    mixed_anomalies = []
    anomalies_per_type = len(X_anomaly_val) // len(ANOMALY_TYPES)  # 각 유형별로 가져올 이상치 수
    
    for anomaly_type in ANOMALY_TYPES:
        # 각 유형에서 필요한 수만큼 이상치 선택
        anomalies = synthetic_anomalies_by_type[anomaly_type]
        if len(anomalies) > anomalies_per_type:
            # 무작위로 필요한 수만큼만 선택
            indices = np.random.RandomState(RANDOM_SEED).choice(
                len(anomalies), anomalies_per_type, replace=False
            )
            selected_anomalies = anomalies[indices]
        else:
            # 이상치 수가 부족하면 전부 사용
            selected_anomalies = anomalies
        
        mixed_anomalies.append(selected_anomalies)
    
    # 모든 선택된 이상치 합치기
    mixed_anomalies = np.vstack(mixed_anomalies)
    
    # 혼합 이상치로 검증 세트 생성
    X_val_mixed = np.vstack([X_normal_val, mixed_anomalies])
    y_val_mixed = np.concatenate([np.zeros(len(X_normal_val)), np.ones(len(mixed_anomalies))])
    X_val_mixed, y_val_mixed = shuffle_data(X_val_mixed, y_val_mixed)
    
    # 혼합 이상치 검증 세트 추가
    synthetic_val_sets['mixed'] = (X_val_mixed, y_val_mixed)
    
    print(f"혼합(mixed) 검증 세트: {X_val_mixed.shape}, 정상: {np.sum(y_val_mixed == 0)}, 이상: {np.sum(y_val_mixed == 1)}")
    print(f"혼합 이상치 구성: 각 유형당 약 {anomalies_per_type}개씩, 총 {len(mixed_anomalies)}개")
    
    # ===== 새로운 방식: GMM CoT 기반 Synthetic Anomaly 생성 =====
    # GMM CoT 검증 세트 생성
    cot_validation_sets = generate_gmm_cot_validation_sets(
        X_normal_val=X_normal_val, 
        X_anomaly_val=X_anomaly_val,
        dataset_name=args.dataset_name,
        results_dir=results_dir
    )
    
    # ===== 모든 검증 세트 통합 =====
    # 기존 합성 검증 세트와 CoT 검증 세트 결합
    all_synthetic_val_sets = {**synthetic_val_sets, **cot_validation_sets}
    
    print(f"\n📋 전체 검증 세트 요약:")
    print(f"• 실제 이상치 검증 세트: 1개")
    print(f"• 기존 합성 이상치 검증 세트: {len(synthetic_val_sets)}개 ({list(synthetic_val_sets.keys())})")
    print(f"• GMM CoT 합성 이상치 검증 세트: {len(cot_validation_sets)}개 ({list(cot_validation_sets.keys())})")
    print(f"• 총 검증 세트 수: {len(all_synthetic_val_sets) + 1}개")
    
    # ===== t-SNE 시각화 =====
    print(f"\n{'='*60}")
    print(f"📈 t-SNE 시각화 생성")
    print(f"{'='*60}")
    
    # 실제 이상치에 대한 시각화
    visualize_tsne(
        X_test, y_test, None,
        title='Real Anomalies t-SNE Visualization',
        filename=os.path.join(results_dir, 'real_anomalies_tsne.png'),
        anomaly_types=None
    )
    
    # 모든 합성 이상치 유형과 정상 데이터 결합 (기존 방식)
    X_all_synthetic = X_normal_test.copy()
    y_all_synthetic = np.zeros(len(X_normal_test))
    y_types = np.zeros(len(X_normal_test))
    
    all_anomaly_types = []
    
    # 기존 합성 이상치 유형 추가 (mixed 제외)
    for i, anomaly_type in enumerate(ANOMALY_TYPES, 1):
        if anomaly_type in synthetic_val_sets:
            X_val, y_val = synthetic_val_sets[anomaly_type]
            synthetic_anomalies = X_val[y_val == 1]
            
            X_all_synthetic = np.vstack([X_all_synthetic, synthetic_anomalies])
            y_all_synthetic = np.concatenate([y_all_synthetic, np.ones(len(synthetic_anomalies))])
            y_types = np.concatenate([y_types, np.full(len(synthetic_anomalies), i)])
            all_anomaly_types.append(anomaly_type)
    
    # 기존 + CoT 합성 이상치 통합 시각화 (샘플링하여 크기 제한)
    if cot_validation_sets:
        X_all_with_cot = X_normal_test.copy()
        y_all_with_cot = np.zeros(len(X_normal_test))
        y_types_with_cot = np.zeros(len(X_normal_test))
        
        cot_anomaly_types = []
        
        # 기존 합성 이상치 추가 (샘플링)
        for i, anomaly_type in enumerate(ANOMALY_TYPES, 1):
            if anomaly_type in synthetic_val_sets:
                X_val, y_val = synthetic_val_sets[anomaly_type]
                synthetic_anomalies = X_val[y_val == 1]
                
                # 시각화를 위해 샘플링 (너무 많으면 시각화가 어려움)
                if len(synthetic_anomalies) > 200:
                    indices = np.random.choice(len(synthetic_anomalies), 200, replace=False)
                    synthetic_anomalies = synthetic_anomalies[indices]
                
                X_all_with_cot = np.vstack([X_all_with_cot, synthetic_anomalies])
                y_all_with_cot = np.concatenate([y_all_with_cot, np.ones(len(synthetic_anomalies))])
                y_types_with_cot = np.concatenate([y_types_with_cot, np.full(len(synthetic_anomalies), i)])
                cot_anomaly_types.append(f'Traditional-{anomaly_type}')
        
        # CoT 합성 이상치 추가 (샘플링)
        for i, (cot_type, (X_val, y_val)) in enumerate(cot_validation_sets.items(), len(ANOMALY_TYPES) + 1):
            synthetic_anomalies = X_val[y_val == 1]
            
            # 시각화를 위해 샘플링
            if len(synthetic_anomalies) > 200:
                indices = np.random.choice(len(synthetic_anomalies), 200, replace=False)
                synthetic_anomalies = synthetic_anomalies[indices]
            
            X_all_with_cot = np.vstack([X_all_with_cot, synthetic_anomalies])
            y_all_with_cot = np.concatenate([y_all_with_cot, np.ones(len(synthetic_anomalies))])
            y_types_with_cot = np.concatenate([y_types_with_cot, np.full(len(synthetic_anomalies), i)])
            cot_anomaly_types.append(f'CoT-{cot_type.replace("cot_", "")}')
        
        # 통합 시각화
        visualize_tsne(
            X_all_with_cot, y_all_with_cot, y_types_with_cot,
            title='All Synthetic Anomalies t-SNE (Traditional + GMM CoT)',
            filename=os.path.join(results_dir, 'all_synthetic_anomalies_with_cot_tsne.png'),
            anomaly_types=cot_anomaly_types
        )
    
    # 기존 합성 이상치 유형별 시각화
    visualize_tsne(
        X_all_synthetic, y_all_synthetic, y_types,
        title='Traditional Synthetic Anomalies t-SNE Visualization',
        filename=os.path.join(results_dir, 'traditional_synthetic_anomalies_tsne.png'),
        anomaly_types=ANOMALY_TYPES
    )
    
    # 혼합 이상치 시각화 (안전하게)
    try:
        # y_types_mixed 생성 (0: 정상, 1: 혼합 이상치)
        y_types_mixed = np.zeros(len(y_val_mixed))
        y_types_mixed[y_val_mixed == 1] = 1
        
        # 혼합 이상치 시각화
        visualize_tsne(
            X_val_mixed, y_val_mixed, y_types_mixed,
            title='Mixed Anomalies t-SNE Visualization',
            filename=os.path.join(results_dir, 'mixed_anomalies_tsne.png'),
            anomaly_types=['normal', 'mixed_anomalies']
        )
    except Exception as e:
        print(f"⚠️ 혼합 이상치 t-SNE 시각화 중 오류 (건너뜀): {e}")
    
    # ===== PyOD 모델 선택 실험 실행 =====
    print(f"\n{'='*60}")
    print(f"🤖 PyOD 모델 선택 실험 실행")
    print(f"{'='*60}")
    
    print(f"총 {len(all_synthetic_val_sets)}개의 합성 검증 세트로 실험:")
    for i, (val_name, (X_val, y_val)) in enumerate(all_synthetic_val_sets.items(), 1):
        val_type = "GMM CoT" if val_name.startswith("cot_") else "Traditional"
        print(f"  {i}. {val_name} ({val_type}) - 크기: {X_val.shape}, 이상치: {np.sum(y_val == 1)}개")
    
    run_model_selection_experiment(
        X_normal_train=X_normal_train,
        X_val_real=X_val_real, y_val_real=y_val_real,
        synthetic_val_sets=all_synthetic_val_sets,  # 기존 + CoT 모든 검증 세트
        X_test=X_test, y_test=y_test,
        results_dir=results_dir
    )
    
    print(f"\n🎉 실험 완료!")
    print(f"📁 결과는 {results_dir}에 저장되었습니다.")
    print(f"\n📋 생성된 주요 파일들:")
    print(f"  • model_selection_results.csv - 전체 모델 성능 결과")
    print(f"  • top3_models_results.csv - Top-3 모델 결과")
    print(f"  • top3_analysis_summary_report.txt - 분석 요약 리포트")
    if cot_validation_sets:
        print(f"  • cot_anomalies_distribution.png - CoT 이상치 분포 (의료 데이터인 경우)")
    print(f"  • *_tsne.png - t-SNE 시각화 파일들")
    print(f"  • top3_models_*_comparison.png - Top-3 모델 비교 그래프들")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMM CoT 기반 이상치 탐지 모델 선택 실험")
    parser.add_argument("--dataset_name", type=str, required=True, 
                       help="데이터셋 이름 (예: cardiotocography, thyroid, arrhythmia 등)")
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 실험이 중단되었습니다.")
    except Exception as e:
        print(f"\n\n❌ 실험 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()