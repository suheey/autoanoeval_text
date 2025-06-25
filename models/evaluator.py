import numpy as np
import time
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.cof import COF
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.copod import COPOD
from pyod.models.pca import PCA
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD
from pyod.models.loda import LODA
from pyod.models.cblof import CBLOF

from evaluation.metrics import calculate_fdr

def get_default_models():
    """
    기본 PyOD 모델 후보군 반환
    """
    models = {
        'ABOD': ABOD(contamination=0.1, n_neighbors=10),
        'KNN': KNN(contamination=0.1, n_neighbors=5),
        'LOF': LOF(contamination=0.1, n_neighbors=20),
        'COF': COF(contamination=0.1, n_neighbors=20),
        'IForest': IForest(contamination=0.1, random_state=42),
        'OCSVM': OCSVM(contamination=0.1, kernel='rbf'),
        'COPOD': COPOD(contamination=0.1),
        'PCA': PCA(contamination=0.1, random_state=42),
        'HBOS': HBOS(contamination=0.1),
        'MCD': MCD(contamination=0.1, random_state=42),
        'LODA': LODA(contamination=0.1), 
        'CBLOF': CBLOF(contamination=0.1, random_state=42),
    }
    return models

def prepare_data(X_normal_train, X_val_real, X_test, synthetic_val_sets):
    """
    데이터 표준화 수행
    
    Parameters:
    - X_normal_train: 학습용 정상 데이터
    - X_val_real: 실제 이상치가 포함된 검증 데이터
    - X_test: 테스트 데이터
    - synthetic_val_sets: 합성 이상치 유형별 검증 데이터 딕셔너리
    
    Returns:
    - tuple: 표준화된 데이터들
    """
    # 데이터 표준화
    scaler = StandardScaler()
    X_normal_train_scaled = scaler.fit_transform(X_normal_train)
    X_val_real_scaled = scaler.transform(X_val_real)
    X_test_scaled = scaler.transform(X_test)
    
    # 합성 이상치 데이터 표준화
    synthetic_val_sets_scaled = {}
    for anomaly_type, (X_val, y_val) in synthetic_val_sets.items():
        synthetic_val_sets_scaled[anomaly_type] = (scaler.transform(X_val), y_val)
    
    return X_normal_train_scaled, X_val_real_scaled, X_test_scaled, synthetic_val_sets_scaled

def evaluate_single_model(model_name, model, X_train, X_val, y_val, X_test, y_test):
    """
    단일 모델을 평가하는 함수
    
    Parameters:
    - model_name: 모델 이름
    - model: PyOD 모델 객체
    - X_train: 학습 데이터 (정상만)
    - X_val: 검증 데이터
    - y_val: 검증 데이터의 레이블
    - X_test: 테스트 데이터
    - y_test: 테스트 데이터의 레이블
    
    Returns:
    - dict: 모델 성능 결과
    """
    print(f"\n{model_name} 모델 평가 중...")
    
    # 시간 측정 시작
    start_time = time.time()
    
    try:
        # 모델 학습 (정상 데이터만 사용)
        model.fit(X_train)
        
        # 검증 세트에서 이상치 점수 계산
        val_scores = model.decision_function(X_val)
        val_auc = roc_auc_score(y_val, val_scores)
        val_ap = average_precision_score(y_val, val_scores)
        
        # 테스트 세트에서 이상치 점수 계산
        test_scores = model.decision_function(X_test)
        test_auc = roc_auc_score(y_test, test_scores)
        test_ap = average_precision_score(y_test, test_scores)
        
        # FDR 계산 (False Discovery Rate)
        val_predictions = model.predict(X_val)
        val_fdr = calculate_fdr(y_val, val_predictions)
        test_predictions = model.predict(X_test)
        test_fdr = calculate_fdr(y_test, test_predictions)
        
        # 시간 측정 종료
        training_time = time.time() - start_time
        
        # 결과 저장
        result = {
            'val_auc': val_auc,
            'val_ap': val_ap,
            'val_fdr': val_fdr,  # Validation FDR 추가
            'test_auc': test_auc,
            'test_ap': test_ap,
            'test_fdr': test_fdr,
            'training_time': training_time,
            'val_scores': val_scores,
            'test_scores': test_scores
        }
        
        print(f"{model_name} - Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}, Val FDR: {val_fdr:.4f}, "
              f"Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}, Test FDR: {test_fdr:.4f}, "
              f"Time: {training_time:.2f}s")
        
        return result
        
    except Exception as e:
        print(f"{model_name} 모델 평가 중 오류 발생: {e}")
        return {
            'val_auc': float('nan'),
            'val_ap': float('nan'),
            'val_fdr': float('nan'),  # Validation FDR 추가
            'test_auc': float('nan'),
            'test_ap': float('nan'),
            'test_fdr': float('nan'),
            'training_time': float('nan'),
            'val_scores': None,
            'test_scores': None
        }

def evaluate_models(models, X_train, X_val, y_val, X_test, y_test):
    """
    여러 모델을 학습, 검증, 테스트하고 결과를 반환하는 함수
    
    Parameters:
    - models: PyOD 모델 딕셔너리
    - X_train: 학습 데이터 (정상만)
    - X_val: 검증 데이터
    - y_val: 검증 데이터의 레이블
    - X_test: 테스트 데이터
    - y_test: 테스트 데이터의 레이블
    
    Returns:
    - results: 모델별 성능 결과 딕셔너리
    """
    results = {}
    
    for model_name, model in models.items():
        result = evaluate_single_model(model_name, model, X_train, X_val, y_val, X_test, y_test)
        results[model_name] = result
    
    return results

def get_best_model_info(results):
    """
    검증 AUC 기준으로 최고 성능 모델 정보 반환
    
    Parameters:
    - results: 모델별 성능 결과 딕셔너리
    
    Returns:
    - dict: 최고 성능 모델 정보
    """
    # 검증 세트 AUC 기준으로 모델 정렬
    sorted_models = {k: v for k, v in sorted(
        results.items(), 
        key=lambda item: item[1]['val_auc'] if not np.isnan(item[1]['val_auc']) else -float('inf'), 
        reverse=True
    )}
    
    # Best 모델 선택 (검증 AUC 기준)
    if sorted_models:
        best_model_name = list(sorted_models.keys())[0]
        best_model_metrics = sorted_models[best_model_name]
        
        best_model_info = {
            'model_name': best_model_name,
            'val_auc': best_model_metrics['val_auc'],
            'test_auc': best_model_metrics['test_auc'],
            'val_ap': best_model_metrics['val_ap'],
            'test_ap': best_model_metrics['test_ap'],
            'val_fdr': best_model_metrics.get('val_fdr', 0),  # Validation FDR 추가
            'test_fdr': best_model_metrics['test_fdr']
        }
        
        print(f"Best 모델: {best_model_name}")
        print(f"검증 AUC: {best_model_metrics['val_auc']:.4f}, 테스트 AUC: {best_model_metrics['test_auc']:.4f}")
        print(f"검증 AP: {best_model_metrics['val_ap']:.4f}, 테스트 AP: {best_model_metrics['test_ap']:.4f}")
        print(f"테스트 FDR: {best_model_metrics['test_fdr']:.4f}")
        
        return best_model_info
    
    return None

def print_model_ranking(results, metric='test_auc', top_k=5):
    """
    모델 순위를 출력하는 함수
    
    Parameters:
    - results: 모델별 성능 결과 딕셔너리
    - metric: 순위를 매길 메트릭 (기본값: 'test_auc')
    - top_k: 출력할 상위 모델 수 (기본값: 5)
    """
    # 지정된 메트릭 기준으로 모델 정렬
    sorted_models = sorted(
        results.items(), 
        key=lambda item: item[1][metric] if not np.isnan(item[1][metric]) else -float('inf'), 
        reverse=True
    )
    
    print(f"\n=== Top {top_k} 모델 순위 ({metric} 기준) ===")
    for i, (model_name, metrics) in enumerate(sorted_models[:top_k], 1):
        print(f"{i}. {model_name}: {metrics[metric]:.4f}")

def filter_valid_results(results):
    """
    유효한 결과만 필터링하는 함수 (NaN 값 제외)
    
    Parameters:
    - results: 모델별 성능 결과 딕셔너리
    
    Returns:
    - dict: 유효한 결과만 포함하는 딕셔너리
    """
    valid_results = {}
    
    for model_name, metrics in results.items():
        if not np.isnan(metrics['val_auc']) and not np.isnan(metrics['test_auc']):
            valid_results[model_name] = metrics
    
    return valid_results