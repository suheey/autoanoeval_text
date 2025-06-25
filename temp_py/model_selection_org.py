import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
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

def run_model_selection_experiment(X_normal_train, X_val_real, y_val_real, 
                                   synthetic_val_sets, X_test, y_test, results_dir):
    """
    PyOD 모델 선택 실험 실행 함수
    
    Parameters:
    - X_normal_train: 학습용 정상 데이터
    - X_val_real: 실제 이상치가 포함된 검증 데이터
    - y_val_real: 실제 이상치가 포함된 검증 데이터의 레이블
    - synthetic_val_sets: 합성 이상치 유형별 검증 데이터 딕셔너리
    - X_test: 테스트 데이터
    - y_test: 테스트 데이터의 레이블
    - results_dir: 결과 저장 디렉토리
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
    
    # PyOD 모델 후보군 정의
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
    
    # 결과 저장 딕셔너리
    results = {}
    
    # 1. 실제 이상치로 검증하는 경우의 실험
    print("\n1. 실제 이상치로 검증하는 경우의 실험 실행 중...")
    real_val_results = evaluate_models(
        models=models,
        X_train=X_normal_train_scaled,
        X_val=X_val_real_scaled,
        y_val=y_val_real,
        X_test=X_test_scaled,
        y_test=y_test
    )
    
    results['real_validation'] = real_val_results
    
    # 2. 각 유형의 합성 이상치로 검증하는 경우의 실험
    for anomaly_type, (X_val_synthetic_scaled, y_val_synthetic) in synthetic_val_sets_scaled.items():
        print(f"\n2. {anomaly_type} 합성 이상치로 검증하는 경우의 실험 실행 중...")
        synthetic_val_results = evaluate_models(
            models=models,
            X_train=X_normal_train_scaled,
            X_val=X_val_synthetic_scaled,
            y_val=y_val_synthetic,
            X_test=X_test_scaled,
            y_test=y_test
        )
        
        results[f'synthetic_{anomaly_type}_validation'] = synthetic_val_results
    
    # 결과 요약 및 시각화
    print("\n결과 요약 생성 중...")
    summarize_results(results, results_dir)
    
    return results

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
            
            # 시간 측정 종료
            training_time = time.time() - start_time
            
            # 결과 저장
            results[model_name] = {
                'val_auc': val_auc,
                'val_ap': val_ap,
                'test_auc': test_auc,
                'test_ap': test_ap,
                'training_time': training_time
            }
            
            print(f"{model_name} - Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}, Time: {training_time:.2f}s")
            
        except Exception as e:
            print(f"{model_name} 모델 평가 중 오류 발생: {e}")
            results[model_name] = {
                'val_auc': float('nan'),
                'val_ap': float('nan'),
                'test_auc': float('nan'),
                'test_ap': float('nan'),
                'training_time': float('nan')
            }
    
    return results

def summarize_results(results, results_dir):
    """
    모델 선택 실험 결과를 요약하고 시각화하는 함수
    
    Parameters:
    - results: 검증 방식별 모델 성능 결과 딕셔너리
    - results_dir: 결과 저장 디렉토리
    """
    # 결과를 저장할 DataFrame 생성
    summary_data = []
    
    # 각 검증 방식별 best 모델 선택
    best_models = {}
    
    for validation_type, val_results in results.items():
        # 검증 세트 AUC 기준으로 모델 정렬
        sorted_models = {k: v for k, v in sorted(
            val_results.items(), 
            key=lambda item: item[1]['val_auc'] if not np.isnan(item[1]['val_auc']) else -float('inf'), 
            reverse=True
        )}
        
        # Best 모델 선택 (검증 AUC 기준)
        if sorted_models:
            best_model_name = list(sorted_models.keys())[0]
            best_model_metrics = sorted_models[best_model_name]
            
            best_models[validation_type] = {
                'model_name': best_model_name,
                'val_auc': best_model_metrics['val_auc'],
                'test_auc': best_model_metrics['test_auc'],
                'val_ap': best_model_metrics['val_ap'],
                'test_ap': best_model_metrics['test_ap']
            }
            
            print(f"\n{validation_type} 검증 방식의 Best 모델: {best_model_name}")
            print(f"검증 AUC: {best_model_metrics['val_auc']:.4f}, 테스트 AUC: {best_model_metrics['test_auc']:.4f}")
            print(f"검증 AP: {best_model_metrics['val_ap']:.4f}, 테스트 AP: {best_model_metrics['test_ap']:.4f}")
        
        # 모든 모델의 결과를 DataFrame에 추가
        for model_name, metrics in val_results.items():
            row = {
                'validation_type': validation_type,
                'model': model_name,
                'val_auc': metrics['val_auc'],
                'test_auc': metrics['test_auc'],
                'val_ap': metrics['val_ap'],
                'test_ap': metrics['test_ap'],
                'training_time': metrics['training_time']
            }
            summary_data.append(row)
    
    # DataFrame 생성 및 CSV로 저장
    summary_df = pd.DataFrame(summary_data)
    csv_filename = os.path.join(results_dir, 'model_selection_results.csv')
    summary_df.to_csv(csv_filename, index=False)
    print(f"\n모델 선택 결과가 {csv_filename}에 저장되었습니다")
    
    # Best 모델 성능 비교 시각화
    plot_best_models_comparison(best_models, results_dir)
    
    # 검증-테스트 성능 상관관계 분석
    analyze_validation_test_correlation(summary_df, results_dir)
    
    return best_models

def plot_best_models_comparison(best_models, results_dir):
    """
    각 검증 방식별 best 모델의 성능을 비교하는 시각화
    
    Parameters:
    - best_models: 검증 방식별 best 모델 정보 딕셔너리
    - results_dir: 결과 저장 디렉토리
    """
    # 데이터 준비
    validation_types = list(best_models.keys())
    model_names = [info['model_name'] for info in best_models.values()]
    val_aucs = [info['val_auc'] for info in best_models.values()]
    test_aucs = [info['test_auc'] for info in best_models.values()]
    val_aps = [info['val_ap'] for info in best_models.values()]
    test_aps = [info['test_ap'] for info in best_models.values()]
    
    # 표시용 레이블 변환
    display_labels = []
    for vtype in validation_types:
        if vtype == 'real_validation':
            display_labels.append('Real Anomaly')
        elif vtype.startswith('synthetic_'):
            anomaly_type = vtype.replace('synthetic_', '').replace('_validation', '')
            display_labels.append(f'Synthetic {anomaly_type.capitalize()}')
    
    # 그래프 그리기 - AUC
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(display_labels))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, val_aucs, width, label='Validation AUC')
    bars2 = plt.bar(x + width/2, test_aucs, width, label='Test AUC')
    
    plt.xlabel('Validation Method', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.title('Best Model AUC Comparison by Validation Method', fontsize=16)
    plt.xticks(x, display_labels, rotation=20, ha='right')
    plt.ylim(0.5, 1.0)
    plt.legend()
    
    # 모델 이름 표시
    for i, bar in enumerate(bars2):
        plt.text(bar.get_x() + bar.get_width()/2, 0.52,
                model_names[i], ha='center', va='bottom', rotation=90, fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 저장
    auc_filename = os.path.join(results_dir, 'best_models_auc_comparison.png')
    plt.savefig(auc_filename, dpi=300)
    plt.close()
    print(f"AUC 비교 그래프가 {auc_filename}에 저장되었습니다")
    
    # 그래프 그리기 - AP
    plt.figure(figsize=(12, 8))
    
    bars1 = plt.bar(x - width/2, val_aps, width, label='Validation AP')
    bars2 = plt.bar(x + width/2, test_aps, width, label='Test AP')
    
    plt.xlabel('Validation Method', fontsize=12)
    plt.ylabel('Average Precision Score', fontsize=12)
    plt.title('Best Model Average Precision Comparison by Validation Method', fontsize=16)
    plt.xticks(x, display_labels, rotation=20, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    
    # 모델 이름 표시
    for i, bar in enumerate(bars2):
        plt.text(bar.get_x() + bar.get_width()/2, 0.02,
                model_names[i], ha='center', va='bottom', rotation=90, fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 저장
    ap_filename = os.path.join(results_dir, 'best_models_ap_comparison.png')
    plt.savefig(ap_filename, dpi=300)
    plt.close()
    print(f"AP 비교 그래프가 {ap_filename}에 저장되었습니다")

def analyze_validation_test_correlation(summary_df, results_dir):
    """
    검증 성능과 테스트 성능 간의 상관관계를 분석하는 함수
    
    Parameters:
    - summary_df: 실험 결과 요약 DataFrame
    - results_dir: 결과 저장 디렉토리
    """
    validation_types = summary_df['validation_type'].unique()
    
    # 각 검증 방식별로 산점도 생성
    plt.figure(figsize=(15, 10))
    
    for i, val_type in enumerate(validation_types):
        df_subset = summary_df[summary_df['validation_type'] == val_type]
        
        # NaN 값 제외
        df_subset = df_subset.dropna(subset=['val_auc', 'test_auc'])
        
        if len(df_subset) > 0:
            # 색상 및 마커 설정
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
            markers = ['o', '^', 's', 'D', '*', 'X']
            
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # 레이블 변환
            if val_type == 'real_validation':
                label = 'Real Anomaly'
            elif val_type.startswith('synthetic_'):
                anomaly_type = val_type.replace('synthetic_', '').replace('_validation', '')
                label = f'Synthetic {anomaly_type.capitalize()}'
            else:
                label = val_type
            
            # 산점도 그리기
            plt.scatter(
                df_subset['val_auc'],
                df_subset['test_auc'],
                alpha=0.7,
                label=label,
                color=color,
                marker=marker,
                s=100
            )
            
            # 모델 이름 표시
            for _, row in df_subset.iterrows():
                plt.annotate(
                    row['model'],
                    (row['val_auc'], row['test_auc']),
                    fontsize=8,
                    xytext=(5, 5),
                    textcoords='offset points'
                )
    
    # 대각선 그리기 (x=y)
    min_val = min(summary_df['val_auc'].min(), summary_df['test_auc'].min())
    max_val = max(summary_df['val_auc'].max(), summary_df['test_auc'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.xlabel('Validation AUC', fontsize=12)
    plt.ylabel('Test AUC', fontsize=12)
    plt.title('Correlation between Validation and Test Performance', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.tight_layout()
    
    # 저장
    corr_filename = os.path.join(results_dir, 'validation_test_correlation.png')
    plt.savefig(corr_filename, dpi=300)
    plt.close()
    print(f"검증-테스트 상관관계 그래프가 {corr_filename}에 저장되었습니다")
    
    # 추가: 각 검증 방식별 Spearman 상관계수 계산
    correlation_results = []
    
    for val_type in validation_types:
        df_subset = summary_df[summary_df['validation_type'] == val_type].dropna(subset=['val_auc', 'test_auc'])
        
        if len(df_subset) > 1:  # 상관계수를 계산하기 위해 최소 2개 이상의 데이터 필요
            corr_auc = df_subset[['val_auc', 'test_auc']].corr(method='spearman').iloc[0, 1]
            corr_ap = df_subset[['val_ap', 'test_ap']].corr(method='spearman').iloc[0, 1]
            
            correlation_results.append({
                'validation_type': val_type,
                'spearman_corr_auc': corr_auc,
                'spearman_corr_ap': corr_ap
            })
    
    # 상관계수 결과 저장
    if correlation_results:
        corr_df = pd.DataFrame(correlation_results)
        corr_csv = os.path.join(results_dir, 'validation_test_correlation_coefficients.csv')
        corr_df.to_csv(corr_csv, index=False)
        print(f"상관계수 결과가 {corr_csv}에 저장되었습니다")
        
        # 상관계수 시각화
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(corr_df))
        width = 0.35
        
        # 표시용 레이블 변환
        display_labels = []
        for vtype in corr_df['validation_type']:
            if vtype == 'real_validation':
                display_labels.append('Real Anomaly')
            elif vtype.startswith('synthetic_'):
                anomaly_type = vtype.replace('synthetic_', '').replace('_validation', '')
                display_labels.append(f'Synthetic {anomaly_type.capitalize()}')
        
        plt.bar(x - width/2, corr_df['spearman_corr_auc'], width, label='AUC Correlation')
        plt.bar(x + width/2, corr_df['spearman_corr_ap'], width, label='AP Correlation')
        
        plt.xlabel('Validation Method', fontsize=12)
        plt.ylabel('Spearman Correlation Coefficient', fontsize=12)
        plt.title('Correlation between Validation and Test Performance Metrics', fontsize=16)
        plt.xticks(x, display_labels, rotation=20, ha='right')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.ylim(-1.1, 1.1)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        corr_bar_filename = os.path.join(results_dir, 'correlation_coefficients_comparison.png')
        plt.savefig(corr_bar_filename, dpi=300)
        plt.close()
        print(f"상관계수 비교 그래프가 {corr_bar_filename}에 저장되었습니다")