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
    모델 선택 실험 결과를 요약하고 시각화하는 함수 (Top-3 포함)
    
    Parameters:
    - results: 검증 방식별 모델 성능 결과 딕셔너리
    - results_dir: 결과 저장 디렉토리
    """
    # 결과를 저장할 DataFrame 생성
    summary_data = []
    
    # 각 검증 방식별 top-3 모델 선택
    top_models = {}
    
    for validation_type, val_results in results.items():
        # 검증 세트 AUC 기준으로 모델 정렬
        sorted_models = {k: v for k, v in sorted(
            val_results.items(), 
            key=lambda item: item[1]['val_auc'] if not np.isnan(item[1]['val_auc']) else -float('inf'), 
            reverse=True
        )}
        
        # Top-3 모델 선택 (검증 AUC 기준)
        top_3_models = {}
        valid_models = [(k, v) for k, v in sorted_models.items() if not np.isnan(v['val_auc'])]
        
        for i, (model_name, metrics) in enumerate(valid_models[:3]):  # top-3만 선택
            rank = i + 1
            top_3_models[f'rank_{rank}'] = {
                'model_name': model_name,
                'val_auc': metrics['val_auc'],
                'test_auc': metrics['test_auc'],
                'val_ap': metrics['val_ap'],
                'test_ap': metrics['test_ap'],
                'training_time': metrics['training_time']
            }
        
        top_models[validation_type] = top_3_models
        
        # Top-3 모델 정보 출력
        print(f"\n{validation_type} 검증 방식의 Top-3 모델:")
        for rank, model_info in top_3_models.items():
            print(f"  {rank}: {model_info['model_name']} - Val AUC: {model_info['val_auc']:.4f}, Test AUC: {model_info['test_auc']:.4f}")
        
        # 모든 모델의 결과를 DataFrame에 추가 (랭킹 정보 포함)
        for model_name, metrics in val_results.items():
            # 랭킹 정보 추가
            rank = None
            for rank_key, model_info in top_3_models.items():
                if model_info['model_name'] == model_name:
                    rank = int(rank_key.split('_')[1])
                    break
            
            row = {
                'validation_type': validation_type,
                'model': model_name,
                'rank': rank,  # Top-3에 들지 않으면 None
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
    
    # Top-3 모델 정보만 별도로 저장
    save_top3_results(top_models, results_dir)
    
    # Top-3 모델 성능 비교 시각화
    plot_top_models_comparison(top_models, results_dir)
    
    # Top-3 모델 성능 히트맵 생성
    plot_top3_performance_heatmap(top_models, results_dir)
    
    # 검증-테스트 성능 상관관계 분석
    analyze_validation_test_correlation(summary_df, results_dir)
    
    # Top-3 분석 요약 리포트 생성
    generate_top3_summary_report(top_models, results_dir)
    
    return top_models

def save_top3_results(top_models, results_dir):
    """
    Top-3 모델 결과를 별도 CSV로 저장
    
    Parameters:
    - top_models: 검증 방식별 top-3 모델 정보 딕셔너리
    - results_dir: 결과 저장 디렉토리
    """
    top3_data = []
    
    for validation_type, ranks in top_models.items():
        for rank_key, model_info in ranks.items():
            rank = int(rank_key.split('_')[1])
            row = {
                'validation_type': validation_type,
                'rank': rank,
                'model_name': model_info['model_name'],
                'val_auc': model_info['val_auc'],
                'test_auc': model_info['test_auc'],
                'val_ap': model_info['val_ap'],
                'test_ap': model_info['test_ap'],
                'training_time': model_info['training_time']
            }
            top3_data.append(row)
    
    top3_df = pd.DataFrame(top3_data)
    top3_csv = os.path.join(results_dir, 'top3_models_results.csv')
    top3_df.to_csv(top3_csv, index=False)
    print(f"Top-3 모델 결과가 {top3_csv}에 저장되었습니다")

def plot_top_models_comparison(top_models, results_dir):
    """
    각 검증 방식별 top-3 모델의 성능을 비교하는 시각화
    
    Parameters:
    - top_models: 검증 방식별 top-3 모델 정보 딕셔너리
    - results_dir: 결과 저장 디렉토리
    """
    validation_types = list(top_models.keys())
    
    # 표시용 레이블 변환
    display_labels = []
    for vtype in validation_types:
        if vtype == 'real_validation':
            display_labels.append('Real Anomaly')
        elif vtype.startswith('synthetic_'):
            anomaly_type = vtype.replace('synthetic_', '').replace('_validation', '')
            display_labels.append(f'Synthetic {anomaly_type.capitalize()}')
    
    # AUC 비교 그래프
    plt.figure(figsize=(16, 10))
    
    x = np.arange(len(display_labels))
    width = 0.25  # 3개 모델을 표시하기 위해 좁게 설정
    
    # 각 랭크별 데이터 수집
    rank1_val_aucs = []
    rank1_test_aucs = []
    rank1_models = []
    
    rank2_val_aucs = []
    rank2_test_aucs = []
    rank2_models = []
    
    rank3_val_aucs = []
    rank3_test_aucs = []
    rank3_models = []
    
    for validation_type in validation_types:
        ranks = top_models[validation_type]
        
        # Rank 1
        if 'rank_1' in ranks:
            rank1_val_aucs.append(ranks['rank_1']['val_auc'])
            rank1_test_aucs.append(ranks['rank_1']['test_auc'])
            rank1_models.append(ranks['rank_1']['model_name'])
        else:
            rank1_val_aucs.append(0)
            rank1_test_aucs.append(0)
            rank1_models.append('')
        
        # Rank 2
        if 'rank_2' in ranks:
            rank2_val_aucs.append(ranks['rank_2']['val_auc'])
            rank2_test_aucs.append(ranks['rank_2']['test_auc'])
            rank2_models.append(ranks['rank_2']['model_name'])
        else:
            rank2_val_aucs.append(0)
            rank2_test_aucs.append(0)
            rank2_models.append('')
        
        # Rank 3
        if 'rank_3' in ranks:
            rank3_val_aucs.append(ranks['rank_3']['val_auc'])
            rank3_test_aucs.append(ranks['rank_3']['test_auc'])
            rank3_models.append(ranks['rank_3']['model_name'])
        else:
            rank3_val_aucs.append(0)
            rank3_test_aucs.append(0)
            rank3_models.append('')
    
    # 검증 AUC 바 그래프
    bars1_1 = plt.bar(x - width, rank1_val_aucs, width, label='Rank 1 (Val)', alpha=0.8, color='darkblue')
    bars1_2 = plt.bar(x, rank2_val_aucs, width, label='Rank 2 (Val)', alpha=0.8, color='blue')
    bars1_3 = plt.bar(x + width, rank3_val_aucs, width, label='Rank 3 (Val)', alpha=0.8, color='lightblue')
    
    plt.xlabel('Validation Method', fontsize=12)
    plt.ylabel('Validation AUC Score', fontsize=12)
    plt.title('Top-3 Models Validation AUC Comparison by Validation Method', fontsize=16)
    plt.xticks(x, display_labels, rotation=20, ha='right')
    plt.ylim(0.5, 1.0)
    plt.legend()
    
    # 모델 이름 표시
    for i, (bar1, bar2, bar3) in enumerate(zip(bars1_1, bars1_2, bars1_3)):
        if rank1_models[i]:
            plt.text(bar1.get_x() + bar1.get_width()/2, 0.52,
                    rank1_models[i], ha='center', va='bottom', rotation=90, fontsize=9)
        if rank2_models[i]:
            plt.text(bar2.get_x() + bar2.get_width()/2, 0.52,
                    rank2_models[i], ha='center', va='bottom', rotation=90, fontsize=9)
        if rank3_models[i]:
            plt.text(bar3.get_x() + bar3.get_width()/2, 0.52,
                    rank3_models[i], ha='center', va='bottom', rotation=90, fontsize=9)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 저장
    val_auc_filename = os.path.join(results_dir, 'top3_models_validation_auc_comparison.png')
    plt.savefig(val_auc_filename, dpi=300)
    plt.close()
    print(f"Top-3 검증 AUC 비교 그래프가 {val_auc_filename}에 저장되었습니다")
    
    # 테스트 AUC 비교 그래프
    plt.figure(figsize=(16, 10))
    
    bars2_1 = plt.bar(x - width, rank1_test_aucs, width, label='Rank 1 (Test)', alpha=0.8, color='darkred')
    bars2_2 = plt.bar(x, rank2_test_aucs, width, label='Rank 2 (Test)', alpha=0.8, color='red')
    bars2_3 = plt.bar(x + width, rank3_test_aucs, width, label='Rank 3 (Test)', alpha=0.8, color='lightcoral')
    
    plt.xlabel('Validation Method', fontsize=12)
    plt.ylabel('Test AUC Score', fontsize=12)
    plt.title('Top-3 Models Test AUC Comparison by Validation Method', fontsize=16)
    plt.xticks(x, display_labels, rotation=20, ha='right')
    plt.ylim(0.5, 1.0)
    plt.legend()
    
    # 모델 이름 표시
    for i, (bar1, bar2, bar3) in enumerate(zip(bars2_1, bars2_2, bars2_3)):
        if rank1_models[i]:
            plt.text(bar1.get_x() + bar1.get_width()/2, 0.52,
                    rank1_models[i], ha='center', va='bottom', rotation=90, fontsize=9)
        if rank2_models[i]:
            plt.text(bar2.get_x() + bar2.get_width()/2, 0.52,
                    rank2_models[i], ha='center', va='bottom', rotation=90, fontsize=9)
        if rank3_models[i]:
            plt.text(bar3.get_x() + bar3.get_width()/2, 0.52,
                    rank3_models[i], ha='center', va='bottom', rotation=90, fontsize=9)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 저장
    test_auc_filename = os.path.join(results_dir, 'top3_models_test_auc_comparison.png')
    plt.savefig(test_auc_filename, dpi=300)
    plt.close()
    print(f"Top-3 테스트 AUC 비교 그래프가 {test_auc_filename}에 저장되었습니다")
    
    # Combined AUC 비교 그래프 (검증 vs 테스트)
    plot_combined_top3_comparison(top_models, results_dir)

def plot_combined_top3_comparison(top_models, results_dir):
    """
    Top-3 모델의 검증 vs 테스트 성능을 함께 비교하는 그래프
    
    Parameters:
    - top_models: 검증 방식별 top-3 모델 정보 딕셔너리
    - results_dir: 결과 저장 디렉토리
    """
    validation_types = list(top_models.keys())
    
    # 표시용 레이블 변환
    display_labels = []
    for vtype in validation_types:
        if vtype == 'real_validation':
            display_labels.append('Real Anomaly')
        elif vtype.startswith('synthetic_'):
            anomaly_type = vtype.replace('synthetic_', '').replace('_validation', '')
            display_labels.append(f'Synthetic {anomaly_type.capitalize()}')
    
    # 서브플롯 생성 (각 검증 방식마다 하나씩)
    fig, axes = plt.subplots(1, len(validation_types), figsize=(5*len(validation_types), 6))
    if len(validation_types) == 1:
        axes = [axes]
    
    for i, (validation_type, display_label) in enumerate(zip(validation_types, display_labels)):
        ax = axes[i]
        ranks = top_models[validation_type]
        
        # 데이터 준비
        models = []
        val_aucs = []
        test_aucs = []
        
        for rank_key in ['rank_1', 'rank_2', 'rank_3']:
            if rank_key in ranks:
                models.append(ranks[rank_key]['model_name'])
                val_aucs.append(ranks[rank_key]['val_auc'])
                test_aucs.append(ranks[rank_key]['test_auc'])
        
        if models:  # 데이터가 있는 경우에만 그래프 그리기
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, val_aucs, width, label='Validation AUC', alpha=0.8)
            bars2 = ax.bar(x + width/2, test_aucs, width, label='Test AUC', alpha=0.8)
            
            ax.set_xlabel('Model Rank', fontsize=10)
            ax.set_ylabel('AUC Score', fontsize=10)
            ax.set_title(f'{display_label}', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels([f'Rank {j+1}\n{model}' for j, model in enumerate(models)], 
                              rotation=45, ha='right', fontsize=9)
            ax.set_ylim(0.5, 1.0)
            ax.legend(fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 저장
    combined_filename = os.path.join(results_dir, 'top3_models_combined_comparison.png')
    plt.savefig(combined_filename, dpi=300)
    plt.close()
    print(f"Top-3 통합 비교 그래프가 {combined_filename}에 저장되었습니다")

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
            
            # Top-3 모델 강조 표시
            top3_subset = df_subset[df_subset['rank'].notna()]
            other_subset = df_subset[df_subset['rank'].isna()]
            
            # Top-3 모델
            if len(top3_subset) > 0:
                plt.scatter(
                    top3_subset['val_auc'],
                    top3_subset['test_auc'],
                    alpha=0.9,
                    label=f'{label} (Top-3)',
                    color=color,
                    marker=marker,
                    s=150,
                    edgecolors='black',
                    linewidth=2
                )
                
                # Top-3 모델 이름 표시
                for _, row in top3_subset.iterrows():
                    plt.annotate(
                        f"{row['model']} (R{int(row['rank'])})",
                        (row['val_auc'], row['test_auc']),
                        fontsize=8,
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontweight='bold'
                    )
            
            # 나머지 모델들
            if len(other_subset) > 0:
                plt.scatter(
                    other_subset['val_auc'],
                    other_subset['test_auc'],
                    alpha=0.4,
                    color=color,
                    marker=marker,
                    s=50
                )
    
    # 대각선 그리기 (x=y)
    min_val = min(summary_df['val_auc'].min(), summary_df['test_auc'].min())
    max_val = max(summary_df['val_auc'].max(), summary_df['test_auc'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.xlabel('Validation AUC', fontsize=12)
    plt.ylabel('Test AUC', fontsize=12)
    plt.title('Correlation between Validation and Test Performance (Top-3 Highlighted)', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.tight_layout()
    
    # 저장
    corr_filename = os.path.join(results_dir, 'validation_test_correlation_top3.png')
    plt.savefig(corr_filename, dpi=300)
    plt.close()
    print(f"검증-테스트 상관관계 그래프 (Top-3 강조)가 {corr_filename}에 저장되었습니다")
    
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

def plot_top3_performance_heatmap(top_models, results_dir):
    """
    Top-3 모델들의 성능을 히트맵으로 시각화
    
    Parameters:
    - top_models: 검증 방식별 top-3 모델 정보 딕셔너리
    - results_dir: 결과 저장 디렉토리
    """
    # 데이터 준비
    validation_methods = []
    model_names = []
    val_aucs = []
    test_aucs = []
    ranks = []
    
    for validation_type, ranks_dict in top_models.items():
        # 표시용 레이블 변환
        if validation_type == 'real_validation':
            display_label = 'Real Anomaly'
        elif validation_type.startswith('synthetic_'):
            anomaly_type = validation_type.replace('synthetic_', '').replace('_validation', '')
            display_label = f'Synthetic {anomaly_type.capitalize()}'
        else:
            display_label = validation_type
        
        for rank_key, model_info in ranks_dict.items():
            rank = int(rank_key.split('_')[1])
            validation_methods.append(display_label)
            model_names.append(f"{model_info['model_name']} (R{rank})")
            val_aucs.append(model_info['val_auc'])
            test_aucs.append(model_info['test_auc'])
            ranks.append(rank)
    
    # DataFrame 생성
    heatmap_df = pd.DataFrame({
        'Validation_Method': validation_methods,
        'Model': model_names,
        'Validation_AUC': val_aucs,
        'Test_AUC': test_aucs,
        'Rank': ranks
    })
    
    # 피벗 테이블 생성 (검증 AUC)
    pivot_val = heatmap_df.pivot_table(
        index='Model', 
        columns='Validation_Method', 
        values='Validation_AUC', 
        aggfunc='mean'
    )
    
    # 피벗 테이블 생성 (테스트 AUC)
    pivot_test = heatmap_df.pivot_table(
        index='Model', 
        columns='Validation_Method', 
        values='Test_AUC', 
        aggfunc='mean'
    )
    
    # 히트맵 그리기
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    
    # 검증 AUC 히트맵
    im1 = ax1.imshow(pivot_val.values, cmap='YlOrRd', aspect='auto', vmin=0.5, vmax=1.0)
    ax1.set_xticks(range(len(pivot_val.columns)))
    ax1.set_yticks(range(len(pivot_val.index)))
    ax1.set_xticklabels(pivot_val.columns, rotation=45, ha='right')
    ax1.set_yticklabels(pivot_val.index)
    ax1.set_title('Top-3 Models Validation AUC Heatmap', fontsize=14, pad=20)
    
    # 값 표시
    for i in range(len(pivot_val.index)):
        for j in range(len(pivot_val.columns)):
            if not np.isnan(pivot_val.iloc[i, j]):
                text = ax1.text(j, i, f'{pivot_val.iloc[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=10)
    
    # 컬러바
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Validation AUC', rotation=270, labelpad=15)
    
    # 테스트 AUC 히트맵
    im2 = ax2.imshow(pivot_test.values, cmap='YlOrRd', aspect='auto', vmin=0.5, vmax=1.0)
    ax2.set_xticks(range(len(pivot_test.columns)))
    ax2.set_yticks(range(len(pivot_test.index)))
    ax2.set_xticklabels(pivot_test.columns, rotation=45, ha='right')
    ax2.set_yticklabels(pivot_test.index)
    ax2.set_title('Top-3 Models Test AUC Heatmap', fontsize=14, pad=20)
    
    # 값 표시
    for i in range(len(pivot_test.index)):
        for j in range(len(pivot_test.columns)):
            if not np.isnan(pivot_test.iloc[i, j]):
                text = ax2.text(j, i, f'{pivot_test.iloc[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=10)
    
    # 컬러바
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Test AUC', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    # 저장
    heatmap_filename = os.path.join(results_dir, 'top3_models_performance_heatmap.png')
    plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top-3 모델 성능 히트맵이 {heatmap_filename}에 저장되었습니다")

def generate_top3_summary_report(top_models, results_dir):
    """
    Top-3 모델 분석에 대한 요약 리포트 생성
    
    Parameters:
    - top_models: 검증 방식별 top-3 모델 정보 딕셔너리
    - results_dir: 결과 저장 디렉토리
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("TOP-3 MODEL SELECTION ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 전체 요약
    report_lines.append("📊 OVERALL SUMMARY")
    report_lines.append("-" * 50)
    
    all_models = set()
    validation_methods = list(top_models.keys())
    
    for validation_type, ranks in top_models.items():
        for rank_key, model_info in ranks.items():
            all_models.add(model_info['model_name'])
    
    report_lines.append(f"• Total validation methods tested: {len(validation_methods)}")
    report_lines.append(f"• Unique models appearing in top-3: {len(all_models)}")
    report_lines.append(f"• Models tested: {', '.join(sorted(all_models))}")
    report_lines.append("")
    
    # 각 검증 방식별 상세 결과
    for validation_type, ranks in top_models.items():
        # 레이블 변환
        if validation_type == 'real_validation':
            display_label = '🎯 Real Anomaly Validation'
        elif validation_type.startswith('synthetic_'):
            anomaly_type = validation_type.replace('synthetic_', '').replace('_validation', '')
            display_label = f'🔧 Synthetic {anomaly_type.capitalize()} Validation'
        else:
            display_label = validation_type
        
        report_lines.append(display_label)
        report_lines.append("-" * len(display_label))
        
        for i in range(1, 4):  # Rank 1, 2, 3
            rank_key = f'rank_{i}'
            if rank_key in ranks:
                model_info = ranks[rank_key]
                report_lines.append(f"  Rank {i}: {model_info['model_name']}")
                report_lines.append(f"    • Validation AUC: {model_info['val_auc']:.4f}")
                report_lines.append(f"    • Test AUC: {model_info['test_auc']:.4f}")
                report_lines.append(f"    • Validation AP: {model_info['val_ap']:.4f}")
                report_lines.append(f"    • Test AP: {model_info['test_ap']:.4f}")
                report_lines.append(f"    • Training Time: {model_info['training_time']:.2f}s")
                
                # 성능 일치도 분석
                val_test_diff = abs(model_info['val_auc'] - model_info['test_auc'])
                if val_test_diff < 0.05:
                    consistency = "🟢 Excellent consistency"
                elif val_test_diff < 0.10:
                    consistency = "🟡 Good consistency"
                else:
                    consistency = "🔴 Poor consistency"
                
                report_lines.append(f"    • Val-Test Consistency: {consistency} (diff: {val_test_diff:.4f})")
                report_lines.append("")
        
        report_lines.append("")
    
    # 모델별 등장 빈도 분석
    report_lines.append("🏆 MODEL RANKING FREQUENCY ANALYSIS")
    report_lines.append("-" * 50)
    
    model_appearances = {}
    model_rank_positions = {}
    
    for validation_type, ranks in top_models.items():
        for rank_key, model_info in ranks.items():
            model_name = model_info['model_name']
            rank = int(rank_key.split('_')[1])
            
            if model_name not in model_appearances:
                model_appearances[model_name] = 0
                model_rank_positions[model_name] = []
            
            model_appearances[model_name] += 1
            model_rank_positions[model_name].append(rank)
    
    # 등장 빈도별 정렬
    sorted_models = sorted(model_appearances.items(), key=lambda x: x[1], reverse=True)
    
    for model_name, count in sorted_models:
        avg_rank = np.mean(model_rank_positions[model_name])
        rank_distribution = {1: 0, 2: 0, 3: 0}
        for rank in model_rank_positions[model_name]:
            rank_distribution[rank] += 1
        
        report_lines.append(f"• {model_name}")
        report_lines.append(f"  - Appearances in top-3: {count}/{len(validation_methods)} ({count/len(validation_methods)*100:.1f}%)")
        report_lines.append(f"  - Average rank: {avg_rank:.2f}")
        report_lines.append(f"  - Rank distribution: R1:{rank_distribution[1]}, R2:{rank_distribution[2]}, R3:{rank_distribution[3]}")
        report_lines.append("")
    
    # 검증 방식별 성능 일치도 분석
    report_lines.append("📈 VALIDATION-TEST CONSISTENCY ANALYSIS")
    report_lines.append("-" * 50)
    
    for validation_type, ranks in top_models.items():
        if validation_type == 'real_validation':
            display_label = 'Real Anomaly'
        elif validation_type.startswith('synthetic_'):
            anomaly_type = validation_type.replace('synthetic_', '').replace('_validation', '')
            display_label = f'Synthetic {anomaly_type.capitalize()}'
        else:
            display_label = validation_type
        
        differences = []
        for rank_key, model_info in ranks.items():
            diff = abs(model_info['val_auc'] - model_info['test_auc'])
            differences.append(diff)
        
        if differences:
            avg_diff = np.mean(differences)
            max_diff = np.max(differences)
            min_diff = np.min(differences)
            
            report_lines.append(f"• {display_label}")
            report_lines.append(f"  - Average val-test AUC difference: {avg_diff:.4f}")
            report_lines.append(f"  - Maximum difference: {max_diff:.4f}")
            report_lines.append(f"  - Minimum difference: {min_diff:.4f}")
            
            if avg_diff < 0.05:
                assessment = "🟢 Excellent reliability"
            elif avg_diff < 0.10:
                assessment = "🟡 Good reliability"
            else:
                assessment = "🔴 Poor reliability"
            
            report_lines.append(f"  - Reliability assessment: {assessment}")
            report_lines.append("")
    
    # 권장사항
    report_lines.append("💡 RECOMMENDATIONS")
    report_lines.append("-" * 50)
    
    # 가장 자주 등장하는 모델 추천
    most_frequent_model = sorted_models[0][0]
    most_frequent_count = sorted_models[0][1]
    
    report_lines.append(f"1. 🏅 Most Consistent Performer: {most_frequent_model}")
    report_lines.append(f"   - Appeared in top-3 across {most_frequent_count}/{len(validation_methods)} validation methods")
    report_lines.append(f"   - Average rank: {np.mean(model_rank_positions[most_frequent_model]):.2f}")
    report_lines.append("")
    
    # 가장 신뢰할 만한 검증 방식 추천
    validation_reliability = {}
    for validation_type, ranks in top_models.items():
        differences = [abs(model_info['val_auc'] - model_info['test_auc']) 
                      for model_info in ranks.values()]
        validation_reliability[validation_type] = np.mean(differences)
    
    most_reliable_validation = min(validation_reliability.items(), key=lambda x: x[1])
    
    if most_reliable_validation[0] == 'real_validation':
        reliable_label = 'Real Anomaly Validation'
    elif most_reliable_validation[0].startswith('synthetic_'):
        anomaly_type = most_reliable_validation[0].replace('synthetic_', '').replace('_validation', '')
        reliable_label = f'Synthetic {anomaly_type.capitalize()} Validation'
    else:
        reliable_label = most_reliable_validation[0]
    
    report_lines.append(f"2. 🎯 Most Reliable Validation Method: {reliable_label}")
    report_lines.append(f"   - Average val-test AUC difference: {most_reliable_validation[1]:.4f}")
    report_lines.append("")
    
    report_lines.append("3. 📋 General Guidelines:")
    report_lines.append("   - Consider ensemble methods using top-3 models for robust performance")
    report_lines.append("   - Monitor validation-test consistency when selecting final model")
    report_lines.append("   - Use multiple validation strategies for comprehensive evaluation")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    
    # 리포트 저장
    report_content = "\n".join(report_lines)
    report_filename = os.path.join(results_dir, 'top3_analysis_summary_report.txt')
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Top-3 분석 요약 리포트가 {report_filename}에 저장되었습니다")
    
    # 리포트 내용을 콘솔에도 출력
    print("\n" + report_content)