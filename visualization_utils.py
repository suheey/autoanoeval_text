import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def convert_validation_labels(validation_types):
    """검증 타입을 표시용 레이블로 변환"""
    display_labels = []
    for vtype in validation_types:
        if vtype == 'real_validation':
            display_labels.append('GT Real Anomaly')
        elif vtype.startswith('synthetic_'):
            anomaly_type = vtype.replace('synthetic_', '').replace('_validation', '')
            display_labels.append(f'Synthetic {anomaly_type.capitalize()}')
        else:
            display_labels.append(vtype.replace('_', ' ').title())
    return display_labels

def plot_core_performance_metrics(evaluation_metrics, best_models, results_dir):
    """
    핵심 성능 메트릭들 시각화
    MSE: 5개 방법 모두 (val_auc vs test_auc 차이)
    나머지: synthetic 4개만 (real_validation과 비교)
    """
    if not evaluation_metrics and not best_models:
        print("⚠️ 평가 메트릭이 없어 시각화를 건너뜁니다.")
        return
    
    # 2x2 서브플롯
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # 1. MSE (Best Model) - 5개 방법 모두
    if best_models:
        validation_types_all = list(best_models.keys())
        display_labels_all = convert_validation_labels(validation_types_all)
        
        mse_values = []
        for vtype in validation_types_all:
            info = best_models[vtype]
            val_test_diff = info['val_auc'] - info['test_auc']
            mse_values.append(val_test_diff ** 2)  # MSE는 차이의 제곱
        
        bars = axes[0].bar(display_labels_all, mse_values, alpha=0.8, color='lightcoral', 
                          edgecolor='black', linewidth=0.5)
        axes[0].set_title('MSE (Val AUC - Test AUC)', fontsize=13, fontweight='bold', pad=15)
        axes[0].set_ylabel('MSE Score', fontsize=11)
        axes[0].tick_params(axis='x', rotation=45, labelsize=10)
        axes[0].grid(True, alpha=0.3, linestyle=':')
        
        # 바 위에 수치 표시
        for bar, value in zip(bars, mse_values):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + height * 0.05,
                        f'{value:.4f}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'No MSE Data', ha='center', va='center', 
                    transform=axes[0].transAxes, fontsize=12)
        axes[0].set_title('MSE (Val AUC - Test AUC)', fontsize=13, fontweight='bold')
    
    # 2-4. 나머지 메트릭들 - Synthetic 4개만
    if evaluation_metrics:
        validation_types_syn = list(evaluation_metrics.keys())
        display_labels_syn = convert_validation_labels(validation_types_syn)
        
        metrics_config = [
            {
                'name': 'Rank Correlation',
                'values': [evaluation_metrics[vtype]['rank_correlation'] for vtype in validation_types_syn],
                'color': 'lightgreen',
                'ylabel': 'Correlation',
                'ylim': (-0.1, 1.1)
            },
            {
                'name': 'Top-3 Overlap',
                'values': [evaluation_metrics[vtype]['top3_overlap'] for vtype in validation_types_syn],
                'color': 'gold',
                'ylabel': 'Overlap Ratio',
                'ylim': (0, 1.1)
            },
            {
                'name': 'Pairwise Win Rate',
                'values': [evaluation_metrics[vtype]['pairwise_win_rate'] for vtype in validation_types_syn],
                'color': 'mediumpurple',
                'ylabel': 'Win Rate',
                'ylim': (0, 1.1)
            }
        ]
        
        for idx, config in enumerate(metrics_config, 1):
            values = config['values']
            
            # NaN 값 체크
            valid_values = [v for v in values if not np.isnan(v)]
            if not valid_values:
                axes[idx].text(0.5, 0.5, 'No Valid Data', ha='center', va='center', 
                              transform=axes[idx].transAxes, fontsize=12)
                axes[idx].set_title(config['name'], fontsize=13, fontweight='bold')
                continue
            
            # 바 그래프
            bars = axes[idx].bar(display_labels_syn, values, alpha=0.8, color=config['color'], 
                                edgecolor='black', linewidth=0.5)
            
            # 축 설정
            axes[idx].set_title(config['name'], fontsize=13, fontweight='bold', pad=15)
            axes[idx].set_ylabel(config['ylabel'], fontsize=11)
            axes[idx].tick_params(axis='x', rotation=45, labelsize=10)
            axes[idx].grid(True, alpha=0.3, linestyle=':')
            axes[idx].set_ylim(config['ylim'])
            
            # 바 위에 수치 표시
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    height = bar.get_height()
                    y_offset = 0.05 * (axes[idx].get_ylim()[1] - axes[idx].get_ylim()[0])
                    axes[idx].text(bar.get_x() + bar.get_width()/2., height + y_offset,
                                  f'{value:.3f}', ha='center', va='bottom', 
                                  fontsize=9, fontweight='bold')
    else:
        for idx in range(1, 4):
            axes[idx].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                          transform=axes[idx].transAxes, fontsize=12)
    
    plt.suptitle('Synthetic Validation Performance Metrics', 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 저장
    filename = os.path.join(results_dir, 'core_performance_metrics.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 핵심 성능 메트릭 비교 시각화가 {filename}에 저장되었습니다")

def plot_best_model_test_performance(best_models, results_dir):
    """
    각 검증 방식별 최고 성능 모델의 Validation vs Test 성능 비교
    각 x축마다 2개의 막대: Validation AUC vs Test AUC (performance drop 확인용)
    """
    if not best_models:
        print("⚠️ Best 모델 정보가 없어 시각화를 건너뜁니다.")
        return

    # x축에 모든 validation 방법 포함
    validation_types = list(best_models.keys())
    display_labels = convert_validation_labels(validation_types)
    
    # 데이터 추출
    model_names = [info['model_name'] for info in best_models.values()]
    val_aucs = [info['val_auc'] for info in best_models.values()]
    test_aucs = [info['test_auc'] for info in best_models.values()]
    val_aps = [info['val_ap'] for info in best_models.values()]
    test_aps = [info['test_ap'] for info in best_models.values()]
    val_fdrs = [info.get('val_fdr', 0) for info in best_models.values()]  # Validation FDR 추가
    test_fdrs = [info.get('test_fdr', 0) for info in best_models.values()]
    
    # 1x3 서브플롯
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # x축 위치 설정 (각 방법마다 2개 막대)
    x = np.arange(len(validation_types))
    width = 0.35  # 막대 너비
    
    # AUC 비교 (Validation vs Test)
    bars1_val = axes[0].bar(x - width/2, val_aucs, width, label='Validation AUC', 
                           color='lightblue', alpha=0.8, edgecolor='black')
    bars1_test = axes[0].bar(x + width/2, test_aucs, width, label='Test AUC', 
                            color='orange', alpha=0.8, edgecolor='black')
    
    axes[0].set_title('🏆 Validation vs Test AUC (Best Models)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('AUC Score', fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display_labels, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, linestyle=':')
    
    # AP 비교 (Validation vs Test)
    bars2_val = axes[1].bar(x - width/2, val_aps, width, label='Validation AP', 
                           color='lightgreen', alpha=0.8, edgecolor='black')
    bars2_test = axes[1].bar(x + width/2, test_aps, width, label='Test AP', 
                            color='red', alpha=0.8, edgecolor='black')
    
    axes[1].set_title('🎯 Validation vs Test AP (Best Models)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('AP Score', fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display_labels, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, linestyle=':')
    
    # Test FDR (2개 막대: Validation FDR vs Test FDR)
    bars3_val = axes[2].bar(x - width/2, val_fdrs, width, label='Validation FDR', 
                           color='lightcoral', alpha=0.8, edgecolor='black')
    bars3_test = axes[2].bar(x + width/2, test_fdrs, width, label='Test FDR', 
                            color='darkred', alpha=0.8, edgecolor='black')
    
    axes[2].set_title('📉 Validation vs Test FDR (Best Models)', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('FDR', fontsize=11)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(display_labels, rotation=45, ha='right')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, linestyle=':')
    max_fdr = max(val_fdrs + test_fdrs) if max(val_fdrs + test_fdrs) > 0 else 0.1
    axes[2].set_ylim(0, max_fdr * 1.2)
    
    # 수치 표시
    # AUC 차트
    for i, (val_bar, test_bar, val_val, test_val, name) in enumerate(zip(bars1_val, bars1_test, val_aucs, test_aucs, model_names)):
        # Validation AUC 수치
        axes[0].text(val_bar.get_x() + val_bar.get_width()/2., val_bar.get_height() + 0.01,
                    f'{val_val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        # Test AUC 수치
        axes[0].text(test_bar.get_x() + test_bar.get_width()/2., test_bar.get_height() + 0.01,
                    f'{test_val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        # 모델명 (x축 하단)
        axes[0].text(i, min(val_aucs + test_aucs) - (max(val_aucs + test_aucs) - min(val_aucs + test_aucs)) * 0.1,
                    name, ha='center', va='top', fontsize=9, fontweight='bold', rotation=0)
    
    # AP 차트
    for i, (val_bar, test_bar, val_val, test_val) in enumerate(zip(bars2_val, bars2_test, val_aps, test_aps)):
        # Validation AP 수치
        axes[1].text(val_bar.get_x() + val_bar.get_width()/2., val_bar.get_height() + 0.01,
                    f'{val_val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        # Test AP 수치
        axes[1].text(test_bar.get_x() + test_bar.get_width()/2., test_bar.get_height() + 0.01,
                    f'{test_val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # FDR 차트
    for i, (val_bar, test_bar, val_val, test_val) in enumerate(zip(bars3_val, bars3_test, val_fdrs, test_fdrs)):
        # Validation FDR 수치
        axes[2].text(val_bar.get_x() + val_bar.get_width()/2., val_bar.get_height() + max_fdr * 0.02,
                    f'{val_val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        # Test FDR 수치
        axes[2].text(test_bar.get_x() + test_bar.get_width()/2., test_bar.get_height() + max_fdr * 0.02,
                    f'{test_val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        # 모델명 (x축 하단)
        axes[2].text(i, -max_fdr * 0.1,
                    model_names[i], ha='center', va='top', fontsize=9, fontweight='bold', rotation=0)
    
    plt.suptitle('🔍 Best Model Performance: Validation vs Test (Performance Drop Analysis)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 저장
    filename = os.path.join(results_dir, 'best_model_test_performance.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🏆 최고 모델 성능 비교 시각화 (Performance Drop)가 {filename}에 저장되었습니다")

def plot_validation_test_correlation(summary_df, results_dir):
    """
    검증 성능과 테스트 성능 간의 상관관계 분석 
    GT 강조, 모델 이름 표기
    """
    if len(summary_df) == 0:
        print("⚠️ 요약 데이터가 없어 상관관계 시각화를 건너뜁니다.")
        return
    
    validation_types = summary_df['validation_type'].unique()
    
    plt.figure(figsize=(14, 10))
    
    # 색상 및 마커 설정
    colors = {
        'real_validation': 'red', 
        'synthetic_local_validation': 'blue', 
        'synthetic_cluster_validation': 'green', 
        'synthetic_global_validation': 'purple', 
        'synthetic_discrepancy_validation': 'orange'
    }
    
    markers = {
        'real_validation': 'o', 
        'synthetic_local_validation': '^', 
        'synthetic_cluster_validation': 's', 
        'synthetic_global_validation': 'D', 
        'synthetic_discrepancy_validation': '*'
    }
    
    for val_type in validation_types:
        df_subset = summary_df[summary_df['validation_type'] == val_type]
        df_subset = df_subset.dropna(subset=['val_auc', 'test_auc'])
        
        if len(df_subset) > 0:
            color = colors.get(val_type, 'gray')
            marker = markers.get(val_type, 'o')
            
            # 레이블 및 스타일 설정
            if val_type == 'real_validation':
                label = 'GT Real Anomaly'
                alpha = 1.0
                size = 120
                zorder = 10
            else:
                anomaly_type = val_type.replace('synthetic_', '').replace('_validation', '')
                label = f'Synthetic {anomaly_type.capitalize()}'
                alpha = 0.7
                size = 80
                zorder = 5
            
            # 산점도
            scatter = plt.scatter(df_subset['val_auc'], df_subset['test_auc'],
                                alpha=alpha, label=label, color=color, marker=marker, s=size, 
                                edgecolors='black', linewidths=0.5, zorder=zorder)
            
            # 각 점에 모델 이름(PCA, IForest 등) 표기
            for _, row in df_subset.iterrows():
                plt.annotate(
                    row['model'],  # 모델명: PCA, IForest, KNN, LOF 등
                    (row['val_auc'], row['test_auc']),
                    xytext=(3, 3),  # 점에서 3픽셀 떨어진 위치 (더 가깝게)
                    textcoords='offset points',
                    fontsize=7,
                    fontweight='bold',
                    color='black',  # 검은색 텍스트로 가독성 향상
                    alpha=0.8,
                    bbox=dict(
                        boxstyle='round,pad=0.1', 
                        facecolor='white',      # 흰색 배경
                        alpha=0.7, 
                        edgecolor=color,        # validation type 색상으로 테두리
                        linewidth=0.5
                    ),
                    ha='left',
                    va='bottom'
                )
    
    # 완벽한 상관관계 대각선
    min_val = summary_df[['val_auc', 'test_auc']].min().min()
    max_val = summary_df[['val_auc', 'test_auc']].max().max()
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, 
             label='Perfect Correlation', linewidth=2)
    
    plt.xlabel('Validation AUC', fontsize=12, fontweight='bold')
    plt.ylabel('Test AUC', fontsize=12, fontweight='bold')
    plt.title('Validation vs Test Performance Correlation (with Model Names)', 
              fontsize=15, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    # 저장
    filename = os.path.join(results_dir, 'validation_test_correlation.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"검증-테스트 상관관계 그래프가 {filename}에 저장되었습니다")

def create_experiment_visualizations(best_models, evaluation_metrics, summary_df, results_dir):
    """
    실험의 모든 핵심 시각화를 생성
    
    Parameters:
    - best_models: 검증 방식별 최고 성능 모델 정보 
    - evaluation_metrics: 평가 메트릭 결과 
    - summary_df: 전체 실험 결과 DataFrame
    - results_dir: 결과 저장 디렉토리
    """
    print(f"\n🎨 핵심 시각화 생성 중... ")
    
    # 1. 핵심 성능 메트릭 비교
    if evaluation_metrics or best_models:
        plot_core_performance_metrics(evaluation_metrics, best_models, results_dir)
    else:
        print("⚠️ 평가 메트릭이 없어 핵심 메트릭 시각화를 건너뜁니다.")
    
    # 2. Best 모델 테스트 성능 비교 (기준선 없음)
    if best_models:
        plot_best_model_test_performance(best_models, results_dir)
    else:
        print("⚠️ Best 모델 정보가 없어 성능 비교 시각화를 건너뜁니다.")
    
    # 3. 검증-테스트 상관관계 (GT 강조, 모델명 표기)
    if len(summary_df) > 0:
        plot_validation_test_correlation(summary_df, results_dir)
    else:
        print("⚠️ 요약 데이터가 없어 상관관계 시각화를 건너뜁니다.")
    
    print(f"✅ 모든 핵심 시각화 완료!")
    print(f"📁 시각화 파일들이 {results_dir}에 저장되었습니다")
    print(f"📊 생성된 파일:")
    print(f"   - core_performance_metrics.png")
    print(f"   - best_model_test_performance.png")
    print(f"   - validation_test_correlation.png")