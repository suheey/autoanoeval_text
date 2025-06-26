import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

def plot_tsne_anomaly_comparison(X_normal_val, synthetic_val_sets, X_anomaly_val=None, 
                                feature_names=None, results_dir="./results", 
                                perplexity=30, n_iter=1000, random_state=42):
    """
    Normal 데이터와 생성된 anomaly들의 t-SNE 시각화
    
    Args:
        X_normal_val: Normal 검증 데이터
        synthetic_val_sets: 생성된 synthetic anomaly 딕셔너리
        X_anomaly_val: 실제 anomaly 데이터 (있는 경우)
        feature_names: 특성 이름들
        results_dir: 결과 저장 디렉토리
        perplexity: t-SNE perplexity 파라미터
        n_iter: t-SNE 반복 횟수
        random_state: 랜덤 시드
    """
    print(f"\n🔍 t-SNE 시각화 생성 중...")
    
    # 데이터 준비
    all_data = []
    all_labels = []
    all_colors = []
    all_markers = []
    legend_labels = []
    
    # Normal 데이터 추가
    all_data.append(X_normal_val)
    all_labels.extend(['Normal'] * len(X_normal_val))
    all_colors.extend(['lightblue'] * len(X_normal_val))
    all_markers.extend(['o'] * len(X_normal_val))
    
    # 실제 anomaly 데이터 추가 (있는 경우)
    if X_anomaly_val is not None and len(X_anomaly_val) > 0:
        all_data.append(X_anomaly_val)
        all_labels.extend(['Real Anomaly'] * len(X_anomaly_val))
        all_colors.extend(['red'] * len(X_anomaly_val))
        all_markers.extend(['X'] * len(X_anomaly_val))
    
    # Synthetic anomaly들 추가
    color_map = {
        'llm_patterns': 'darkviolet',
        'local': 'blue',
        'cluster': 'green', 
        'global': 'purple',
        'discrepancy': 'orange',
        'contextual': 'brown'
    }
    
    marker_map = {
        'llm_patterns': 'D',
        'local': '^',
        'cluster': 's', 
        'global': 'P',
        'discrepancy': '*',
        'contextual': 'v'
    }
    
    for anomaly_type, (X_val_syn, y_val_syn) in synthetic_val_sets.items():
        # Synthetic anomaly만 추출 (y_val_syn == 1)
        anomaly_indices = y_val_syn == 1
        if np.sum(anomaly_indices) > 0:
            synthetic_anomalies = X_val_syn[anomaly_indices]
            all_data.append(synthetic_anomalies)
            
            # 라벨 및 색상 설정
            if anomaly_type == 'llm_patterns':
                label = 'LLM Patterns'
                color = color_map['llm_patterns']
                marker = marker_map['llm_patterns']
            else:
                # synthetic_xxx_validation -> xxx
                clean_type = anomaly_type.replace('synthetic_', '').replace('_validation', '')
                label = f'Synthetic {clean_type.capitalize()}'
                color = color_map.get(clean_type, 'gray')
                marker = marker_map.get(clean_type, 'o')
            
            all_labels.extend([label] * len(synthetic_anomalies))
            all_colors.extend([color] * len(synthetic_anomalies))
            all_markers.extend([marker] * len(synthetic_anomalies))
    
    if len(all_data) < 2:
        print("⚠️ 시각화할 데이터가 충분하지 않습니다.")
        return
    
    # 모든 데이터 결합
    X_combined = np.vstack(all_data)
    print(f"📊 전체 데이터 크기: {X_combined.shape}")
    
    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # t-SNE 실행
    print(f"🔄 t-SNE 실행 중 (perplexity={perplexity}, n_iter={n_iter})...")
    
    # 샘플 수에 따라 perplexity 조정
    max_perplexity = min(perplexity, (len(X_combined) - 1) // 3)
    if max_perplexity < perplexity:
        print(f"⚠️ Perplexity를 {perplexity}에서 {max_perplexity}로 조정합니다.")
        perplexity = max_perplexity
    
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                   random_state=random_state, init='pca', learning_rate='auto')
        X_tsne = tsne.fit_transform(X_scaled)
        
        print(f"✅ t-SNE 완료 (최종 KL divergence: {tsne.kl_divergence_:.4f})")
        
    except Exception as e:
        print(f"❌ t-SNE 실행 실패: {e}")
        return
    
    # 시각화
    plt.figure(figsize=(14, 10))
    
    # 고유 라벨별로 그리기
    unique_labels = list(dict.fromkeys(all_labels))  # 순서 유지하면서 중복 제거
    
    for label in unique_labels:
        indices = [i for i, l in enumerate(all_labels) if l == label]
        
        if not indices:
            continue
            
        x_coords = X_tsne[indices, 0]
        y_coords = X_tsne[indices, 1]
        colors = [all_colors[i] for i in indices]
        markers = [all_markers[i] for i in indices]
        
        # 같은 라벨의 첫 번째 색상과 마커 사용
        color = colors[0]
        marker = markers[0]
        
        # 크기와 투명도 설정
        if label == 'Normal':
            alpha = 0.6
            size = 20
            zorder = 1
        elif label == 'Real Anomaly':
            alpha = 1.0
            size = 60
            zorder = 10
        elif label == 'LLM Patterns':
            alpha = 0.9
            size = 50
            zorder = 9
        else:
            alpha = 0.8
            size = 40
            zorder = 5
        
        plt.scatter(x_coords, y_coords, c=color, marker=marker, s=size, 
                   alpha=alpha, label=label, edgecolors='black', linewidths=0.5,
                   zorder=zorder)
    
    plt.title('t-SNE Visualization: Normal vs Synthetic Anomalies (Including LLM Patterns)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
    plt.ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
    
    # 범례 설정
    legend = plt.legend(fontsize=11, loc='upper right', framealpha=0.9, 
                       fancybox=True, shadow=True)
    legend.set_title('Data Types', prop={'weight': 'bold', 'size': 12})
    
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    
    # 저장
    filename = os.path.join(results_dir, 'tsne_anomaly_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 t-SNE 시각화가 {filename}에 저장되었습니다")
    
    # 통계 정보 출력
    print(f"\n📈 시각화 통계:")
    for label in unique_labels:
        count = sum(1 for l in all_labels if l == label)
        print(f"   • {label}: {count:,}개")

def create_detailed_tsne_plots(X_normal_val, synthetic_val_sets, X_anomaly_val=None, 
                              results_dir="./results", random_state=42):
    """
    개별 anomaly 타입별 상세 t-SNE 플롯들 생성
    """
    print(f"\n🔍 개별 anomaly 타입별 상세 t-SNE 플롯 생성...")
    
    if not synthetic_val_sets:
        print("⚠️ 생성된 synthetic anomaly가 없습니다.")
        return
    
    # 전체 비교 플롯
    plot_tsne_anomaly_comparison(X_normal_val, synthetic_val_sets, X_anomaly_val, 
                                results_dir=results_dir, random_state=random_state)
    
    # 개별 타입별 플롯 (Normal + 실제 anomaly + 하나의 synthetic type)
    for anomaly_type, (X_val_syn, y_val_syn) in synthetic_val_sets.items():
        print(f"📊 {anomaly_type} 개별 플롯 생성 중...")
        
        # 개별 타입만 포함한 딕셔너리 생성
        single_type_dict = {anomaly_type: (X_val_syn, y_val_syn)}
        
        # 파일명 설정
        if anomaly_type == 'llm_patterns':
            plot_name = 'tsne_llm_patterns_detailed.png'
            plot_title = 't-SNE: Normal vs LLM Generated Anomaly Patterns'
        else:
            clean_type = anomaly_type.replace('synthetic_', '').replace('_validation', '')
            plot_name = f'tsne_{clean_type}_detailed.png'
            plot_title = f't-SNE: Normal vs Synthetic {clean_type.capitalize()} Anomalies'
        
        # 개별 플롯 생성
        _plot_single_anomaly_type(X_normal_val, single_type_dict, X_anomaly_val,
                                 results_dir, plot_name, plot_title, random_state)

def _plot_single_anomaly_type(X_normal_val, single_anomaly_dict, X_anomaly_val, 
                             results_dir, filename, title, random_state):
    """단일 anomaly 타입에 대한 상세 플롯"""
    try:
        # 데이터 준비
        all_data = [X_normal_val]
        all_labels = ['Normal'] * len(X_normal_val)
        all_colors = ['lightblue'] * len(X_normal_val)
        all_markers = ['o'] * len(X_normal_val)
        
        # 실제 anomaly 추가
        if X_anomaly_val is not None and len(X_anomaly_val) > 0:
            all_data.append(X_anomaly_val)
            all_labels.extend(['Real Anomaly'] * len(X_anomaly_val))
            all_colors.extend(['red'] * len(X_anomaly_val))
            all_markers.extend(['X'] * len(X_anomaly_val))
        
        # Synthetic anomaly 추가
        anomaly_type, (X_val_syn, y_val_syn) = list(single_anomaly_dict.items())[0]
        anomaly_indices = y_val_syn == 1
        if np.sum(anomaly_indices) > 0:
            synthetic_anomalies = X_val_syn[anomaly_indices]
            all_data.append(synthetic_anomalies)
            
            if anomaly_type == 'llm_patterns':
                label = 'LLM Patterns'
                color = 'darkviolet'
                marker = 'D'
            else:
                clean_type = anomaly_type.replace('synthetic_', '').replace('_validation', '')
                label = f'Synthetic {clean_type.capitalize()}'
                color = 'purple'
                marker = '^'
            
            all_labels.extend([label] * len(synthetic_anomalies))
            all_colors.extend([color] * len(synthetic_anomalies))
            all_markers.extend([marker] * len(synthetic_anomalies))
        
        # 데이터 결합 및 t-SNE
        X_combined = np.vstack(all_data)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        
        perplexity = min(30, (len(X_combined) - 1) // 3)
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, 
                   random_state=random_state, init='pca', learning_rate='auto')
        X_tsne = tsne.fit_transform(X_scaled)
        
        # 플롯
        plt.figure(figsize=(12, 9))
        
        unique_labels = list(dict.fromkeys(all_labels))
        for label in unique_labels:
            indices = [i for i, l in enumerate(all_labels) if l == label]
            x_coords = X_tsne[indices, 0]
            y_coords = X_tsne[indices, 1]
            
            if label == 'Normal':
                alpha, size, zorder = 0.6, 20, 1
            elif label == 'Real Anomaly':
                alpha, size, zorder = 1.0, 60, 10
            else:
                alpha, size, zorder = 0.9, 50, 9
            
            color = all_colors[indices[0]]
            marker = all_markers[indices[0]]
            
            plt.scatter(x_coords, y_coords, c=color, marker=marker, s=size,
                       alpha=alpha, label=label, edgecolors='black', linewidths=0.5,
                       zorder=zorder)
        
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('t-SNE Component 1', fontsize=11, fontweight='bold')
        plt.ylabel('t-SNE Component 2', fontsize=11, fontweight='bold')
        plt.legend(fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.tight_layout()
        
        # 저장
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ {filename} 저장 완료")
        
    except Exception as e:
        print(f"   ❌ {filename} 생성 실패: {e}")