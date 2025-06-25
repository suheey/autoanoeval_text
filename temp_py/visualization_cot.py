import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_tsne(X, y, y_types=None, title="t-SNE Visualization", filename="tsne_visualization.png", anomaly_types=None, max_samples=10000):
    """
    t-SNE 시각화 함수
    
    Parameters:
    - X: 특성 데이터
    - y: 이진 레이블 (0: 정상, 1: 이상)
    - y_types: 이상치 유형 레이블 (0: 정상, 1+: 이상치 유형)
    - title: 시각화 제목
    - filename: 저장할 파일 이름
    - anomaly_types: 이상치 유형 이름 목록
    """
    # ✅ 샘플 수 제한
    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]
        y = y[idx]
        if y_types is not None:
            y_types = y_types[idx]
    
    # t-SNE 변환
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    # 1. 이상치 유형별 시각화 (y_types가 제공된 경우)
    if y_types is not None and anomaly_types is not None:
        plt.figure(figsize=(14, 10))
        
        # 색상 및 마커 설정 - 더 많은 색상 추가
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
        markers = ['o', '^', 's', 'D', '*', 'P', 'X', 'h', 'v', '<', '>', '8']
        labels = ['Normal'] + [f"{t.capitalize()} Anomaly" for t in anomaly_types]
        
        # 정상 데이터 그리기
        plt.scatter(
            X_tsne[y_types == 0, 0], X_tsne[y_types == 0, 1],
            c=colors[0], label=labels[0], alpha=0.5, s=10, marker=markers[0]
        )
        
        # 각 이상치 유형 그리기 - 안전한 인덱싱
        for i, anomaly_type in enumerate(anomaly_types, 1):
            if np.sum(y_types == i) > 0:  # 해당 유형의 이상치가 있는 경우에만
                # 색상과 마커 인덱스를 안전하게 계산
                color_idx = i % len(colors)  # 색상 배열 크기로 나눈 나머지
                marker_idx = min(i, len(markers) - 1)  # 마커 배열 크기를 초과하지 않도록
                
                plt.scatter(
                    X_tsne[y_types == i, 0], X_tsne[y_types == i, 1],
                    c=colors[color_idx], label=labels[i], alpha=0.8, s=20, marker=markers[marker_idx]
                )
        
        plt.title(title, fontsize=16)
        plt.legend(fontsize=12, loc='best')  # 범례 위치 자동 조정
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 시각화 저장
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"이상치 유형별 시각화가 {filename}에 저장되었습니다")
    
    # 2. 정상/이상치 구분만 보여주는 시각화
    plt.figure(figsize=(14, 10))
    
    plt.scatter(
        X_tsne[y == 0, 0], X_tsne[y == 0, 1],
        c='blue', label='Normal', alpha=0.5, s=10
    )
    plt.scatter(
        X_tsne[y == 1, 0], X_tsne[y == 1, 1],
        c='red', label='Anomaly', alpha=0.8, s=20
    )
    
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 이진 시각화 저장
    binary_filename = filename.replace('.png', '_binary.png')
    plt.savefig(binary_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"이진 이상치 시각화가 {binary_filename}에 저장되었습니다")