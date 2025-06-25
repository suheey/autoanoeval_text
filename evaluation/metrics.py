import numpy as np
from scipy.stats import spearmanr

def calculate_fdr(y_true, y_pred):
    """False Discovery Rate 계산: FDR = FP / (FP + TP)"""
    if np.sum(y_pred == 1) == 0:
        return 0.0
    
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    
    return fp / (fp + tp)

def get_best_model_name(results):
    """검증 AUC 기준 최고 성능 모델명 반환"""
    return max(results.items(), 
               key=lambda x: x[1]['val_auc'] if not np.isnan(x[1]['val_auc']) else -float('inf'))[0]

def calculate_mse_best_model(real_results, synthetic_results):
    """GT best model vs Synthetic best model의 Val AUC vs Test AUC MSE 계산"""
    real_best = get_best_model_name(real_results)
    synthetic_best = get_best_model_name(synthetic_results)
    
    # Validation AUC vs Test AUC 차이의 MSE
    real_val_auc = real_results[real_best]['val_auc']
    real_test_auc = real_results[real_best]['test_auc']
    synthetic_val_auc = synthetic_results[synthetic_best]['val_auc']
    synthetic_test_auc = synthetic_results[synthetic_best]['test_auc']
    
    real_diff = real_val_auc - real_test_auc
    synthetic_diff = synthetic_val_auc - synthetic_test_auc
    
    mse = (real_diff - synthetic_diff) ** 2
    
    return {
        'mse': mse,
        'real_best_model': real_best,
        'synthetic_best_model': synthetic_best,
        'real_val_auc': real_val_auc,
        'real_test_auc': real_test_auc,
        'synthetic_val_auc': synthetic_val_auc,
        'synthetic_test_auc': synthetic_test_auc
    }

def calculate_rank_correlation(real_results, synthetic_results):
    """Validation AUC 기준 순위 간 Spearman 상관계수 계산"""
    models = list(real_results.keys())
    
    # validation AUC로 순위 매기기 (실제 model selection 기준)
    real_val_aucs = [real_results[model]['val_auc'] for model in models]
    synthetic_val_aucs = [synthetic_results[model]['val_auc'] for model in models]
    
    correlation, p_value = spearmanr(real_val_aucs, synthetic_val_aucs)
    
    return correlation, p_value

def calculate_top_k_overlap(real_results, synthetic_results, k):
    """Validation AUC 기준 Top-K 모델 겹침 비율 계산"""
    if len(real_results) < k:
        return np.nan, set(), set()
    
    # Validation AUC로 Top-K 모델 추출 (실제 model selection 기준)
    real_sorted = sorted(real_results.items(), key=lambda x: x[1]['val_auc'], reverse=True)
    synthetic_sorted = sorted(synthetic_results.items(), key=lambda x: x[1]['val_auc'], reverse=True)
    
    real_top_k = set([model for model, _ in real_sorted[:k]])
    synthetic_top_k = set([model for model, _ in synthetic_sorted[:k]])
    
    overlap_ratio = len(real_top_k.intersection(synthetic_top_k)) / k
    
    return overlap_ratio, real_top_k, synthetic_top_k

def calculate_pairwise_win_rate(real_results, synthetic_results):
    """Validation AUC 기준 모델 쌍별 순위 비교 정확도 계산"""
    models = list(real_results.keys())
    total_pairs = 0
    correct_pairs = 0
    
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model_a, model_b = models[i], models[j]
            
            # validation AUC에서의 순위 일치 여부 (실제 model selection 기준)
            real_a_better = real_results[model_a]['val_auc'] > real_results[model_b]['val_auc']
            synthetic_a_better = synthetic_results[model_a]['val_auc'] > synthetic_results[model_b]['val_auc']
            
            total_pairs += 1
            if real_a_better == synthetic_a_better:
                correct_pairs += 1
    
    return correct_pairs / total_pairs if total_pairs > 0 else 0.0

def calculate_evaluation_metrics(real_results, synthetic_results):
    """
    모든 평가 메트릭을 계산하여 반환
    
    Returns:
        dict: 모든 평가 메트릭 결과
    """
    # MSE 계산 (validation AUC vs test AUC 차이)
    mse_result = calculate_mse_best_model(real_results, synthetic_results)
    
    # Rank Correlation 계산 (validation AUC 기준)
    rank_corr, p_value = calculate_rank_correlation(real_results, synthetic_results)
    
    # Top-K Overlap 계산 (validation AUC 기준)
    top1_overlap, _, _ = calculate_top_k_overlap(real_results, synthetic_results, 1)
    top3_overlap, real_top3, synthetic_top3 = calculate_top_k_overlap(real_results, synthetic_results, 3)
    top5_overlap, real_top5, synthetic_top5 = calculate_top_k_overlap(real_results, synthetic_results, 5)
    
    # Pairwise Win Rate 계산 (validation AUC 기준)
    pairwise_win_rate = calculate_pairwise_win_rate(real_results, synthetic_results)
    
    return {
        # MSE 관련
        'mse_best_model': mse_result['mse'],
        'real_best_model': mse_result['real_best_model'],
        'synthetic_best_model': mse_result['synthetic_best_model'],
        'real_val_auc': mse_result['real_val_auc'],
        'real_test_auc': mse_result['real_test_auc'],
        'synthetic_val_auc': mse_result['synthetic_val_auc'],
        'synthetic_test_auc': mse_result['synthetic_test_auc'],
        
        # 순위 상관관계
        'rank_correlation': rank_corr,
        'rank_correlation_pvalue': p_value,
        
        # Top-K Overlap
        'top1_overlap': top1_overlap,
        'top3_overlap': top3_overlap,
        'top5_overlap': top5_overlap,
        'real_top3': real_top3,
        'synthetic_top3': synthetic_top3,
        'real_top5': real_top5,
        'synthetic_top5': synthetic_top5,
        
        # 쌍별 비교
        'pairwise_win_rate': pairwise_win_rate
    }