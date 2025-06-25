import numpy as np
import pandas as pd
import os

from model_evaluator import get_default_models, prepare_data, evaluate_models, get_best_model_info, filter_valid_results
from metrics_calculator import calculate_evaluation_metrics
from visualization_utils import create_experiment_visualizations

def run_model_selection_experiment(X_normal_train, X_val_real, y_val_real, 
                                   synthetic_val_sets, X_test, y_test, results_dir):
    """
    🔬 Synthetic Anomaly 기반 모델 선택의 실용성 검증 실험
    
    핵심 질문: Synthetic anomaly로 선택한 best model이 
              Real anomaly test에서도 좋은 성능을 보이는가?
    
    Returns:
        tuple: (all_results, best_models, evaluation_metrics)
    """
    print("\n" + "="*60)
    print("🔬 모델 선택 실험 실행")
    print("="*60)
    
    # 1. 실험 준비
    print(f"\n📊 데이터 준비 및 표준화...")
    X_normal_train_scaled, X_val_real_scaled, X_test_scaled, synthetic_val_sets_scaled = prepare_data(
        X_normal_train, X_val_real, X_test, synthetic_val_sets
    )
    
    models = get_default_models()
    print(f"✅ 평가 모델: {len(models)}개")
    print(f"📋 모델 목록: {list(models.keys())}")
    
    # 2. 실험 실행
    all_results = {}
    
    # 2-1. GT Real Anomaly Validation (기준선)
    print(f"\n🎯 GT Real Anomaly Validation (기준선)")
    real_results = evaluate_models(
        models, X_normal_train_scaled, X_val_real_scaled, y_val_real, X_test_scaled, y_test
    )
    all_results['real_validation'] = real_results
    _print_validation_summary('GT Real Anomaly', real_results)
    
    # 2-2. Synthetic Anomaly Validations
    print(f"\n🧪 Synthetic Anomaly Validations")
    for anomaly_type, (X_val_syn, y_val_syn) in synthetic_val_sets_scaled.items():
        print(f"\n--- {anomaly_type.capitalize()} Synthetic ---")
        
        synthetic_results = evaluate_models(
            models, X_normal_train_scaled, X_val_syn, y_val_syn, X_test_scaled, y_test
        )
        all_results[f'synthetic_{anomaly_type}_validation'] = synthetic_results
        _print_validation_summary(f'Synthetic {anomaly_type}', synthetic_results)
    
    # 3. 결과 분석
    print(f"\n📈 결과 분석 및 저장...")
    best_models, evaluation_metrics, summary_df = _analyze_and_save_results(all_results, results_dir)
    
    # 4. 시각화 생성 (수정된 함수 호출)
    print(f"Debug - best_models keys: {list(best_models.keys())}")
    print(f"Debug - evaluation_metrics keys: {list(evaluation_metrics.keys())}")
    create_experiment_visualizations(best_models, evaluation_metrics, summary_df, results_dir)
    
    # 5. 최종 리포트
    _generate_summary_report(best_models, evaluation_metrics, results_dir)
    _print_experiment_conclusion(evaluation_metrics)
    
    return all_results, best_models, evaluation_metrics

def _print_validation_summary(validation_name, results):
    """검증 결과 간단 요약 출력"""
    valid_results = filter_valid_results(results)
    if not valid_results:
        print(f"⚠️  {validation_name}: 유효한 결과 없음")
        return
    
    best_info = get_best_model_info(valid_results)
    if best_info:
        print(f"🏆 Best: {best_info['model_name']} (Test AUC: {best_info['test_auc']:.4f})")

def _analyze_and_save_results(all_results, results_dir):
    """결과 분석 및 파일 저장"""
    summary_data = []
    best_models = {}
    evaluation_metrics = {}
    
    real_results = filter_valid_results(all_results['real_validation'])
    
    for validation_type, results in all_results.items():
        valid_results = filter_valid_results(results)
        if not valid_results:
            continue
        
        # Best 모델 정보 수집
        best_info = get_best_model_info(valid_results)
        if best_info:
            best_models[validation_type] = best_info
        
        # 평가 메트릭 계산 (Real과 비교, synthetic만)
        if validation_type != 'real_validation' and real_results:
            metrics = calculate_evaluation_metrics(real_results, valid_results)
            evaluation_metrics[validation_type] = metrics
            
            # 간단한 결과 출력
            synthetic_type = validation_type.replace('synthetic_', '').replace('_validation', '')
            print(f"📊 {synthetic_type}: Corr={metrics['rank_correlation']:.3f}, Overlap={metrics['top3_overlap']:.3f}")
        
        # 전체 결과 데이터 수집
        for model_name, model_metrics in results.items():
            summary_data.append({
                'validation_type': validation_type,
                'model': model_name,
                'val_auc': model_metrics['val_auc'],
                'test_auc': model_metrics['test_auc'],
                'val_ap': model_metrics['val_ap'],
                'test_ap': model_metrics['test_ap'],
                'test_fdr': model_metrics['test_fdr'],
                'training_time': model_metrics['training_time']
            })
    
    # 결과 파일 저장
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(results_dir, 'model_selection_results.csv'), index=False)
    
    if evaluation_metrics:
        metrics_df = pd.DataFrame.from_dict(evaluation_metrics, orient='index')
        metrics_df.to_csv(os.path.join(results_dir, 'evaluation_metrics.csv'))
    
    print(f"💾 결과 파일 저장 완료")
    
    return best_models, evaluation_metrics, summary_df

def _generate_summary_report(best_models, evaluation_metrics, results_dir):
    """실험 결과 요약 리포트 생성"""
    report_path = os.path.join(results_dir, 'experiment_summary_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("🔬 Synthetic Anomaly 기반 모델 선택 실용성 검증 실험 결과\n")
        f.write("=" * 80 + "\n\n")
        
        # 실험 목적
        f.write("📋 실험 목적\n")
        f.write("-" * 40 + "\n")
        f.write("Synthetic anomaly validation이 real anomaly validation만큼\n")
        f.write("효과적으로 best model을 선택할 수 있는지 검증\n\n")
        
        # Best 모델 비교
        f.write("🏆 각 검증 방식별 Best Model (Test Set 성능)\n")
        f.write("-" * 40 + "\n")
        
        gt_auc = None
        if 'real_validation' in best_models:
            gt_info = best_models['real_validation']
            gt_auc = gt_info['test_auc']
            f.write(f"GT Real Anomaly: {gt_info['model_name']} (AUC: {gt_auc:.4f}) [기준선]\n")
        
        for val_type, info in best_models.items():
            if val_type != 'real_validation':
                auc_diff = info['test_auc'] - gt_auc if gt_auc else 0
                synthetic_type = val_type.replace('synthetic_', '').replace('_validation', '')
                f.write(f"Synthetic {synthetic_type}: {info['model_name']} "
                       f"(AUC: {info['test_auc']:.4f}, 차이: {auc_diff:+.4f})\n")
        
        # 핵심 평가 메트릭
        f.write(f"\n📊 핵심 평가 메트릭 (1.0에 가까울수록 GT와 유사)\n")
        f.write("-" * 40 + "\n")
        
        if evaluation_metrics:
            for val_type, metrics in evaluation_metrics.items():
                synthetic_type = val_type.replace('synthetic_', '').replace('_validation', '')
                f.write(f"\nSynthetic {synthetic_type}:\n")
                f.write(f"  - Rank Correlation: {metrics['rank_correlation']:.4f}\n")
                f.write(f"  - Top-3 Overlap: {metrics['top3_overlap']:.4f}\n")
                f.write(f"  - Pairwise Win Rate: {metrics['pairwise_win_rate']:.4f}\n")
                f.write(f"  - MSE (Best Model): {metrics['mse_best_model']:.6f}\n")
            
            # 최고 성능 찾기
            best_synthetic = max(evaluation_metrics.items(), 
                               key=lambda x: x[1]['rank_correlation'])
            best_type = best_synthetic[0].replace('synthetic_', '').replace('_validation', '')
            f.write(f"\n🥇 최고 성능: Synthetic {best_type} ")
            f.write(f"(Rank Correlation: {best_synthetic[1]['rank_correlation']:.4f})\n")
        
        # 결론
        f.write(f"\n💡 주요 발견사항\n")
        f.write("-" * 40 + "\n")
        if evaluation_metrics:
            avg_correlation = np.mean([m['rank_correlation'] for m in evaluation_metrics.values()])
            avg_overlap = np.mean([m['top3_overlap'] for m in evaluation_metrics.values() 
                                 if not np.isnan(m['top3_overlap'])])
            avg_win_rate = np.mean([m['pairwise_win_rate'] for m in evaluation_metrics.values()])
            
            f.write(f"- 평균 순위 상관관계: {avg_correlation:.4f}\n")
            f.write(f"- 평균 Top-3 일치율: {avg_overlap:.4f}\n")
            f.write(f"- 평균 쌍별 정확도: {avg_win_rate:.4f}\n\n")
            
            if avg_correlation >= 0.8:
                f.write("✅ Synthetic validation이 매우 효과적\n")
            elif avg_correlation >= 0.6:
                f.write("⚠️ Synthetic validation이 어느 정도 효과적\n")
            else:
                f.write("❌ Synthetic validation의 효과가 제한적\n")
        
        f.write(f"\n📈 상세 결과는 CSV 파일과 시각화를 참조하세요.\n")
    
    print(f"📋 실험 요약 리포트: {report_path}")

def _print_experiment_conclusion(evaluation_metrics):
    """실험 결론 출력"""
    if not evaluation_metrics:
        return
    
    print("\n" + "="*60)
    print("🎯 실험 결론")
    print("="*60)
    
    # 평균 성능 계산
    correlations = [m['rank_correlation'] for m in evaluation_metrics.values()]
    overlaps = [m['top3_overlap'] for m in evaluation_metrics.values() 
               if not np.isnan(m['top3_overlap'])]
    win_rates = [m['pairwise_win_rate'] for m in evaluation_metrics.values()]
    
    avg_corr = np.mean(correlations)
    avg_overlap = np.mean(overlaps) if overlaps else 0
    avg_win = np.mean(win_rates)
    
    print(f"📊 전체 평균 성능:")
    print(f"   • 순위 상관관계: {avg_corr:.4f}")
    print(f"   • Top-3 일치율: {avg_overlap:.4f}")
    print(f"   • 쌍별 정확도: {avg_win:.4f}")
    
    # 최고/최저 성능
    best_method = max(evaluation_metrics.items(), key=lambda x: x[1]['rank_correlation'])
    worst_method = min(evaluation_metrics.items(), key=lambda x: x[1]['rank_correlation'])
    
    best_type = best_method[0].replace('synthetic_', '').replace('_validation', '')
    worst_type = worst_method[0].replace('synthetic_', '').replace('_validation', '')
    
    print(f"\n🥇 최고: Synthetic {best_type} (상관관계: {best_method[1]['rank_correlation']:.4f})")
    print(f"🥉 최저: Synthetic {worst_type} (상관관계: {worst_method[1]['rank_correlation']:.4f})")
    
    # 실용성 평가
    if avg_corr >= 0.8:
        conclusion = "✅ 매우 실용적: Synthetic validation 완전 대체 가능"
    elif avg_corr >= 0.6:
        conclusion = "⚠️ 실용적: Synthetic validation 부분적 활용 가능"
    else:
        conclusion = "❌ 제한적: Synthetic validation 효과 낮음"
    
    print(f"\n💡 종합 평가: {conclusion}")
    print("="*60)