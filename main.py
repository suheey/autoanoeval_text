import numpy as np
import pandas as pd
import os
import time
import argparse
import sys
from datetime import datetime

# 새로운 모듈 imports
from config.settings import RANDOM_SEED, ANOMALY_TYPES
from data.loader import load_dataset
from data.preprocessor import prepare_dataset_splits
from generators.statistical_generator import StatisticalAnomalyGenerator
from generators.llm_generator import LLMAnomalyGenerator
from models.selection import run_model_selection_experiment
from utils.io import setup_logging

def generate_validation_sets(X_original, y_original, X_normal_val, X_anomaly_val, 
                           feature_names=None, dataset_name="Unknown", 
                           openai_api_key=None, use_statistical_fallback=False):
    """LLM 및 통계적 합성 이상치 검증 세트 통합 생성"""
    print(f"\n🧪 합성 이상치 검증 세트 생성...")
    
    all_val_sets = {}
    
    # 1. LLM 기반 이상치 생성 시도
    if openai_api_key:
        print(f"\n🤖 LLM 기반 이상치 패턴 분석 및 생성...")
        try:
            llm_generator = LLMAnomalyGenerator(api_key=openai_api_key)
            
            synthetic_anomalies = llm_generator.generate_anomalies(
                X=X_original,
                y=y_original,
                anomaly_count=len(X_anomaly_val),
                feature_names=feature_names,
                dataset_name=dataset_name
            )
            
            if len(synthetic_anomalies) > 0:
                # 검증 세트 구성
                X_val_synthetic = np.vstack([X_normal_val, synthetic_anomalies])
                y_val_synthetic = np.concatenate([
                    np.zeros(len(X_normal_val)), 
                    np.ones(len(synthetic_anomalies))
                ])
                
                # 데이터 셔플
                idx = np.random.RandomState(RANDOM_SEED).permutation(len(y_val_synthetic))
                X_val_synthetic, y_val_synthetic = X_val_synthetic[idx], y_val_synthetic[idx]
                
                all_val_sets['llm_patterns'] = (X_val_synthetic, y_val_synthetic)
                
                print(f"✅ LLM 기반 검증 세트 생성 완료: {X_val_synthetic.shape}")
                print(f"   정상: {np.sum(y_val_synthetic == 0):,}, 이상: {np.sum(y_val_synthetic == 1):,}")
            else:
                print("❌ LLM 이상치 생성 실패")
                
        except Exception as e:
            print(f"❌ LLM 생성 중 오류: {e}")
    else:
        print("⚠️ OpenAI API 키가 없어 LLM 생성을 건너뜁니다.")
    
    # 2. 통계적 방법 (LLM 실패 시 또는 fallback 옵션 활성화 시)
    if not all_val_sets or use_statistical_fallback:
        if not all_val_sets:
            print("\n🔄 LLM 생성이 실패하여 통계적 방법으로 폴백...")
        else:
            print("\n📊 통계적 방법도 함께 생성...")
        
        data_generator = StatisticalAnomalyGenerator(seed=RANDOM_SEED)
        
        for anomaly_type in ANOMALY_TYPES:
            print(f"   🔬 {anomaly_type} 유형 생성 중...")
            
            try:
                synthetic_anomalies = data_generator.generate_anomalies(
                    X=X_original,
                    y=y_original,
                    anomaly_type=anomaly_type,
                    alpha=5,
                    percentage=0.2,
                    anomaly_count=len(X_anomaly_val)
                )
                
                # 검증 세트 구성
                X_val_synthetic = np.vstack([X_normal_val, synthetic_anomalies])
                y_val_synthetic = np.concatenate([np.zeros(len(X_normal_val)), np.ones(len(synthetic_anomalies))])
                
                # 데이터 셔플
                idx = np.random.RandomState(RANDOM_SEED).permutation(len(y_val_synthetic))
                X_val_synthetic, y_val_synthetic = X_val_synthetic[idx], y_val_synthetic[idx]
                
                all_val_sets[anomaly_type] = (X_val_synthetic, y_val_synthetic)
                
                print(f"      ✅ {anomaly_type}: {X_val_synthetic.shape} "
                      f"(정상: {np.sum(y_val_synthetic == 0):,}, 이상: {np.sum(y_val_synthetic == 1):,})")
                
            except Exception as e:
                print(f"      ❌ {anomaly_type} 생성 실패: {e}")
    
    return all_val_sets

def main(args):
    """메인 실험 실행 함수"""
    print("🔬 LLM 기반 이상치 패턴 분석 실험 시작")
    print("=" * 80)
    
    # 결과 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    llm_suffix = "_llm" if args.openai_api_key else ""
    results_dir = f"./results/{args.dataset_name}_experiment_results_{timestamp}{llm_suffix}"
    os.makedirs(results_dir, exist_ok=True)

    # 로그 파일 설정
    log_file = setup_logging(results_dir)
    
    try:
        # 1. 데이터셋 준비 (CSV 로드)
        print(f"\n📊 데이터셋 로드: {args.dataset_name}")
        X_original, y_original = load_dataset(args.dataset_name)
        
        # 2. 데이터셋 분할
        experiment_start_time = time.time()
        
        X_normal_train, X_normal_val, X_val_real, y_val_real, X_test, y_test, X_anomaly_val = prepare_dataset_splits(
            X_original, y_original
        )
        
        # 특성 이름 생성 (실제 데이터에 맞게 수정 필요)
        feature_names = [f"Feature_{i}" for i in range(X_original.shape[1])]
        
        # 3. 통합 검증 세트 생성 (LLM + 통계적 방법)
        synthetic_val_sets = generate_validation_sets(
            X_original=X_original, 
            y_original=y_original, 
            X_normal_val=X_normal_val, 
            X_anomaly_val=X_anomaly_val,
            feature_names=feature_names,
            dataset_name=args.dataset_name,
            openai_api_key=args.openai_api_key,
            use_statistical_fallback=args.use_statistical_fallback
        )
        
        # 생성된 검증 세트 확인
        print(f"\n📋 생성된 검증 세트: {list(synthetic_val_sets.keys())}")
        
        if not synthetic_val_sets:
            print("❌ 검증 세트가 하나도 생성되지 않았습니다.")
            return
        
        # LLM 패턴 포함 여부 확인
        has_llm = 'llm_patterns' in synthetic_val_sets
        has_statistical = any(key in ANOMALY_TYPES for key in synthetic_val_sets.keys())
        
        print(f"🤖 LLM 패턴: {'포함' if has_llm else '없음'}")
        print(f"📊 통계적 방법: {'포함' if has_statistical else '없음'}")
        
        # 4. 모델 선택 실험 실행
        try:
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
            
            # 성능 요약
            print(f"\n⏱️ 실험 시간 요약:")
            print(f"   데이터 준비: {model_start_time - experiment_start_time:.2f}s")
            print(f"   모델 학습/평가: {model_time:.2f}s")
            print(f"   전체 실험 시간: {total_time:.2f}s")
            
            # LLM vs 통계적 방법 성능 비교
            if has_llm and has_statistical:
                print(f"\n🎯 LLM vs 통계적 방법 성능 비교:")
                
                # Best models 비교
                if 'llm_patterns' in best_models:
                    llm_auc = best_models['llm_patterns']['test_auc']
                    print(f"   🤖 LLM 패턴 Best Model: {best_models['llm_patterns']['model_name']} (Test AUC: {llm_auc:.4f})")
                
                stat_aucs = []
                for key in best_models.keys():
                    if key in ANOMALY_TYPES:
                        stat_aucs.append(best_models[key]['test_auc'])
                        print(f"   📊 {key}: {best_models[key]['model_name']} (Test AUC: {best_models[key]['test_auc']:.4f})")
                
                if stat_aucs and 'llm_patterns' in best_models:
                    avg_stat_auc = np.mean(stat_aucs)
                    print(f"   📈 통계적 방법 평균 AUC: {avg_stat_auc:.4f}")
                    print(f"   🔍 LLM vs 통계적 방법 차이: {llm_auc - avg_stat_auc:+.4f}")
            
        except ImportError:
            print(f"⚠️ model_selection_enhanced 모듈이 없습니다. 패턴 분석까지만 수행됩니다.")
        except Exception as e:
            print(f"❌ 모델 선택 실험 실패: {e}")
        
        print(f"\n🎉 실험 성공적으로 완료!")
        print(f"📁 결과 위치: {results_dir}")
        
        # 최종 요약
        if has_llm:
            print(f"🤖 LLM 패턴 기반 이상치 생성이 포함되었습니다!")
        if has_statistical:
            print(f"📊 통계적 방법 기반 이상치도 함께 비교되었습니다!")
        
    except Exception as e:
        print(f"\n❌ 실험 중 오류 발생: {e}")
        raise
    finally:
        log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 기반 이상치 패턴 분석 실험")
    parser.add_argument("--dataset_name", type=str, required=True, 
                       help="데이터셋 이름")
    parser.add_argument("--openai_api_key", type=str, default=None,
                       help="OpenAI API 키 (없으면 통계적 방법만 사용)")
    parser.add_argument("--use_statistical_fallback", action="store_true", default=False,
                       help="LLM과 함께 통계적 방법도 사용")
    
    args = parser.parse_args()
    main(args)