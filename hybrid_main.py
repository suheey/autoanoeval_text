import numpy as np
import pandas as pd
import os
import time
import argparse
import sys
from datetime import datetime

# 새로운 모듈 imports
from config.settings import RANDOM_SEED, ANOMALY_TYPES, OPENAI_API_KEY, GEMINI_API_KEY
from data.loader import load_dataset
from data.preprocessor import prepare_dataset_splits
from generators.validation_set_generator import generate_validation_sets
from models.selection import run_model_selection_experiment
from utils.io import setup_logging

def main(args):
    """메인 실험 실행 함수 (하이브리드 LLM 지원)"""
    print("🔬 LLM 기반 이상치 패턴 분석 실험 시작")
    print("=" * 80)
    
    # LLM 모드 확인
    if args.hybrid_llm_mode:
        print("🔄 하이브리드 LLM 모드: 분석(수동) + 생성(자동)")
        if args.llm_step:
            print(f"📋 현재 단계: {args.llm_step}")
    
    # 결과 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.hybrid_llm_mode:
        llm_suffix = "_hybrid_llm"
    else:
        llm_suffix = "_llm" if OPENAI_API_KEY else ""
    
    results_dir = f"./results/{args.dataset_name}_experiment_results_{timestamp}{llm_suffix}"
    os.makedirs(results_dir, exist_ok=True)

    # 로그 파일 설정
    log_file = setup_logging(results_dir)
    
    try:
        # 1. 데이터셋 준비
        print(f"\n📊 데이터셋 로드: {args.dataset_name}")
        
        # 카테고리 인코딩 방식 선택
        cat_encoding = getattr(args, 'cat_encoding', 'int')
        
        X_original, y_original, metadata = load_dataset(
            dataset_name=args.dataset_name,
            cat_encoding=cat_encoding
        )
        
        # 2. 데이터셋 분할
        experiment_start_time = time.time()
        
        X_normal_train, X_normal_val, X_val_real, y_val_real, X_test, y_test, X_anomaly_val = prepare_dataset_splits(
            X_original, y_original, metadata
        )
        
        # 특성 이름 생성
        feature_names = metadata.get('column_names', [f"Feature_{i}" for i in range(X_original.shape[1])])
        
        # 3. 통합 검증 세트 생성 (하이브리드 LLM 지원)
        synthetic_val_sets = generate_validation_sets(
            X_original=X_original, 
            y_original=y_original, 
            X_normal_val=X_normal_val, 
            X_anomaly_val=X_anomaly_val,
            feature_names=feature_names,
            dataset_name=args.dataset_name,
            llm_generate=args.llm_generate,
            openai_api_key=OPENAI_API_KEY,
            use_statistical_fallback=args.use_statistical_fallback,
            num_anomaly_conditions=args.num_anomaly_conditions,
            results_dir=results_dir,
            hybrid_llm_mode=args.hybrid_llm_mode,    # 하이브리드 LLM 모드 (새로 추가)
            llm_step=args.llm_step                   # LLM 단계
        )
        
        # 하이브리드 모드에서 수동 개입이 필요한 경우
        if args.hybrid_llm_mode and len(synthetic_val_sets) == 0:
            print(f"\n⏸️ 하이브리드 모드 - 수동 개입 완료 후 다음 명령어로 자동 진행:")
            if args.llm_step == "start" or args.llm_step is None:
                print(f"   python main.py --dataset_name {args.dataset_name} --llm_generate --hybrid_llm_mode --llm_step continue --use_statistical_fallback")
            return
        
        # 수동 모드에서 수동 개입이 필요한 경우
        if len(synthetic_val_sets) == 0:
            print(f"\n⏸️ 수동 개입 완료 후 다음 명령어로 계속 진행:")
            if args.llm_step == "start" or args.llm_step is None:
                print(f"   python main.py --dataset_name {args.dataset_name} --llm_generate --llm_step continue_analysis")
            elif args.llm_step == "continue_analysis":
                print(f"   python main.py --dataset_name {args.dataset_name} --llm_generate --llm_step continue_generation")
            elif args.llm_step == "continue_generation":
                print(f"   python main.py --dataset_name {args.dataset_name} --llm_generate --llm_step auto")
            return
        
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
        print(f"🔍 t-SNE 시각화: 생성 완료")
        
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
                
                if 'llm_patterns' in best_models:
                    llm_auc = best_models['llm_patterns']['test_auc']
                    if args.hybrid_llm_mode:
                        mode_text = "하이브리드 LLM"
                    else:
                        mode_text = "API LLM"
                    print(f"   🤖 {mode_text} 패턴 Best Model: {best_models['llm_patterns']['model_name']} (Test AUC: {llm_auc:.4f})")
                
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
            if args.hybrid_llm_mode:
                mode_text = "하이브리드 LLM (분석: 수동, 생성: 자동)"
            else:
                mode_text = "API LLM"
            print(f"🤖 {mode_text} 패턴 기반 이상치 생성이 포함되었습니다!")
        if has_statistical:
            print(f"📊 통계적 방법 기반 이상치도 함께 비교되었습니다!")
        print(f"🔍 t-SNE 시각화를 통해 생성된 anomaly들의 분포를 확인할 수 있습니다!")
        
        if args.hybrid_llm_mode:
            print(f"🔄 하이브리드 모드로 분석은 무료(웹), 생성은 자동(API)으로 처리했습니다!")
        
    except Exception as e:
        print(f"\n❌ 실험 중 오류 발생: {e}")
        raise
    finally:
        log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 기반 이상치 패턴 분석 실험 (하이브리드 LLM 지원)")
    parser.add_argument("--dataset_name", type=str, required=True, 
                       help="데이터셋 이름")
    parser.add_argument("--llm_generate", action="store_true", default=False,
                       help="LLM 기반 generation")
    parser.add_argument("--use_statistical_fallback", action="store_true", default=False,
                       help="LLM과 함께 통계적 방법도 사용")
    parser.add_argument("--num_anomaly_conditions", type=int, default=5,
                       help="LLM이 생성할 이상치 조건 개수")
    parser.add_argument("--cat_encoding", type=str, default="int", 
                       choices=["int", "onehot", "int_emb"],
                       help="카테고리 인코딩 방식")
    parser.add_argument("--scaling_type", type=str, default="standard",
                       choices=["standard", "minmax", "none"],
                       help="스케일링 방식")
    parser.add_argument("--hybrid_llm_mode", action="store_true", default=False,
                       help="하이브리드 LLM 모드: 분석(수동) + 생성(자동) (부분 비용 절약)")
    parser.add_argument("--llm_step", type=str, 
                       choices=["start", "continue", "continue_analysis", "continue_generation", "auto"],
                       help="LLM 단계 제어")
    
    args = parser.parse_args()
    
    # LLM 모드 검증
    if args.hybrid_llm_mode and not args.llm_generate:
        print("❌ 수동/하이브리드 LLM 모드를 사용하려면 --llm_generate도 함께 설정해야 합니다.")
        sys.exit(1)
    
    main(args)