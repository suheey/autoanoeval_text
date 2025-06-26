import numpy as np
from config.settings import RANDOM_SEED, ANOMALY_TYPES
from .statistical_generator import StatisticalAnomalyGenerator
from .llm_generator import LLMAnomalyGenerator
from .hybrid_llm_generator import HybridLLMAnomalyGenerator

def generate_validation_sets(X_original, y_original, X_normal_val, X_anomaly_val, 
                           feature_names=None, dataset_name="Unknown", llm_generate=False,
                           openai_api_key=None, gemini_api_key=None, use_statistical_fallback=False, 
                           num_anomaly_conditions=5, results_dir="./results",
                        hybrid_llm_mode=False, llm_step=None):
    """LLM 및 통계적 합성 이상치 검증 세트 통합 생성 (하이브리드 LLM 지원)"""
    print(f"\n🧪 합성 이상치 검증 세트 생성...")
    
    all_val_sets = {}
    
    # 1. LLM 기반 이상치 생성 시도
    if llm_generate:
        if hybrid_llm_mode:
            print(f"\n🔄 하이브리드 LLM 모드: 분석(수동) + 생성(자동)")
            try:
                hybrid_generator = HybridLLMAnomalyGenerator(
                    api_key=openai_api_key,
                    seed=RANDOM_SEED,
                    num_anomaly_conditions=num_anomaly_conditions
                )
                
                prompts_dir = f"./prompts"
                
                # 단계별 처리
                if llm_step == "start" or llm_step is None:
                    # 1단계: 분석 프롬프트 생성 (수동)
                    synthetic_anomalies = hybrid_generator.generate_anomalies(
                        X=X_original,
                        y=y_original,
                        anomaly_count=len(X_anomaly_val),
                        feature_names=feature_names,
                        dataset_name=dataset_name,
                        save_path=prompts_dir,
                        hybrid_step="start"
                    )
                    
                elif llm_step == "continue":
                    # 2단계: 분석 완료 후 자동 생성 (API)
                    synthetic_anomalies = hybrid_generator.continue_with_auto_generation(
                        X=X_original,
                        y=y_original,
                        feature_names=feature_names,
                        dataset_name=dataset_name,
                        anomaly_count=len(X_anomaly_val),
                        save_path=prompts_dir
                    )
                else:
                    print(f"❌ 잘못된 llm_step 값: {llm_step}")
                    synthetic_anomalies = np.array([])
                
                # 검증 세트 구성 (이상치가 실제로 생성된 경우만)
                if len(synthetic_anomalies) > 0:
                    X_val_synthetic = np.vstack([X_normal_val, synthetic_anomalies])
                    y_val_synthetic = np.concatenate([
                        np.zeros(len(X_normal_val)), 
                        np.ones(len(synthetic_anomalies))
                    ])
                    
                    # 데이터 셔플
                    idx = np.random.RandomState(RANDOM_SEED).permutation(len(y_val_synthetic))
                    X_val_synthetic, y_val_synthetic = X_val_synthetic[idx], y_val_synthetic[idx]
                    
                    all_val_sets['llm_patterns'] = (X_val_synthetic, y_val_synthetic)
                    
                    print(f"✅ 하이브리드 LLM 기반 검증 세트 생성 완료: {X_val_synthetic.shape}")
                    print(f"   정상: {np.sum(y_val_synthetic == 0):,}, 이상: {np.sum(y_val_synthetic == 1):,}")
                else:
                    if llm_step == "start" or llm_step is None:
                        print("⏸️ 수동 개입 필요: 위 안내에 따라 웹 LLM 사용 후 재실행")
                        return {}  # 수동 개입 대기
                    else:
                        print("❌ 하이브리드 LLM 이상치 생성 실패")
                        
            except Exception as e:
                print(f"❌ 하이브리드 LLM 생성 중 오류: {e}")
        
        else:
            print(f"\n🤖 API LLM 기반 이상치 패턴 분석 및 생성...")
            try:
                llm_generator = LLMAnomalyGenerator(
                    api_key=openai_api_key, 
                    num_anomaly_conditions=num_anomaly_conditions
                )
                
                synthetic_anomalies = llm_generator.generate_anomalies(
                    X=X_original,
                    y=y_original,
                    anomaly_count=len(X_anomaly_val),
                    feature_names=feature_names,
                    dataset_name=dataset_name,
                    num_conditions=num_anomaly_conditions
                )
                
                if len(synthetic_anomalies) > 0:
                    X_val_synthetic = np.vstack([X_normal_val, synthetic_anomalies])
                    y_val_synthetic = np.concatenate([
                        np.zeros(len(X_normal_val)), 
                        np.ones(len(synthetic_anomalies))
                    ])
                    
                    idx = np.random.RandomState(RANDOM_SEED).permutation(len(y_val_synthetic))
                    X_val_synthetic, y_val_synthetic = X_val_synthetic[idx], y_val_synthetic[idx]
                    
                    all_val_sets['llm_patterns'] = (X_val_synthetic, y_val_synthetic)
                    
                    print(f"✅ API LLM 기반 검증 세트 생성 완료: {X_val_synthetic.shape}")
                    print(f"   정상: {np.sum(y_val_synthetic == 0):,}, 이상: {np.sum(y_val_synthetic == 1):,}")
                else:
                    print("❌ API LLM 이상치 생성 실패")
                    
            except Exception as e:
                print(f"❌ API LLM 생성 중 오류: {e}")
    else:
        print("⚠️ LLM 생성이 비활성화되어 있습니다.")
    
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
    
    # 3. t-SNE 시각화 생성 (이상치가 실제로 생성된 경우만)
    if all_val_sets and len(all_val_sets) > 0:
        try:
            from evaluation.tsne_visualization import create_detailed_tsne_plots
            print(f"\n🔍 t-SNE 시각화 생성 중...")
            
            create_detailed_tsne_plots(
                X_normal_val=X_normal_val,
                synthetic_val_sets=all_val_sets,
                X_anomaly_val=X_anomaly_val,
                results_dir=results_dir
            )
            
            print(f"✅ t-SNE 시각화 완료")
        except ImportError:
            print("⚠️ t-SNE 시각화 모듈을 가져올 수 없습니다.")
        except Exception as e:
            print(f"❌ t-SNE 시각화 생성 실패: {e}")
    
    return all_val_sets