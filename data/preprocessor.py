"""
Data preprocessing module for tabular anomaly detection.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config.settings import RANDOM_SEED, MAX_NORMAL_SAMPLES, MAX_ANOMALY_SAMPLES
from .data_utils import split_data, compute_feature_indices


class Preprocessor:
    """
    Preprocessor for tabular data in anomaly detection tasks.
    
    Handles data scaling, splitting, and preparation for model training.
    """
    
    def __init__(
        self,
        scaling_type: str = "standard",
        seed: int = RANDOM_SEED,
        max_normal: int = MAX_NORMAL_SAMPLES,
        max_anomaly: int = MAX_ANOMALY_SAMPLES,
    ):
        """
        Initialize Preprocessor.
        
        Args:
            scaling_type: Type of scaling ('standard', 'minmax', or None)
            seed: Random seed
            max_normal: Maximum number of normal samples to use
            max_anomaly: Maximum number of anomaly samples to use
        """
        np.random.seed(seed)
        
        self.scaling_type = scaling_type
        self.seed = seed
        self.max_normal = max_normal
        self.max_anomaly = max_anomaly
        self.scaling_params = None

    def prepare_dataset_splits(self, X, y, metadata):
        """
        Split dataset into train/validation/test sets for anomaly detection.
        
        Args:
            X: Feature matrix
            y: Labels (0=normal, 1=anomaly)
            metadata: Dataset metadata
            
        Returns:
            tuple: (X_normal_train, X_normal_val, X_val_real, y_val_real, X_test, y_test, X_anomaly_val)
        """
        print(f"📊 원본 데이터: {X.shape}")
        print(f"📊 클래스 분포 - 정상: {np.sum(y == 0):,}, 이상: {np.sum(y == 1):,}")

        # 대용량 데이터 최적화
        if np.sum(y == 0) > self.max_normal * 2:
            print(f"⚡ 대용량 정상 데이터 감지. {self.max_normal:,}개로 제한")
        if np.sum(y == 1) > self.max_anomaly * 2:
            print(f"⚡ 대용량 이상 데이터 감지. {self.max_anomaly:,}개로 제한")

        # 데이터 제한 및 분리
        X_normal = X[y == 0][:self.max_normal]
        X_anomaly = X[y == 1][:self.max_anomaly]
        
        # 데이터 분할
        X_normal_train, X_normal_holdout = train_test_split(
            X_normal, test_size=0.4, random_state=self.seed
        )
        X_anomaly_val, X_anomaly_test = train_test_split(
            X_anomaly, test_size=0.7, random_state=self.seed
        )
        X_normal_val, X_normal_test = train_test_split(
            X_normal_holdout, test_size=0.5, random_state=self.seed
        )
        
        # 최종 데이터셋 구성
        X_val_real = np.vstack([X_normal_val, X_anomaly_val])
        y_val_real = np.concatenate([np.zeros(len(X_normal_val)), np.ones(len(X_anomaly_val))])
        
        X_test = np.vstack([X_normal_test, X_anomaly_test])
        y_test = np.concatenate([np.zeros(len(X_normal_test)), np.ones(len(X_anomaly_test))])
        
        # 데이터 셔플
        idx = np.random.RandomState(self.seed).permutation(len(y_val_real))
        X_val_real, y_val_real = X_val_real[idx], y_val_real[idx]
        
        idx = np.random.RandomState(self.seed).permutation(len(y_test))
        X_test, y_test = X_test[idx], y_test[idx]
        
        print(f"\n📋 데이터셋 분할 완료:")
        print(f"   Train (정상만): {X_normal_train.shape}")
        print(f"   Real Validation: {X_val_real.shape} (정상: {np.sum(y_val_real == 0):,}, 이상: {np.sum(y_val_real == 1):,})")
        print(f"   Test: {X_test.shape} (정상: {np.sum(y_test == 0):,}, 이상: {np.sum(y_test == 1):,})")
            
        return X_normal_train, X_normal_val, X_val_real, y_val_real, X_test, y_test, X_anomaly_val

    def prepare_advanced_splits(self, X, y, metadata):
        """
        Advanced data preparation following the reference preprocessing pipeline.
        
        Args:
            X: Feature matrix  
            y: Labels
            metadata: Dataset metadata
            
        Returns:
            tuple: (train_dict, test_dict) with processed data
        """
        print(f"📊 고급 데이터 전처리 시작...")
        
        # Convert to DataFrame for processing
        df = pd.DataFrame(X, columns=metadata['column_names'])
        
        # Create NaN mask (assume no missing values for now)
        nan_mask = df.notnull().astype(int)
        
        # Split indices
        normal_idx = np.where(y == 0)[0]
        anomaly_idx = np.where(y == 1)[0]

        np.random.shuffle(normal_idx)
        num_train = len(normal_idx) // 2
        train_idx = normal_idx[:num_train]
        test_idx = np.concatenate([normal_idx[num_train:], anomaly_idx])

        # Split data
        X_train, y_train = split_data(df, y, nan_mask, train_idx)
        X_test, y_test = split_data(df, y, nan_mask, test_idx)

        # Compute feature indices
        cat_idxs, con_idxs = compute_feature_indices(
            df, metadata['cat_encoding'], 
            metadata['categorical_columns'], 
            metadata['continuous_columns']
        )

        # Compute scaling parameters
        self.scaling_params = self._compute_scaling_stats(X_train, con_idxs, metadata)

        # Create datasets
        train_dict = self._make_dataset(X_train, y_train, cat_idxs, con_idxs, metadata)
        test_dict = self._make_dataset(X_test, y_test, cat_idxs, con_idxs, metadata)

        return train_dict, test_dict

    def _compute_scaling_stats(self, X_train, con_idxs, metadata):
        """Compute scaling statistics for continuous features."""
        if not con_idxs or self.scaling_type is None:
            return {"type": "none"}
            
        data = X_train["data"][:, con_idxs].astype(float)
        n_features = len(con_idxs)

        if self.scaling_type == "minmax":
            d_min = data.min(0)
            d_max = data.max(0)
            d_range = d_max - d_min
            d_range = np.where(d_range < 1e-6, 1.0, d_range)

            return {"type": "minmax", "min": d_min, "range": d_range}

        elif self.scaling_type == "standard":
            mean = data.mean(0)
            std = data.std(0)
            std = np.where(std < 1e-6, 1.0, std)

            return {"type": "standard", "mean": mean, "std": std}

        else:
            raise NotImplementedError(f"Unsupported scaling_type: {self.scaling_type}")

    def _make_dataset(self, X, y, cat_idxs, con_idxs, metadata):
        """Create dataset dictionary with processed features."""
        X_data, X_mask = X["data"], X["mask"]
        y_data = y["data"]

        # Process continuous features
        if con_idxs:
            con_data = X_data[:, con_idxs].astype(np.float32)
            con_mask = X_mask[:, con_idxs].astype(np.int64)
            
            # Apply scaling
            if self.scaling_params["type"] == "standard":
                con_data = (con_data - self.scaling_params["mean"]) / self.scaling_params["std"]
            elif self.scaling_params["type"] == "minmax":
                con_data = (con_data - self.scaling_params["min"]) / self.scaling_params["range"]
        else:
            con_data = np.empty((len(y_data), 0), dtype=np.float32)
            con_mask = np.empty((len(y_data), 0), dtype=np.int64)

        # Process categorical features
        if cat_idxs and metadata['cat_encoding'] == "int_emb":
            cat_data = X_data[:, cat_idxs].astype(np.int64)
            cat_mask = X_mask[:, cat_idxs].astype(np.int64)

            # Add CLS token
            cls_token = np.zeros((y_data.shape[0], 1), dtype=np.int64)
            cls_mask = np.ones((y_data.shape[0], 1), dtype=np.int64)

            cat_data = np.concatenate([cls_token, cat_data], axis=1)
            cat_mask = np.concatenate([cls_mask, cat_mask], axis=1)
        else:
            cat_data = None
            cat_mask = None

        return {
            "X_cat_data": cat_data,
            "X_cat_mask": cat_mask,
            "X_cont_data": con_data,
            "X_cont_mask": con_mask,
            "X_data": X_data,  # Original data for compatibility
            "y": y_data,
            "metadata": metadata,
        }

    def apply_standard_scaling(self, X_train, X_val, X_test):
        """
        Apply standard scaling to datasets.
        
        Args:
            X_train: Training data
            X_val: Validation data  
            X_test: Test data
            
        Returns:
            tuple: (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✅ 표준화 스케일링 적용 완료")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def prepare_dataset_splits(X_original, y_original, metadata=None, **kwargs):
    """
    Convenience function for dataset splitting.
    
    Args:
        X_original: Original feature matrix
        y_original: Original labels
        metadata: Dataset metadata (optional)
        **kwargs: Additional arguments for Preprocessor
        
    Returns:
        tuple: Dataset splits
    """
    preprocessor = Preprocessor(**kwargs)
    return preprocessor.prepare_dataset_splits(X_original, y_original, metadata)