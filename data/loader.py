"""
Data loading module for tabular anomaly detection.
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .data_utils import (
    infer_column_types,
    impute_and_cast,
    split_data,
    compute_feature_indices,
)


class DataLoader:
    """
    Data loader for tabular anomaly detection datasets.
    
    Handles data loading, encoding, and basic preprocessing.
    """
    
    def __init__(
        self,
        dataset_name: str,
        data_dir: str = '/lab-di/nfsdata/home/suhee.yoon/autoanoeval/data/adbench_column',
        seed: int = 42,
        cat_encoding: str = "int",
    ):
        """
        Initialize DataLoader.
        
        Args:
            dataset_name: Name of the dataset
            data_dir: Directory containing the dataset
            seed: Random seed
            cat_encoding: Categorical encoding method ('int', 'onehot', 'int_emb')
        """
        np.random.seed(seed)
        
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.cat_encoding = cat_encoding
        self.seed = seed
        
        # Load data
        self.data = self._load_csv()
        self.X = self.data.drop(columns=["label"], errors="ignore")
        self.column_names = self.X.columns.tolist()
        self.y = np.array(self.data["label"], dtype=int)
        
        # Infer column types
        self.categorical_columns, self.continuous_columns = infer_column_types(self.X)
        self.org_continuous_columns = self.continuous_columns.copy()
        self.cat_dims = []
        
        print(f"📊 데이터셋 로드 완료: {self.dataset_name}")
        print(f"📈 전체 샘플: {len(self.data):,}")
        print(f"📊 특성 수: {len(self.column_names)}")
        print(f"🔤 범주형 컬럼: {len(self.categorical_columns)}개 - {self.categorical_columns}")
        print(f"📈 연속형 컬럼: {len(self.continuous_columns)}개")
        print(f"🏷️ 레이블 분포: 정상 {np.sum(self.y == 0):,}개, 이상 {np.sum(self.y == 1):,}개")

    def _load_csv(self):
        """Load CSV file."""
        csv_path = os.path.join(self.data_dir, f"{self.dataset_name}.csv")
        print(f"📥 CSV 파일 로드: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"📊 로드된 데이터 형태: {df.shape}")
        
        return df

    def prepare_data(self):
        """
        Prepare data with encoding and preprocessing.
        
        Returns:
            tuple: (X_processed, y_processed, metadata)
        """
        # Apply categorical encoding
        if self.cat_encoding == "onehot":
            self._encode_onehot()
        elif self.cat_encoding == "int":
            self._encode_int()
        elif self.cat_encoding == "int_emb":
            self._encode_int_emb()
        else:
            raise NotImplementedError(f"Unsupported cat_encoding: {self.cat_encoding}")

        # Impute and cast data types
        self.X = impute_and_cast(
            self.X, self.categorical_columns, self.continuous_columns
        )
        
        # Encode labels
        self.y = LabelEncoder().fit_transform(self.y)
        
        print(f"✅ 데이터 전처리 완료")
        print(f"📊 최종 특성 차원: {self.X.shape[1]}")
        
        # Create metadata
        metadata = {
            'column_names': self.column_names,
            'categorical_columns': self.categorical_columns,
            'continuous_columns': self.continuous_columns,
            'org_continuous_columns': self.org_continuous_columns,
            'cat_dims': self.cat_dims,
            'cat_encoding': self.cat_encoding,
            'dataset_name': self.dataset_name
        }
        
        return self.X.values, self.y, metadata

    def _encode_onehot(self):
        """Apply one-hot encoding to categorical columns."""
        if not self.categorical_columns:
            return
            
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe_arr = ohe.fit_transform(self.X[self.categorical_columns])
        ohe_cols = ohe.get_feature_names_out(self.categorical_columns)
        df_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols, index=self.X.index)
        
        self.X = pd.concat(
            [self.X.drop(columns=self.categorical_columns), df_ohe], axis=1
        )
        
        # Update column types
        self.continuous_columns = self.X.columns.tolist()
        self.categorical_columns = []
        
        print(f"🔄 One-hot 인코딩 완료: {len(ohe_cols)}개 특성 생성")

    def _encode_int(self):
        """Apply integer encoding to categorical columns."""
        if not self.categorical_columns:
            return
            
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col].astype(str))
            print(f"🔢 {col}: {len(le.classes_)}개 클래스 → 정수 인코딩")
        
        # Update column types
        self.continuous_columns = self.X.columns.tolist()
        self.categorical_columns = []

    def _encode_int_emb(self):
        """Apply integer encoding with embedding dimensions tracking."""
        if not self.categorical_columns:
            return
            
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col].astype(str))
            self.cat_dims.append(len(le.classes_))
            print(f"🔢 {col}: {len(le.classes_)}개 클래스 → 임베딩용 정수 인코딩")


def load_dataset(dataset_name: str, cat_encoding: str = "int", **kwargs):
    """
    Convenience function to load and prepare dataset.
    
    Args:
        dataset_name: Name of the dataset
        cat_encoding: Categorical encoding method
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        tuple: (X, y, metadata)
    """
    loader = DataLoader(dataset_name=dataset_name, cat_encoding=cat_encoding, **kwargs)
    return loader.prepare_data()