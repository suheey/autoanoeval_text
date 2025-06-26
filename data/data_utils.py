"""
Data utilities for tabular anomaly detection preprocessing.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def infer_column_types(df):
    """
    Infer categorical and continuous column types from DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        tuple: (categorical_columns, continuous_columns)
    """
    categorical_columns = []
    continuous_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_columns.append(col)
        elif df[col].dtype in ['int64', 'float64']:
            # Check if it's actually categorical (few unique values)
            if df[col].nunique() <= 20 and df[col].dtype == 'int64':
                categorical_columns.append(col)
            else:
                continuous_columns.append(col)
        else:
            continuous_columns.append(col)
    
    return categorical_columns, continuous_columns


def impute_and_cast(df, categorical_columns, continuous_columns):
    """
    Impute missing values and cast data types.
    
    Args:
        df: Input DataFrame
        categorical_columns: List of categorical column names
        continuous_columns: List of continuous column names
        
    Returns:
        DataFrame: Processed DataFrame
    """
    df = df.copy()
    
    # Impute categorical columns with mode
    for col in categorical_columns:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col].fillna(mode_val[0], inplace=True)
            else:
                df[col].fillna('unknown', inplace=True)
    
    # Impute continuous columns with median
    for col in continuous_columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    # Cast data types
    for col in categorical_columns:
        df[col] = df[col].astype(str)
    
    for col in continuous_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill any remaining NaN values after conversion
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    return df


def split_data(X, y, nan_mask, indices):
    """
    Split data based on given indices.
    
    Args:
        X: Feature DataFrame
        y: Target array
        nan_mask: NaN mask array
        indices: Indices to select
        
    Returns:
        tuple: (X_split, y_split) dictionaries with data and mask
    """
    X_split = {
        'data': X.iloc[indices].values,
        'mask': nan_mask.iloc[indices].values
    }
    
    y_split = {
        'data': y[indices]
    }
    
    return X_split, y_split


def compute_feature_indices(X, cat_encoding, categorical_columns, continuous_columns):
    """
    Compute categorical and continuous feature indices.
    
    Args:
        X: Feature DataFrame
        cat_encoding: Categorical encoding type
        categorical_columns: List of categorical column names
        continuous_columns: List of continuous column names
        
    Returns:
        tuple: (cat_idxs, con_idxs)
    """
    if cat_encoding in ["onehot", "int"]:
        # All features are continuous after encoding
        cat_idxs = []
        con_idxs = list(range(X.shape[1]))
    elif cat_encoding == "int_emb":
        # Categorical features maintain their indices
        cat_idxs = [X.columns.get_loc(col) for col in categorical_columns if col in X.columns]
        con_idxs = [X.columns.get_loc(col) for col in continuous_columns if col in X.columns]
    else:
        cat_idxs = []
        con_idxs = list(range(X.shape[1]))
    
    return cat_idxs, con_idxs