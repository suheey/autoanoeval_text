"""
Tabular data serialization utilities for LLM prompts.
"""

import numpy as np
import pandas as pd
import random
from typing import List, Dict, Any, Optional


def serialize_tabular_sample(
    data_row: np.ndarray, 
    column_names: List[str], 
    shuffle_columns: bool = True,
    format_style: str = "natural"
) -> str:
    """
    Serialize a single tabular data sample into text format for LLM prompts.
    
    Args:
        data_row: Single row of data as numpy array
        column_names: List of column names
        shuffle_columns: Whether to shuffle column order
        format_style: Serialization format ('natural', 'structured', 'csv')
        
    Returns:
        str: Serialized text representation
    """
    if len(data_row) != len(column_names):
        raise ValueError(f"Data row length ({len(data_row)}) doesn't match column names length ({len(column_names)})")
    
    # Create column indices
    if shuffle_columns:
        indices = list(range(len(column_names)))
        random.shuffle(indices)
    else:
        indices = list(range(len(column_names)))
    
    if format_style == "natural":
        # Natural language format: "feature_name is value"
        parts = []
        for idx in indices:
            col_name = column_names[idx]
            value = str(data_row[idx]).strip()
            if len(value) == 0 or value == 'nan':
                value = "None"
            parts.append(f"{col_name} is {value}")
        return ", ".join(parts)
    
    elif format_style == "structured":
        # Structured format: "feature_name: value"
        parts = []
        for idx in indices:
            col_name = column_names[idx]
            value = str(data_row[idx]).strip()
            if len(value) == 0 or value == 'nan':
                value = "None"
            parts.append(f"{col_name}: {value}")
        return " | ".join(parts)
    
    elif format_style == "csv":
        # CSV-like format
        header = ", ".join([column_names[idx] for idx in indices])
        values = ", ".join([str(data_row[idx]).strip() for idx in indices])
        return f"{header}\n{values}"
    
    else:
        raise ValueError(f"Unsupported format_style: {format_style}")


def serialize_multiple_samples(
    data_matrix: np.ndarray,
    column_names: List[str],
    labels: Optional[np.ndarray] = None,
    max_samples: int = 5,
    shuffle_columns: bool = True,
    format_style: str = "natural",
    include_labels: bool = False
) -> List[str]:
    """
    Serialize multiple tabular data samples.
    
    Args:
        data_matrix: Data matrix (n_samples, n_features)
        column_names: List of column names
        labels: Optional labels for each sample
        max_samples: Maximum number of samples to serialize
        shuffle_columns: Whether to shuffle column order
        format_style: Serialization format
        include_labels: Whether to include labels in serialization
        
    Returns:
        List[str]: List of serialized samples
    """
    n_samples = min(max_samples, len(data_matrix))
    
    # Randomly sample if we have more data than max_samples
    if len(data_matrix) > max_samples:
        indices = random.sample(range(len(data_matrix)), n_samples)
        selected_data = data_matrix[indices]
        selected_labels = labels[indices] if labels is not None else None
    else:
        selected_data = data_matrix
        selected_labels = labels
    
    serialized_samples = []
    
    for i in range(n_samples):
        row_text = serialize_tabular_sample(
            data_row=selected_data[i],
            column_names=column_names,
            shuffle_columns=shuffle_columns,
            format_style=format_style
        )
        
        if include_labels and selected_labels is not None:
            label = "normal" if selected_labels[i] == 0 else "anomaly"
            row_text = f"[{label}] {row_text}"
        
        serialized_samples.append(row_text)
    
    return serialized_samples


def create_dataset_description(
    data_matrix: np.ndarray,
    column_names: List[str],
    labels: np.ndarray,
    dataset_name: str = "Unknown",
    max_examples: int = 3
) -> str:
    """
    Create a comprehensive dataset description for LLM prompts.
    
    Args:
        data_matrix: Full data matrix
        column_names: List of column names  
        labels: Data labels
        dataset_name: Name of the dataset
        max_examples: Maximum number of examples to include
        
    Returns:
        str: Dataset description text
    """
    n_samples, n_features = data_matrix.shape
    n_normal = np.sum(labels == 0)
    n_anomaly = np.sum(labels == 1)
    
    # Basic statistics
    description = f"""Dataset: {dataset_name}
- Total samples: {n_samples:,}
- Features: {n_features}
- Normal samples: {n_normal:,} ({n_normal/n_samples:.1%})
- Anomaly samples: {n_anomaly:,} ({n_anomaly/n_samples:.1%})

Feature names: {', '.join(column_names)}

"""
    
    # Add example samples
    if max_examples > 0:
        description += "Example samples:\n"
        
        # Normal examples
        normal_indices = np.where(labels == 0)[0]
        if len(normal_indices) > 0:
            normal_examples = serialize_multiple_samples(
                data_matrix=data_matrix[normal_indices],
                column_names=column_names,
                labels=labels[normal_indices],
                max_samples=min(max_examples, len(normal_indices)),
                shuffle_columns=False,
                format_style="natural",
                include_labels=True
            )
            for example in normal_examples:
                description += f"  {example}\n"
        
        # Anomaly examples  
        anomaly_indices = np.where(labels == 1)[0]
        if len(anomaly_indices) > 0:
            anomaly_examples = serialize_multiple_samples(
                data_matrix=data_matrix[anomaly_indices],
                column_names=column_names,
                labels=labels[anomaly_indices],
                max_samples=min(max_examples, len(anomaly_indices)),
                shuffle_columns=False,
                format_style="natural",
                include_labels=True
            )
            for example in anomaly_examples:
                description += f"  {example}\n"
    
    return description


def prepare_llm_context_samples(
    X_normal: np.ndarray,
    X_anomaly: np.ndarray,
    column_names: List[str],
    max_normal_samples: int = 5,
    max_anomaly_samples: int = 3,
    format_style: str = "natural"
) -> Dict[str, Any]:
    """
    Prepare sample data for LLM context in anomaly generation.
    
    Args:
        X_normal: Normal samples
        X_anomaly: Anomaly samples
        column_names: List of column names
        max_normal_samples: Maximum normal samples to include
        max_anomaly_samples: Maximum anomaly samples to include
        format_style: Serialization format
        
    Returns:
        Dict containing serialized samples and metadata
    """
    # Serialize normal samples
    normal_samples = serialize_multiple_samples(
        data_matrix=X_normal,
        column_names=column_names,
        max_samples=max_normal_samples,
        shuffle_columns=True,
        format_style=format_style
    )
    
    # Serialize anomaly samples
    anomaly_samples = serialize_multiple_samples(
        data_matrix=X_anomaly,
        column_names=column_names,
        max_samples=max_anomaly_samples,
        shuffle_columns=True,
        format_style=format_style
    )
    
    return {
        "normal_samples": normal_samples,
        "anomaly_samples": anomaly_samples,
        "column_names": column_names,
        "format_style": format_style,
        "n_normal_samples": len(normal_samples),
        "n_anomaly_samples": len(anomaly_samples)
    }