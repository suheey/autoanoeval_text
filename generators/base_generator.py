from abc import ABC, abstractmethod
import numpy as np

class BaseAnomalyGenerator(ABC):
    """이상치 생성기 추상 베이스 클래스"""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
    
    @abstractmethod
    def generate_anomalies(self, X, y, **kwargs):
        """이상치 생성 추상 메서드"""
        pass
    
