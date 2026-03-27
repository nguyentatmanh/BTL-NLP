from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import time

class BaseQAModel(ABC):
    def __init__(self, model_name: str):
        """
        Base blueprint for Question Answering models.
        """
        self.model_name = model_name

    @abstractmethod
    def predict(self, question: str, context: str) -> Dict[str, Any]:
        """
        Predict an answer given a question and context.
        Outputs a Dictionary that at minimum contains:
        {
            "answer": str,
            "latency_ms": float,
            "confidence": float,
            "supported_by_context": bool,
            "model_type": str
        }
        """
        pass
        
    def _measure_latency(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Utility wrapper to measure function execution time in milliseconds."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        return result, latency_ms
