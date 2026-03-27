import torch
from transformers import pipeline
from typing import Dict, Any
from .base import BaseQAModel

class ExtractiveQAModel(BaseQAModel):
    def __init__(self, model_name_or_path: str = "nguyenvulebinh/vi-mrc-base"):
        """
        Inference class that wraps `pipeline`. 
        `model_name_or_path` can be a Hub ID or a local checkpoint folder.
        """
        super().__init__(model_name_or_path)
        # We auto-detect cuda but fallback to cpu
        self.device = 0 if torch.cuda.is_available() else -1
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            device=self.device
        )

    def _predict(self, question: str, context: str) -> tuple:
        # returns dict with 'score', 'start', 'end', 'answer'
        res = self.qa_pipeline(question=question, context=context)
        return res['answer'], res.get('score', 1.0)
        
    def predict(self, question: str, context: str) -> Dict[str, Any]:
        (answer, score), latency_ms = self._measure_latency(self._predict, question, context)
        return {
            "answer": answer,
            "latency_ms": latency_ms,
            "confidence": score,
            "supported_by_context": True, # Extractive models fundamentally guarantee this 
            "model_type": "extractive",
            "model_name": self.model_name
        }
