import torch
from transformers import pipeline
from typing import Dict, Any
from .base import BaseQAModel
import os

class ExtractiveQAModel(BaseQAModel):
    # Default to the locally trained checkpoint
    _DEFAULT_CKPT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints", "extractive"
    )

    def __init__(self, model_name_or_path: str = None):
        """
        Inference class that wraps `pipeline`. 
        `model_name_or_path` can be a Hub ID or a local checkpoint folder.
        """
        if model_name_or_path is None:
            model_name_or_path = self._DEFAULT_CKPT
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
        
        # Tiền xử lý xóa bỏ các dấu câu rác dính ở 2 đầu do Tokenizer (dấu phẩy, chấm, v.v...)
        ans = res['answer'].strip(' ,.;:?!\n\t"\'')
        
        return ans, res.get('score', 1.0)
        
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
