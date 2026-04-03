import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Dict, Any
from .base import BaseQAModel

class GenerativeQAModel(BaseQAModel):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        super().__init__(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        if self.device == "cpu":
            self.model.to(self.device)
            
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def _create_prompt(self, question: str, context: str) -> str:
        prompt = f"""Dựa vào văn bản dưới đây, hãy trả lời câu hỏi một cách ngắn gọn và chính xác nhất trích xuất từ văn bản. Nếu không có thông tin, hãy trả lời "Tôi không biết".\n\nVăn bản: {context}\n\nCâu hỏi: {question}\n\nTrả lời:"""
        
        messages = [
            {"role": "system", "content": "Bạn là một AI chuyên đọc hiểu tiếng Việt. Hãy trích xuất câu trả lời ngắn gọn nhất từ ngữ cảnh được cho."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt

    def _predict(self, question: str, context: str) -> tuple:
        prompt = self._create_prompt(question, context)
        outputs = self.generator(
            prompt,
            max_new_tokens=64,
            do_sample=False,  # greedy decoding for factual extraction
            temperature=None, # ignore temp if do_sample=False
            top_p=None,
            return_full_text=False # pipeline handles removing prompt
        )
        answer = outputs[0]["generated_text"].strip()
        
        # --- Confidence & Supported Heuristics ---
        # 1. Check for refusal phrases
        refusals = ["không biết", "không có", "không đề cập", "không rõ", "tôi không", "chưa được"]
        lower_ans = answer.lower()
        if any(r in lower_ans for r in refusals):
            return answer, 0.1, False
            
        # 2. Check overlap to ensure it's context-supported
        import string
        def normalize(txt):
            return " ".join(txt.lower().translate(str.maketrans('', '', string.punctuation)).split())
            
        norm_ans = normalize(answer).split()
        norm_ctx = normalize(context).split()
        
        if len(norm_ans) == 0:
            return answer, 0.0, False
            
        ctx_set = set(norm_ctx)
        overlap = sum(1 for w in norm_ans if w in ctx_set)
        overlap_ratio = overlap / len(norm_ans)
        
        # We consider it conceptually supported if at least half the substantive words are in context
        supported = overlap_ratio >= 0.5 
        
        # Penalize confidence if heavily hallucinated
        confidence = float(max(overlap_ratio, 0.2)) if supported else 0.2
        
        return answer, min(confidence, 1.0), supported
        
    def predict(self, question: str, context: str) -> Dict[str, Any]:
        (answer, confidence, supported), latency_ms = self._measure_latency(self._predict, question, context)
        return {
            "answer": answer,
            "latency_ms": latency_ms,
            "confidence": confidence,
            "supported_by_context": supported,
            "model_type": "generative",
            "model_name": self.model_name
        }
