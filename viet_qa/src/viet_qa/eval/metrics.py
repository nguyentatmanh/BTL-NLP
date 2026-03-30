import string
import re
import time
from typing import List, Dict

def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    
    # Remove punctuation
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_exact_match(prediction: str, truth: str) -> int:
    """Checks if the normalized strings match exactly."""
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction: str, truth: str) -> float:
    """Computes F1 score based on word overlap."""
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
        
    common_tokens = set(pred_tokens) & set(truth_tokens)
    if not common_tokens:
        return 0.0
        
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def evaluate_predictions(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Calculate average EM and F1 over a dataset.
    references is a list of lists since one question can have multiple valid answers.
    """
    em_scores = []
    f1_scores = []
    
    for pred, refs in zip(predictions, references):
        # Compute best score against all references for a single prediction
        best_em = max([compute_exact_match(pred, ref) for ref in refs]) if refs else 0
        best_f1 = max([compute_f1(pred, ref) for ref in refs]) if refs else 0.0
        
        em_scores.append(best_em)
        f1_scores.append(best_f1)
        
    return {
        "exact_match": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    }

class Timer:
    """Simple context manager for profiling latency."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.latency_ms = (self.end - self.start) * 1000
