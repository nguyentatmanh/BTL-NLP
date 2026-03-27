from typing import Dict, Any, List

def preprocess_extractive(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess example for Extractive QA extraction.
    Ensures that we extract a valid span and rectifies mismatch.
    """
    context = example["context"]
    answers = example.get("answers", {})
    text_answers = answers.get("text", [])
    starts = answers.get("answer_start", [])
    
    valid_spans = []
    
    for text, start in zip(text_answers, starts):
        is_valid = False
        # If strict validation fails, we try to find it natively
        if start >= 0 and start < len(context) and context[start:start+len(text)] == text:
            is_valid = True
        else:
            # Fallback for mismatched dataset offsets
            idx = context.find(text)
            if idx != -1:
                start = idx
                is_valid = True
                
        if is_valid:
            valid_spans.append({"text": text, "answer_start": start})
            
    example["valid_answers"] = valid_spans
    return example
