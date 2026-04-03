from typing import Dict, Any, List

def preprocess_extractive(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess example for Extractive QA extraction.
    Supports both:
      - SQuAD-style: answers = {"text": [...], "answer_start": [...]}
      - Flat-style (ViSpanExtractQA): answer_text = "...", answer_start = int
    """
    context = example["context"]

    # Try SQuAD-style nested answers dict first
    answers = example.get("answers", None)
    if isinstance(answers, dict) and answers.get("text"):
        text_answers = answers.get("text", [])
        starts = answers.get("answer_start", [])
    else:
        # Flat format used by ntphuc149/ViSpanExtractQA
        raw_text = example.get("answer_text", "") or ""
        raw_start = example.get("answer_start", -1)
        text_answers = [raw_text] if raw_text else []
        starts = [int(raw_start) if raw_start is not None else -1] if raw_text else []
    
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
