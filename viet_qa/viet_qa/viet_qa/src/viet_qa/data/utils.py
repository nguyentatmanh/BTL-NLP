import string
import re

def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def validate_span(context: str, answer_text: str, answer_start: int) -> bool:
    """
    Validates if the provided answer_start index yields the exact answer_text in the context.
    """
    if answer_start < 0 or answer_start >= len(context):
        return False
        
    extracted = context[answer_start : answer_start + len(answer_text)]
    return extracted == answer_text
