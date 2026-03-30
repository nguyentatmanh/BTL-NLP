import pytest
from viet_qa.data.preprocess import preprocess_extractive
from viet_qa.data.utils import validate_span, normalize_text

def test_normalize_text():
    assert normalize_text(" Xin Chào! ") == "xin chào"
    assert normalize_text("This is... a TEST!!!") == "this is a test"

def test_validate_span():
    context = "Đại học Bách khoa Hà Nội nằm ở đường Giải Phóng."
    ans = "đường Giải Phóng"
    
    start = context.find(ans)
    assert validate_span(context, ans, start) == True
    assert validate_span(context, ans, start + 1) == False

def test_preprocess_extractive():
    example = {
        "context": "Hà Nội là thủ đô của Việt Nam.",
        "answers": {
            "text": ["Hà Nội", "Việt Nam", "Sai mốc"],
            "answer_start": [0, 21, 100] # 100 is intentionally invalid
        }
    }
    
    res = preprocess_extractive(example)
    valid = res["valid_answers"]
    
    assert len(valid) == 2
    assert valid[0]["text"] == "Hà Nội"
    assert valid[1]["text"] == "Việt Nam"
