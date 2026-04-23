from typing import Dict, Any, List

def preprocess_extractive(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tiền xử lý ngữ liệu dành cho mô hình Extractive QA (Hỏi đáp trích xuất).
    Mô hình Extractive cần phải biết vị trí (index) chính xác của câu trả lời trong đoạn văn bản.
    Hàm này sẽ đối chiếu nội dung thực tế để tìm vị trí thật sự của đáp án.
    """
    context = example["context"]

    # Ưu tiên cấu trúc mảng lồng nhau chuẩn SQuAD
    answers = example.get("answers", None)
    if isinstance(answers, dict) and answers.get("text"):
        text_answers = answers.get("text", [])
        starts = answers.get("answer_start", [])
    else:
        # Nếu không có, dự phòng cấu trúc dữ liệu phẳng (phổ biến ở ntphuc149/ViSpanExtractQA)
        raw_text = example.get("answer_text", "") or ""
        raw_start = example.get("answer_start", -1)
        text_answers = [raw_text] if raw_text else []
        starts = [int(raw_start) if raw_start is not None else -1] if raw_text else []
    
    valid_spans = []
    
    for text, start in zip(text_answers, starts):
        is_valid = False
        # Thử nghiệm 1: Cắt chuỗi theo vị trí start đã cho xem có khớp 100% với đáp án hay không
        if start >= 0 and start < len(context) and context[start:start+len(text)] == text:
            is_valid = True
        else:
            # Thử nghiệm 2: Người gắn nhãn có thể đã đếm nhầm vị trí kí tự. Ta sẽ tự scan lại bằng context.find().
            idx = context.find(text)
            if idx != -1:
                start = idx
                is_valid = True
                
        # Chỉ khi nào định vị được đáp án thì mới đẩy vào mảng dữ liệu có hiệu lực
        if is_valid:
            valid_spans.append({"text": text, "answer_start": start})
            
    example["valid_answers"] = valid_spans
    return example
