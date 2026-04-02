import string
import re
import time
import unicodedata
from typing import List, Dict

def normalize_text(text: str) -> str:
    """
    Chuẩn hóa văn bản Tiếng Việt trước khi chấm điểm.
    Mục đích: Không để những lỗi vặt (như viết HOA, thừa dấu phẩy, khoảng trắng bị dôi) làm trừ điểm oan uổng của Mô hình.
    """
    if not text:
        return ""
    # Chuẩn hóa về chuỗi Unicode NFC (Thể thức tốt nhất để tránh lỗi font chữ tiếng Việt bị tách rời dấu)
    text = unicodedata.normalize("NFC", str(text)).lower().strip()
    
    # Loại bỏ dấu câu bằng phân loại category Unicode (Tránh lỗi mất dấu cấu Việt ngữ nếu dùng string.punctuation thô)
    text = "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("P")
    )
    
    # Giết sạch các khoảng trắng bị thừa (chuyển 3 space thành 1 space)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_exact_match(prediction: str, truth: str) -> int:
    """
    Hàm chấm điểm EM (Exact Match)
    1.0 Nếu Khớp 100% không trệch 1 chữ cái.
    0.0 Nếu Lệch dù chỉ 1 ký tự.
    (Khá khắc nghiệt, thường áp dụng cho QA SQuAD).
    """
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction: str, truth: str) -> float:
    """
    Hàm chấm điểm F1 (Độ dính)
    Cho phép du di nếu đáp án dự đoán có dính 1 phần tới đáp án chuẩn bị sót/thừa chữ.
    """
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
        
    common_tokens = set(pred_tokens) & set(truth_tokens)
    if not common_tokens:
        return 0.0
        
    prec = len(common_tokens) / len(pred_tokens)  # Độ Chính Xác (Precision)
    rec = len(common_tokens) / len(truth_tokens)  # Độ Bám Phủ (Recall)
    
    # Trung bình điều hòa (Harmonic Mean)
    return 2 * (prec * rec) / (prec + rec)

def evaluate_predictions(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Hàm cốt lõi: Tiếp nhận toàn bộ mảng các câu trả lời do Model nhả ra vs mảng Đáp án chuẩn do Chuyên gia gắn nhãn.
    Nó sẽ duyệt liên tục và lấy Trung Bình Cầm (Average) điểm EM và báo cáo ra thành bảng.
    """
    em_scores = []
    f1_scores = []
    
    for pred, refs in zip(predictions, references):
        # Vì 1 câu hỏi có thể có nhiều đáp án chuẩn (vd: 'việt nam', 'nước việt nam'), 
        # nên ta tính điểm với mọi đáp án và Lấy Cái Điểm Cao Nhất (Thương cảm cho thí sinh)
        best_em = max([compute_exact_match(pred, ref) for ref in refs]) if refs else 0
        best_f1 = max([compute_f1(pred, ref) for ref in refs]) if refs else 0.0
        
        em_scores.append(best_em)
        f1_scores.append(best_f1)
        
    return {
        "exact_match": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    }

class Timer:
    """Đồng hồ bấm giờ để chấm điểm Tốc độ của AI (Latency)."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.latency_ms = (self.end - self.start) * 1000
