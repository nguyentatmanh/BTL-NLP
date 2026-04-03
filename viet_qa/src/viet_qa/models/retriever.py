from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Tuple
import time
import re

class TfidfRetriever:
    """
    Module Truy hồi Ngữ cảnh (Retriever): Sử dụng thuật toán BM25.
    (Lưu ý: Tên class vẫn gán là TfidfRetriever nhằm đảm bảo tính tương thích với API).
    BM25 khắc phục hoàn toàn điểm yếu thống kê từ thông dụng của TF-IDF, đem lại độ Recall cực kỳ ổn định cho QA.
    """

    def __init__(self):
        self.bm25_model = None
        self.contexts: List[str] = []
        self._is_built = False

    def _tokenize(self, text: str) -> List[str]:
        """Tiền xử lý thô: Ép chữ thường, lược bỏ dấu câu và tách thành từng từ khóa rời rạc."""
        text = text.lower()
        # Tìm tất cả các cụm từ dính nhau, lược bỏ các khoảng trắng và dấu câu
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def build_index(self, contexts: List[str]):
        """Cào toàn bộ danh sách ngữ cảnh từ Dataset và Index vào hệ thống của BM25."""
        start = time.perf_counter()
        
        # Băm nhỏ tất cả các đoạn văn bản thành danh sách token
        print("  Tiến hành Tokenize mảng ngữ cảnh cho BM25...")
        tokenized_corpus = [self._tokenize(ctx) for ctx in contexts]
        
        print("  Đang khởi tạo nhân BM25...")
        self.bm25_model = BM25Okapi(tokenized_corpus)
        
        self.contexts = contexts
        self._is_built = True
        
        elapsed = time.perf_counter() - start
        print(f"  Hoàn tất cắm mốc BM25: {len(contexts)} văn bản trong {elapsed:.2f} giây")

    @property
    def is_built(self) -> bool:
        return self._is_built

    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
        """
        Dò tìm Top-K kết quả ngữ cảnh có tương quan lớn nhất đối với câu hỏi gốc.
        Đầu ra là danh sách mảng Bộ ba: Vị trí (Index), Điểm số (Score), Nguyên văn ngữ cảnh.
        """
        if not self._is_built:
            raise RuntimeError("Chưa khởi tạo Index. Vui lòng chạy hàm build_index() trước.")

        tokenized_query = self._tokenize(query)
        
        # Nhờ BM25 chấm điểm toàn bộ kho dữ liệu
        doc_scores = self.bm25_model.get_scores(tokenized_query)
        
        # Chuẩn hóa phổ điểm về dải 0.0 -> 1.0 (Giúp đồng bộ với điểm Confidence của mô hình QA)
        max_score = np.max(doc_scores)
        if max_score > 0:
            normalized_scores = doc_scores / max_score
        else:
            normalized_scores = doc_scores

        # Lấy Top K giá trị bự nhất (Hàm argsort mặc định chia từ nhỏ đến lớn, dán [::-1] để đảo ngược lại)
        top_indices = np.argsort(normalized_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((int(idx), float(normalized_scores[idx]), self.contexts[idx]))

        return results
