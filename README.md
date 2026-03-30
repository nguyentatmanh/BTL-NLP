# 🇻🇳 Vietnamese Open-Domain Question Answering System

Một hệ thống Hỏi-Đáp tự động (Open-Domain QA) dành cho tiếng Việt, kết hợp giữa thuật toán tìm kiếm truyền thống và các mô hình ngôn ngữ hiện đại (Transformers & RAG).

## 🚀 Tính năng nổi bật
- **Kiến trúc Retriever-Reader:** Tự động tìm kiếm ngữ cảnh dựa trên bộ dữ liệu hơn 26,000 đoạn văn bản.
- **Hybrid Search:** Sử dụng thuật toán **BM25** (Best Matching 25) cho tốc độ truy xuất cực nhanh.
- **Đa mô hình Reader:**
  - **Extractive QA:** Sử dụng **XLM-RoBERTa** được fine-tune chuyên biệt trên tiếng Việt để trích xuất đáp án chính xác từ văn bản.
  - **Generative QA (RAG):** Sử dụng **Qwen2.5-1.5B-Instruct** để tổng hợp câu trả lời tự nhiên, đầy đủ ngữ pháp.
- **Giao diện trực quan:** Streamlit Dashboard hỗ trợ so sánh song song 2 mô hình, hiển thị tốc độ phản hồi (Latency) và điểm tự tin (Confidence).

## 📊 Kết quả Đánh giá (Metrics)
Đánh giá trên 500 mẫu (validation set) của tập dữ liệu `ViSpanExtractQA`:

| Metric | Extractive Model (XLM-R) |
| :--- | :--- |
| **Exact Match (EM)** | **0.5160** |
| **F1 Score** | **0.7150** |
| **Avg Latency** | ~185ms |

## 🛠 Cài đặt

1. **Clone project:**
   ```bash
   git clone https://github.com/nguyentatmanh/BTL-NLP.git
   cd BTL-NLP
   ```

2. **Khởi tạo môi trường:**
   ```bash
   conda create -n vietqa python=3.10
   conda activate vietqa
   pip install -r requirements.txt
   ```

3. **Tải Model đã Train:**
   *Hiện tại thư mục `src/viet_qa/checkpoints/` bị bỏ qua trên GitHub do dung lượng lớn (>1GB). Bạn chỉ cần chạy lệnh sau để tự động tải và giải nén:*
   ```bash
   python src/viet_qa/utils/download_weights.py
   ```

## 🖥 Cách chạy hệ thống

### 1. Khởi chạy Backend (API)
Mở terminal 1:
```bash
set PYTHONPATH=%cd%\src
uvicorn src.viet_qa.api.main:app --reload --port 8000
```

### 2. Khởi chạy Frontend (UI)
Mở terminal 2:
```bash
streamlit run src/viet_qa/ui/app.py
```

## 📂 Cấu trúc thư mục
- `src/viet_qa/api/`: Chứa mã nguồn FastAPI điều phối hệ thống.
- `src/viet_qa/models/`: Định nghĩa các lớp Retriever, Extractive và Generative.
- `src/viet_qa/ui/`: Giao diện Streamlit người dùng.
- `src/viet_qa/train/`: Scripts huấn luyện và đánh giá mô hình.

## 🤝 CONTRIBUTING
Mọi đóng góp hoặc báo lỗi xin vui lòng tạo Issue hoặc gửi Pull Request trên GitHub.

---
© 2026 - [Nguyen Tat Manh]
