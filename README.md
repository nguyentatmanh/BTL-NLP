# Vietnamese Open-Domain Question Answering System

Hệ thống **Hỏi - Đáp mở miền (Open-Domain QA) cho tiếng Việt**, kết hợp giữa bộ truy hồi ngữ cảnh và các mô hình đọc hiểu để trả lời câu hỏi từ tập văn bản lớn. Dự án hỗ trợ cả hai hướng tiếp cận:

- **Extractive QA**: trích xuất câu trả lời trực tiếp từ ngữ cảnh.
- **Generative QA**: sinh câu trả lời tự nhiên bằng mô hình ngôn ngữ.

Repo hiện gồm:

- **Backend FastAPI** để phục vụ suy luận và đánh giá.
- **Frontend Streamlit** để nhập câu hỏi và so sánh kết quả.
- **Retriever dựa trên BM25** cho bước tìm ngữ cảnh liên quan.
- **Pipeline huấn luyện và đánh giá** cho mô hình Extractive QA.

---

## 1. Kiến trúc hệ thống

Luồng xử lý chính của hệ thống:

1. Người dùng nhập câu hỏi.
2. Retriever tìm ra các đoạn ngữ cảnh liên quan nhất trong tập dữ liệu.
3. Reader xử lý các ngữ cảnh đó để tìm hoặc sinh câu trả lời.
4. Hệ thống trả về đáp án tốt nhất cùng thông tin debug/candidate.

Các thành phần chính:

- **Retriever**: lớp `TfidfRetriever`, hiện đã được nâng cấp để dùng **BM25**.
- **Extractive Reader**: mô hình fine-tune từ `xlm-roberta-base`.
- **Generative Reader**: mặc định dùng `Qwen/Qwen2.5-1.5B-Instruct`.
- **API**: các endpoint như `/health`, `/ask`, `/predict/extractive`, `/predict/generative`, `/compare`.
- **UI**: giao diện Streamlit để chạy demo và quan sát kết quả.

---

## 2. Cấu trúc thư mục

> Các lệnh trong README này giả định bạn làm việc trong thư mục **`viet_qa/`** của repo.

```text
viet_qa/
├── src/
│   └── viet_qa/
│       ├── api/               # FastAPI app
│       ├── config/            # Cấu hình train
│       ├── data/              # Loader / preprocess dữ liệu
│       ├── eval/              # Metrics đánh giá
│       ├── models/            # Retriever + QA models
│       ├── train/             # Script train / eval extractive
│       ├── ui/                # Streamlit app
│       ├── utils/             # Script tải weights
│       └── checkpoints/       # Nơi lưu checkpoint local
├── tests/                     # Test và smoke test
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 3. Yêu cầu môi trường

Khuyến nghị:

- Python virtual environment
- `pip`
- Windows / Linux / macOS
- Docker (nếu chạy bằng container)

---

## 4. Cài đặt dự án

### Bước 1. Clone repo

```bash
git clone https://github.com/nguyentatmanh/BTL-NLP.git
cd BTL-NLP/viet_qa
```

### Bước 2. Tạo môi trường ảo

#### Windows (CMD)

```bash
python -m venv venv
venv\Scripts\activate
```

#### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Bước 3. Cài thư viện

```bash
pip install -r requirements.txt
```

---

## 5. Tải model weights đã train

Do thư mục checkpoint có dung lượng lớn, weights không được đẩy trực tiếp lên GitHub. Repo đã có sẵn script để tải model từ Google Drive và giải nén đúng thư mục mà hệ thống extractive đang sử dụng.

Chạy lệnh:

```bash
python src/viet_qa/utils/download_weights.py
```

Script này sẽ:

- tải file từ Google Drive bằng `gdown`
- dùng file ID: `1viAih2eZk7X8C1BO7YW4zh_fusDfrzcA`
- giải nén vào thư mục:

```text
src/viet_qa/checkpoints/extractive
```

### Link model gốc

```text
https://drive.google.com/file/d/1viAih2eZk7X8C1BO7YW4zh_fusDfrzcA/view?usp=drive_link
```

## ĐỪNG TẢI MODEL NÀY VỀ TỪ LINK DRIVE TRÊN, SẮP TỚI SẼ UPDATE MODEL MỚI LÊN!!!

### Nếu tải bằng script bị lỗi

Một số nguyên nhân thường gặp:

- Kết nối mạng bị gián đoạn.
- Máy chưa cài đủ dependency trong `requirements.txt`.

Khi đó bạn có thể:

1. tải file zip thủ công từ link Drive ở trên
2. giải nén vào đúng thư mục:

```text
src/viet_qa/checkpoints/extractive
```

> Sau khi giải nén xong, thư mục checkpoint phải chứa các file model/tokenizer của Hugging Face để `AutoTokenizer.from_pretrained(...)` và `AutoModelForQuestionAnswering.from_pretrained(...)` đọc được.

---

## 6. Chạy hệ thống local

### 6.1. Thiết lập `PYTHONPATH`

Để import package ổn định khi chạy từ thư mục `viet_qa/`, hãy thêm `src` vào `PYTHONPATH`.

#### Windows (CMD)

```bash
set PYTHONPATH=%cd%\src
```

#### Windows (PowerShell)

```powershell
$env:PYTHONPATH = "$PWD\src"
```

#### Linux / macOS

```bash
export PYTHONPATH="$(pwd)/src"
```

---

### 6.2. Chạy backend FastAPI

```bash
uvicorn src.viet_qa.api.main:app --reload --port 8000
```

Sau khi chạy thành công:

- API: `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/docs`

### Kiểm tra nhanh backend

Mở trình duyệt hoặc dùng curl:

```bash
curl http://localhost:8000/health
```

---

### 6.3. Chạy frontend Streamlit

Mở terminal mới, kích hoạt lại môi trường ảo rồi chạy:

```bash
streamlit run src/viet_qa/ui/app.py
```

Giao diện thường mở tại:

```text
http://localhost:8501
```

UI hiện cho phép:

- nhập câu hỏi tiếng Việt
- chọn `extractive` hoặc `generative`
- gọi endpoint `/ask`
- hiển thị câu trả lời tốt nhất, điểm số và các candidate context

---

## 7. Chạy bằng Docker Compose

Nếu bạn muốn khởi động toàn bộ hệ thống bằng container:

```bash
docker-compose up --build
```

Docker Compose hiện cấu hình:

- service `api` chạy trên port `8000`
- service `ui` chạy trên port `8501`
- UI gọi backend qua biến môi trường `API_URL=http://api:8000`

---

## 8. Các endpoint chính

### `GET /health`

Kiểm tra trạng thái hệ thống:

- backend đã chạy chưa
- model nào đã được load
- retriever đã sẵn sàng chưa
- số lượng context đã index

### `POST /ask`

Endpoint chính cho bài toán **open-domain QA**.

Ví dụ request:

```json
{
  "question": "Hà Nội nằm ở đâu?",
  "top_k": 3,
  "model_type": "extractive"
}
```

Gợi ý giá trị `model_type`:

- `extractive`
- `generative`

### `POST /predict/extractive`

Dự đoán với mô hình extractive khi bạn đã có sẵn `question` và `context`.

### `POST /predict/generative`

Dự đoán với mô hình generative khi bạn đã có sẵn `question` và `context`.

### `POST /compare`

So sánh đầu ra giữa hai mô hình trên cùng một câu hỏi/ngữ cảnh.

---

## 9. Huấn luyện mô hình Extractive QA

Pipeline train extractive hiện dùng cấu hình trong `src/viet_qa/config/train_config.py`:

- model backbone: `xlm-roberta-base`
- `MAX_SEQ_LENGTH = 384`
- `STRIDE = 128`
- `LEARNING_RATE = 2e-5`
- `NUM_EPOCHS = 3`
- `BATCH_SIZE = 4`
- output checkpoint: `src/viet_qa/checkpoints/extractive`

Chạy train:

```bash
python -m viet_qa.train.train_extractive
```

Script train sẽ:

- load dataset QA
- preprocess theo kiểu sliding window cho extractive QA
- fine-tune mô hình
- lưu checkpoint cuối cùng vào thư mục checkpoint local

---

## 10. Đánh giá mô hình Extractive QA

Sau khi train xong hoặc khi đã có sẵn checkpoint local, bạn có thể đánh giá như sau:

```bash
python -m viet_qa.train.eval_extractive --model_path "src/viet_qa/checkpoints/extractive" --samples 500
```

Ý nghĩa tham số:

- `--model_path`: đường dẫn checkpoint local hoặc tên model trên Hugging Face Hub
- `--samples`: số mẫu validation dùng để đánh giá

Script eval sẽ tính các metric QA và thống kê độ trễ suy luận trung bình.

---

## 11. Chạy test

```bash
pip install pytest httpx
pytest tests/
```

Nếu gặp lỗi import, hãy chắc chắn rằng bạn đã thiết lập `PYTHONPATH` như ở phần trên.

---

## 12. Một số lỗi thường gặp

### 1. Không load được model extractive

Nguyên nhân phổ biến:

- chưa chạy `download_weights.py`
- giải nén model sai thư mục
- checkpoint chưa đầy đủ file tokenizer/model

Cách xử lý:

- kiểm tra lại thư mục `src/viet_qa/checkpoints/extractive`
- chạy lại script tải weights
- nếu cần, xoá thư mục cũ rồi tải lại

### 2. Streamlit không kết nối được backend

Hãy chắc chắn backend đang chạy tại:

```text
http://localhost:8000
```

Nếu frontend báo lỗi kết nối, kiểm tra lại lệnh:

```bash
uvicorn src.viet_qa.api.main:app --reload --port 8000
```

### 3. Lỗi import module `viet_qa`

Đây thường là do chưa set `PYTHONPATH`.

Hãy chạy lại phần thiết lập:

```bash
set PYTHONPATH=%cd%\src
```

hoặc:

```bash
export PYTHONPATH="$(pwd)/src"
```

### 4. Lỗi khi tải weights từ Google Drive

Script tải model có ghi rõ: hãy đảm bảo link Google Drive đã được cấp quyền truy cập công khai.

---

## 13. Gợi ý phát triển thêm

Một số hướng nâng cấp tiếp theo cho dự án:

- thêm benchmark rõ ràng giữa extractive và generative
- bổ sung logging và tracing cho từng request
- cache retriever index để giảm thời gian khởi động
- thêm endpoint batch inference
- đóng gói checkpoint và config thành release artifact
- triển khai production bằng Nginx + Docker Compose / Kubernetes

---

