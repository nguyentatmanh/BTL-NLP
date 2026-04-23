# Hệ thống Hỏi đáp Tiếng Việt

Hệ thống này là một bài toán hỏi đáp tiếng Việt theo kiến trúc Retriever-Reader. Ở tầng truy hồi, hệ thống dùng BM25 để tìm ra những ngữ cảnh phù hợp nhất trong bộ dữ liệu. Ở tầng đọc hiểu, hệ thống hỗ trợ hai chế độ:

- `extractive`: trích xuất câu trả lời trực tiếp từ ngữ cảnh bằng mô hình QA đã fine-tune.
- `generative`: sinh câu trả lời ngắn gọn bằng mô hình ngôn ngữ `Qwen/Qwen2.5-1.5B-Instruct`.

Toàn bộ hệ thống được đóng gói thành:

- API backend bằng FastAPI
- giao diện demo bằng Streamlit
- script train/evaluate riêng cho mô hình extractive
- công cụ vẽ loss sau huấn luyện

## 1. Kiến trúc tổng thể

Luồng xử lý chính:

1. Người dùng nhập câu hỏi.
2. Backend nạp tập ngữ cảnh từ dataset `ntphuc149/ViSpanExtractQA`.
3. BM25 Retriever tìm `top-k` ngữ cảnh liên quan nhất.
4. Reader (`extractive` hoặc `generative`) suy luận câu trả lời trên từng ứng viên.
5. Backend kết hợp điểm retriever và reader để xếp hạng lại kết quả.
6. UI hiển thị câu trả lời tốt nhất, độ tin cậy và các candidate để debug.

## 2. Tính năng chính

- Hỏi đáp mở bằng tiếng Việt trên tập ngữ cảnh lớn.
- Hỗ trợ 2 chiến lược trả lời: trích xuất và sinh.
- Có sẵn API `/ask`, `/predict/*`, `/compare`, `/evaluate`.
- Có giao diện Streamlit để demo nhanh.
- Có checkpoint extractive lưu sẵn trong repo.
- Có script train, evaluate và trực quan hóa loss.

## 3. Cấu trúc thư mục

```text
.
├── Dockerfile
├── docker-compose.yml
├── plot_loss.py
├── requirements.txt
├── loss_chart.png
├── loss_report.md
└── viet_qa
    ├── src
    │   ├── api
    │   │   └── main.py
    │   ├── checkpoints
    │   │   └── extractive
    │   ├── config
    │   │   └── train_config.py
    │   ├── data
    │   │   ├── loader.py
    │   │   ├── preprocess.py
    │   │   └── utils.py
    │   ├── eval
    │   │   ├── metrics.py
    │   │   └── run_evaluation.py
    │   ├── models
    │   │   ├── base.py
    │   │   ├── extractive.py
    │   │   ├── generative.py
    │   │   └── retriever.py
    │   ├── train
    │   │   ├── eval_extractive.py
    │   │   └── train_extractive.py
    │   ├── ui
    │   │   └── app.py
    │   ├── utils
    │   │   ├── download_kaggle_model.py
    │   │   └── download_weights.py
    │   └── viet_qa
    │       └── __init__.py
    └── tests
        ├── conftest.py
        ├── test_api.py
        └── test_preprocess.py
```

## 4. Yêu cầu hệ thống

Khuyến nghị:

- Python `3.10`
- `pip` mới
- Docker Desktop nếu muốn chạy bằng container
- GPU CUDA nếu muốn chạy `generative` mượt hơn

Lưu ý quan trọng:

- Lần chạy đầu tiên, hệ thống có thể tải dataset từ Hugging Face.
- Nếu dùng chế độ `generative`, mô hình `Qwen/Qwen2.5-1.5B-Instruct` sẽ được tải về ở lần đầu suy luận.
- Nếu bạn muốn tải lại checkpoint extractive từ Kaggle, bạn cần có Kaggle API token.
- Chạy trên CPU vẫn được, nhưng `generative` sẽ chậm đáng kể.
- Mặc định backend dùng cổng `8000`, frontend dùng cổng `8501`.

## 5. Cài đặt môi trường local

### Bước 0: clone repo từ GitHub

```powershell
git clone https://github.com/nguyentatmanh/BTL-NLP.git
cd BTL-NLP
```

### Bước 1: tạo virtual environment

PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

### Bước 2: cài dependency

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Nếu bạn muốn chạy test local:

```powershell
pip install pytest httpx torchvision
```

### Bước 3: kiểm tra checkpoint

Repo có thể đã chứa checkpoint extractive tại:

`viet_qa/src/checkpoints/extractive`

Nếu bạn clone repo mới, muốn đồng bộ lại model mới nhất từ Kaggle, hoặc muốn ghi đè checkpoint hiện có, hãy dùng downloader mới bên dưới.

### Bước 4: cấu hình Kaggle API để tải model

Kaggle model page:

`https://www.kaggle.com/models/huynguyen199/vietnamese-open-domain`

Cách 1: dùng `kaggle.json` trên Windows

1. Vào `https://www.kaggle.com/settings`
2. Ở mục `API`, bấm `Create Legacy API Key`
3. Chép file `kaggle.json` vào:

```text
C:\Users\<TEN_USER>\.kaggle\kaggle.json
```

Cách 2: dùng biến môi trường

```powershell
$env:KAGGLE_API_TOKEN = "your_token_here"
```

### Bước 5: tải checkpoint extractive từ Kaggle

```powershell
py .\viet_qa\src\utils\download_kaggle_model.py
```

Script sẽ đồng bộ checkpoint vào:

`viet_qa/src/checkpoints/extractive`

Nếu bạn muốn ghi đè checkpoint hiện có:

```powershell
py .\viet_qa\src\utils\download_kaggle_model.py --force
```

Lưu ý:

- Theo tài liệu Kaggle, download handle của model có dạng `<owner>/<model>/<framework>/<variation>`.
- Link bạn cung cấp là model-level page, chưa bao gồm `framework` và `variation`.
- Script mặc định sẽ thử handle `huynguyen199/vietnamese-open-domain/transformers/default`.
- Nếu variation thực tế trên Kaggle khác `transformers/default`, hãy chạy lại với:

```powershell
py .\viet_qa\src\utils\download_kaggle_model.py --model-handle "huynguyen199/vietnamese-open-domain/<framework>/<variation>" --force
```

File `viet_qa/src/utils/download_weights.py` hiện được giữ lại như wrapper tương thích và sẽ gọi sang downloader Kaggle mới.

## 6. Chạy hệ thống local

### Cách chạy nhanh nhất

Mở 2 terminal.

Terminal 1: chạy API backend

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn viet_qa.api.main:app --app-dir .\viet_qa\src --host 0.0.0.0 --port 8000 --reload
```

Terminal 2: chạy Streamlit UI

```powershell
.\.venv\Scripts\Activate.ps1
$env:API_URL = "http://127.0.0.1:8000"
streamlit run .\viet_qa\src\ui\app.py
```

Sau khi chạy:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Streamlit UI: `http://127.0.0.1:8501`

### Điều gì xảy ra khi backend khởi động

Ngay khi API start, hàm `lifespan` trong `viet_qa/src/api/main.py` sẽ:

1. tải dataset `ViSpanExtractQA`
2. gom tất cả `context` duy nhất
3. build chỉ mục BM25 trong RAM

Vì vậy lần khởi động đầu có thể mất thêm thời gian, đặc biệt nếu mạng chậm hoặc dataset chưa được cache.

### Kiểm tra backend đã sẵn sàng

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

Kết quả mong đợi sẽ có dạng:

```json
{
  "status": "ok",
  "models_loaded": [],
  "retriever_ready": true,
  "total_contexts": 12345
}
```

### Gọi thử API hỏi đáp

PowerShell:

```powershell
$body = @{
  question   = "Ai là hiệu trưởng đầu tiên của Đại học Bách khoa Hà Nội?"
  top_k      = 3
  model_type = "extractive"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

`model_type` hỗ trợ:

- `extractive`
- `generative`

## 7. Chạy bằng Docker

### Chạy toàn bộ stack

```powershell
docker compose up --build
```

Sau khi container chạy xong:

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- UI: `http://localhost:8501`

### Chạy riêng từng service

Chỉ backend:

```powershell
docker compose up --build api
```

Chỉ frontend:

```powershell
docker compose up --build ui
```

Lưu ý:

- `docker-compose.yml` đang mount mã nguồn vào container bằng volume, nên thay đổi code local sẽ phản ánh vào container.
- Ở lần build đầu, Docker sẽ cài toàn bộ dependency Python nên mất thời gian hơn.

## 8. Huấn luyện mô hình extractive

Chạy script fine-tune:

```powershell
.\.venv\Scripts\Activate.ps1
py .\viet_qa\src\train\train_extractive.py
```

Script này sẽ:

- tải dataset train/validation
- tiền xử lý đáp án về dạng span hợp lệ
- tokenize bằng `xlm-roberta-base`
- huấn luyện với Hugging Face `Trainer`
- lưu checkpoint vào `viet_qa/src/checkpoints/extractive`

Hyperparameter chính đang được cấu hình trong `viet_qa/src/config/train_config.py`:

- backbone: `xlm-roberta-base`
- max sequence length: `448`
- stride: `160`
- learning rate: `2e-5`
- batch size: `4`
- epoch: `4`

## 9. Đánh giá mô hình

### Đánh giá checkpoint extractive đã train

```powershell
.\.venv\Scripts\Activate.ps1
py .\viet_qa\src\train\eval_extractive.py --model_path .\viet_qa\src\checkpoints\extractive --samples 100
```

### Đánh giá 2 chế độ reader bằng script tổng quát

Extractive:

```powershell
py .\viet_qa\src\eval\run_evaluation.py --model_type extractive --samples 100
```

Generative:

```powershell
py .\viet_qa\src\eval\run_evaluation.py --model_type generative --samples 20
```

Khuyến nghị dùng số mẫu nhỏ hơn cho `generative` nếu máy chỉ có CPU.

## 10. Vẽ biểu đồ loss sau huấn luyện

```powershell
py .\plot_loss.py
```

Script sẽ:

- tìm `trainer_state.json` trong thư mục checkpoint
- sinh bảng loss theo epoch
- lưu `loss_report.md`
- lưu ảnh `loss_chart.png`

## 11. Kiểm thử

Sau khi cài thêm dependency test:

```powershell
py -m pytest -q
```

Hai nhóm test hiện có:

- `viet_qa/tests/test_preprocess.py`: kiểm tra tiền xử lý span
- `viet_qa/tests/test_api.py`: smoke test cho API

## 12. API chính

### `GET /health`

Kiểm tra tình trạng server, model đã nạp hay chưa, retriever đã build xong chưa.

### `POST /ask`

Endpoint chính cho bài toán open-domain QA.

Request:

```json
{
  "question": "Ai là hiệu trưởng đầu tiên của Đại học Bách khoa Hà Nội?",
  "top_k": 3,
  "model_type": "extractive"
}
```

### `POST /predict/extractive`

Suy luận extractive trên một `context` cụ thể.

### `POST /predict/generative`

Suy luận generative trên một `context` cụ thể.

### `POST /compare`

Chạy song song extractive và generative trên cùng một câu hỏi/ngữ cảnh để so sánh.

### `POST /evaluate`

Đánh giá nhanh mô hình trên tập validation từ API.

## 13. Hiệu năng và artifact hiện có

Repo hiện có sẵn:

- checkpoint extractive đã train
- `loss_chart.png`
- `loss_report.md`

Bảng loss hiện tại:

| Epoch | Train Loss | Validation Loss |
|-------|------------|-----------------|
| 0 | 3.2785 | N/A |
| 1 | 1.5974 | 1.3817 |
| 2 | 1.1849 | 1.3290 |
| 3 | 0.9832 | 1.3103 |
| 4 | 0.7578 | 1.4300 |

## 14. Lỗi thường gặp

### `ModuleNotFoundError: No module named 'viet_qa'`

Hãy chạy backend bằng đúng lệnh:

```powershell
uvicorn viet_qa.api.main:app --app-dir .\viet_qa\src --host 0.0.0.0 --port 8000 --reload
```

Hoặc chạy các script train/eval bằng đúng path trong README này.

### Download Kaggle model báo lỗi xác thực

Hãy kiểm tra một trong hai cách sau:

- file `C:\Users\<TEN_USER>\.kaggle\kaggle.json`
- biến môi trường `KAGGLE_API_TOKEN`

Bạn có thể lấy token tại:

`https://www.kaggle.com/settings`

### Download Kaggle model báo sai handle

Nguyên nhân là link `https://www.kaggle.com/models/huynguyen199/vietnamese-open-domain` chỉ là model page tổng, trong khi Kaggle download handle cần đủ:

`<owner>/<model>/<framework>/<variation>`

Khi đó, hãy mở đúng variation trên Kaggle rồi chạy lại với:

```powershell
py .\viet_qa\src\utils\download_kaggle_model.py --model-handle "huynguyen199/vietnamese-open-domain/<framework>/<variation>" --force
```

### API khởi động lâu

Nguyên nhân thường là:

- đang tải dataset lần đầu
- đang build BM25 index
- máy đang chạy trên CPU chậm

### Chế độ `generative` trả lời chậm

Điều này là bình thường nếu bạn:

- chạy bằng CPU
- chưa có cache model
- dùng `Qwen2.5-1.5B-Instruct` lần đầu

Nếu chỉ cần demo nhanh, hãy dùng `extractive`.

### PowerShell không cho activate `.venv`

Chạy:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### Port `8000` hoặc `8501` đã bị chiếm

Đổi port khi chạy:

```powershell
uvicorn viet_qa.api.main:app --app-dir .\viet_qa\src --host 0.0.0.0 --port 8001 --reload
streamlit run .\viet_qa\src\ui\app.py --server.port 8502
```

## 15. Ghi chú triển khai

- Backend dùng lazy loading cho model reader, nên model chỉ được nạp khi endpoint cần đến.
- Retriever được build một lần khi server khởi động.
- Checkpoint extractive là lựa chọn phù hợp nhất nếu bạn cần inference ổn định và nhẹ hơn.
- Reader generative phù hợp hơn cho các câu trả lời cần diễn đạt mềm hơn, đổi lại chi phí suy luận cao hơn.
