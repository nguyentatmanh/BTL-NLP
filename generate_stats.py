import sys
import os

sys.path.append(os.path.join(r"d:\BTL NLP\viet_qa", "src"))

from viet_qa.data.loader import load_qa_dataset
from viet_qa.data.preprocess import preprocess_extractive
from datasets import DatasetDict

TARGET_FILE = r"C:\Users\ADMIN\.gemini\antigravity\brain\6dbe417e-adf1-427b-ae54-d75db7e155c6\dataset_statistics.md"

def main():
    """
    Script Thống Kê Dữ Liệu (Stat Generator).
    Duyệt lại toàn bộ Dataset ntphuc149/ViSpanExtractQA.
    Đo đạc tỷ lệ nhiễu, số ký tự văn bản trung bình, số từ trung bình để lấy căn cứ đặt cấu hình MAX_SEQ_LENGTH.
    """
    print("Đang tải dữ liệu...")
    dataset = load_qa_dataset(split="all")
    
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"dataset": dataset})
        
    stats = {}
    
    # Quét qua từng tập Train, Validation, Test
    for split_name, ds in dataset.items():
        print(f"Đang phân tích tập {split_name}...")
        total_samples = len(ds)
        valid_spans_count = 0
        avg_context_len = 0
        avg_question_len = 0
        
        for item in ds:
            # Cho đi qua bộ lọc nhiễu xem có lọc được con data nào hỏng không
            processed = preprocess_extractive(item)
            if len(processed.get("valid_answers", [])) > 0:
                valid_spans_count += 1
                
            avg_context_len += len(item.get("context", ""))
            avg_question_len += len(item.get("question", ""))
            
        avg_context_len = avg_context_len / total_samples if total_samples > 0 else 0
        avg_question_len = avg_question_len / total_samples if total_samples > 0 else 0
        
        # Lưu vào sổ tay
        stats[split_name] = {
            "Total Samples": total_samples,
            "Samples with Valid Spans": valid_spans_count,
            "Average Context Length (chars)": round(avg_context_len, 2),
            "Average Question Length (chars)": round(avg_question_len, 2)
        }
        
    # Tạo mã nguồn Markdown
    md_content = "# Báo cáo Thống kê Dữ liệu (ViSpanExtractQA Dataset)\n\n"
    md_content += "Bản báo cáo này được tự động xuất ra để kiểm định thông số của bộ Dataset.\n\n"
    
    for split, data in stats.items():
        md_content += f"## Tập học: `{split}`\n"
        md_content += f"- **Tổng số mẫu**: {data['Total Samples']}\n"
        md_content += f"- **Số lượng mẫu hợp lệ (Không bị lệch khung)**: {data['Samples with Valid Spans']} ({data['Samples with Valid Spans']/data['Total Samples']*100:.2f}%)\n"
        md_content += f"- **Độ dài trung bình Ngữ Cảnh**: {data['Average Context Length (chars)']} ký tự\n"
        md_content += f"- **Độ dài trung bình Câu Hỏi**: {data['Average Question Length (chars)']} ký tự\n\n"
        
    md_content += "\n> [!NOTE]\n> Mẫu Hợp Lệ (Valid Span) là những mẫu có trường `answer_start` chỉ đúng tọa độ của chỉ mục `text` nằm trong ruột của đoạn `context`.\n"
    
    # Ghi xuất báo cáo
    os.makedirs(os.path.dirname(TARGET_FILE), exist_ok=True)
    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"Đã xuất thành công báo cáo đánh giá ra file {TARGET_FILE}. Bạn có thể copy ném vào Báo Cáo Latex!")

if __name__ == "__main__":
    main()
