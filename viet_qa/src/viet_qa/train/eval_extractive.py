import sys
import os
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from viet_qa.data.loader import load_qa_dataset
from viet_qa.models.extractive import ExtractiveQAModel
from viet_qa.eval.metrics import evaluate_predictions

def main():
    """Script Đánh giá Độc lập dành riêng cho mô hình Extractive sau khi Train xong."""
    parser = argparse.ArgumentParser(description="Evaluate a trained Extractive QA model")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the local checkpoint or HuggingFace model hub ID")
    parser.add_argument("--samples", type=int, default=100, 
                        help="Number of samples from validation split to evaluate")
    args = parser.parse_args()
    
    print(f"Tiến hành load bộ não từ: {args.model_path}")
    model = ExtractiveQAModel(args.model_path)
    
    print(f"Đang tải bài thi tệp Validation... (Max: {args.samples} câu)")
    val_dataset = load_qa_dataset("validation", max_samples=args.samples)
    
    predictions = []
    references = []
    latencies = []
    
    print("Mô hình đang làm bài kiểm tra...")
    for item in tqdm(val_dataset, desc="Evaluating"):
        context = item.get("context", "")
        question = item.get("question", "")

        # Dataset ViSpanExtractQA dùng field 'answer_text' dạng chuỗi phẳng, chứ không phải mảng lồng SQuAD.
        raw = item.get("answer_text", "") or item.get("answers", {})
        if isinstance(raw, str):
            answers = [raw] if raw else []
        elif isinstance(raw, dict):
            answers = raw.get("text", [])
        else:
            answers = list(raw) if raw else []

        try:
            # Cho mô hình làm bài
            res = model.predict(question, context)
            predictions.append(res.get("answer", "")) # Bài ráp của model
            references.append(answers)                # Đáp án chuẩn (Đáp án gốc)
            latencies.append(res.get("latency_ms", 0))
        except Exception as e:
            print(f"Warning: Failed to predict for sample: {e}")
            predictions.append("")
            references.append(answers)
            latencies.append(0)
            
    # Nộp bài đi so sánh điểm Exact Match và F1.
    metrics = evaluate_predictions(predictions, references)
    avg_lat = sum(latencies)/len(latencies) if latencies else 0
    
    # In thẻ điểm
    print(f"\n--- THẺ ĐIỂM {args.model_path} ({len(predictions)} câu) ---")
    print(f"Exact Match (Khớp 100%): {metrics['exact_match']:.4f}")
    print(f"F1 Score (Độ dính):      {metrics['f1']:.4f}")
    print(f"Avg Latency (Trễ):       {avg_lat:.2f} ms")

if __name__ == "__main__":
    main()
