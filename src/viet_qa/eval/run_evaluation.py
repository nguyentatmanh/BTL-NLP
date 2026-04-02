import sys
import os
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from viet_qa.data.loader import load_qa_dataset
from viet_qa.eval.metrics import evaluate_predictions

def main():
    """
    Script Chấm Điểm (Evaluation) cho phép chọn một trong hai Mô Hình để thi đấu:
    - extractive: Chấm điểm khả năng 'Học Vẹt' (Cắt chữ từ văn bản).
    - generative: Chấm điểm khả năng 'Diễn Đạt' (LLM sinh chữ).
    """
    parser = argparse.ArgumentParser(description="Evaluate QA Models")
    parser.add_argument("--model_type", type=str, choices=["extractive", "generative"], required=True)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    
    print(f"--- BẮT ĐẦU CHẤM ĐIỂM MÔ HÌNH {args.model_type.upper()} ---")
    
    # Kích hoạt thí sinh
    if args.model_type == "extractive":
        from viet_qa.models.extractive import ExtractiveQAModel
        model = ExtractiveQAModel() # Load bộ não đã train ở dưới máy
    else:
        from viet_qa.models.generative import GenerativeQAModel
        model = GenerativeQAModel() # Kéo Qwen 1.5B về múa
        
    print(f"Đang tải bài thi... (Max: {args.samples} câu)")
    val_dataset = load_qa_dataset("validation", max_samples=args.samples)
    
    predictions = []
    references = []
    latencies = []
    
    print("Bắt đầu làm bài thi (Inference)...")
    for item in tqdm(val_dataset, desc=f"Eval {args.model_type}"):
        context = item.get("context", "")
        question = item.get("question", "")

        # Trích lọc đáp án trúng tủ từ Dataset (Để lát so sánh)
        raw = item.get("answer_text", "") or item.get("answers", {})
        if isinstance(raw, str):
            answers = [raw] if raw else []
        elif isinstance(raw, dict):
            answers = raw.get("text", [])
        else:
            answers = list(raw) if raw else []

        try:
            # Giao cho mô hình làm
            res = model.predict(question, context)
            ans = res.get("answer", "")
            
            predictions.append(ans)
            references.append(answers)
            latencies.append(res.get("latency_ms", 0))
        except Exception as e:
            predictions.append("")
            references.append(answers)
            latencies.append(0)
            
    # Tính điểm
    metrics = evaluate_predictions(predictions, references)
    avg_lat = sum(latencies)/len(latencies) if latencies else 0
    
    print(f"\n=====================================")
    print(f" BÁO CÁO KẾT QUẢ: {args.model_type.upper()} ({len(predictions)} mẫu)")
    print(f"=====================================")
    print(f" Exact Match (Khớp 100%): {metrics['exact_match']:.4f}")
    print(f" F1 Score (Độ dính):      {metrics['f1']:.4f}")
    print(f" Avg Latency (Trễ):       {avg_lat:.2f} ms")
    print(f"=====================================\n")

if __name__ == "__main__":
    main()
