import sys
import os
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from viet_qa.data.loader import load_qa_dataset
from viet_qa.eval.metrics import evaluate_predictions

def main():
    parser = argparse.ArgumentParser(description="Evaluate QA Models")
    parser.add_argument("--model_type", type=str, choices=["extractive", "generative"], required=True)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    
    print(f"--- Evaluating {args.model_type.upper()} Model ---")
    
    if args.model_type == "extractive":
        from viet_qa.models.extractive import ExtractiveQAModel
        model = ExtractiveQAModel() # Load default local checkpoint
    else:
        from viet_qa.models.generative import GenerativeQAModel
        model = GenerativeQAModel()
        
    print(f"Loading validation dataset... (Max Samples: {args.samples})")
    val_dataset = load_qa_dataset("validation", max_samples=args.samples)
    
    predictions = []
    references = []
    latencies = []
    
    print("Running inference...")
    for item in tqdm(val_dataset, desc=f"Eval {args.model_type}"):
        context = item.get("context", "")
        question = item.get("question", "")

        raw = item.get("answer_text", "") or item.get("answers", {})
        if isinstance(raw, str):
            answers = [raw] if raw else []
        elif isinstance(raw, dict):
            answers = raw.get("text", [])
        else:
            answers = list(raw) if raw else []

        try:
            res = model.predict(question, context)
            ans = res.get("answer", "")
            predictions.append(ans)
            references.append(answers)
            latencies.append(res.get("latency_ms", 0))
        except Exception as e:
            predictions.append("")
            references.append(answers)
            latencies.append(0)
            
    metrics = evaluate_predictions(predictions, references)
    avg_lat = sum(latencies)/len(latencies) if latencies else 0
    
    print(f"\n=====================================")
    print(f" REPORT METRICS: {args.model_type.upper()} ({len(predictions)} samples)")
    print(f"=====================================")
    print(f" Exact Match (EM): {metrics['exact_match']:.4f}")
    print(f" F1 Score:         {metrics['f1']:.4f}")
    print(f" Avg Latency:      {avg_lat:.2f} ms")
    print(f"=====================================\n")

if __name__ == "__main__":
    main()
