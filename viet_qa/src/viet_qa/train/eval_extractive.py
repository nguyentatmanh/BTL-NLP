import sys
import os
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from viet_qa.data.loader import load_qa_dataset
from viet_qa.models.extractive import ExtractiveQAModel
from viet_qa.eval.metrics import evaluate_predictions

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Extractive QA model")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the local checkpoint or HuggingFace model hub ID")
    parser.add_argument("--samples", type=int, default=100, 
                        help="Number of samples from validation split to evaluate")
    args = parser.parse_args()
    
    print(f"Loading checkpoint from: {args.model_path}")
    model = ExtractiveQAModel(args.model_path)
    
    print(f"Loading validation dataset... (Max Samples: {args.samples})")
    val_dataset = load_qa_dataset("validation", max_samples=args.samples)
    
    predictions = []
    references = []
    latencies = []
    
    print("Running inference...")
    for item in tqdm(val_dataset, desc="Evaluating"):
        context = item.get("context", "")
        question = item.get("question", "")
        answers = item.get("answers", {}).get("text", [])
        
        try:
            res = model.predict(question, context)
            predictions.append(res.get("answer", ""))
            references.append(answers)
            latencies.append(res.get("latency_ms", 0))
        except Exception as e:
            print(f"Warning: Failed to predict for sample: {e}")
            predictions.append("")
            references.append(answers)
            latencies.append(0)
            
    metrics = evaluate_predictions(predictions, references)
    avg_lat = sum(latencies)/len(latencies) if latencies else 0
    
    print(f"\n--- Evaluation Results ({len(predictions)} samples) ---")
    print(f"Exact Match (EM): {metrics['exact_match']:.4f}")
    print(f"F1 Score:         {metrics['f1']:.4f}")
    print(f"Avg Latency:      {avg_lat:.2f} ms")

if __name__ == "__main__":
    main()
