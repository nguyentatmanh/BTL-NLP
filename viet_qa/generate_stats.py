import sys
import os

sys.path.append(os.path.join(r"d:\BTL NLP\viet_qa", "src"))

from viet_qa.data.loader import load_qa_dataset
from viet_qa.data.preprocess import preprocess_extractive
from datasets import DatasetDict

TARGET_FILE = r"C:\Users\ADMIN\.gemini\antigravity\brain\6dbe417e-adf1-427b-ae54-d75db7e155c6\dataset_statistics.md"

def main():
    print("Loading dataset...")
    dataset = load_qa_dataset(split="all")
    
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"dataset": dataset})
        
    stats = {}
    
    for split_name, ds in dataset.items():
        print(f"Processing split {split_name}...")
        total_samples = len(ds)
        valid_spans_count = 0
        avg_context_len = 0
        avg_question_len = 0
        
        for item in ds:
            processed = preprocess_extractive(item)
            if len(processed.get("valid_answers", [])) > 0:
                valid_spans_count += 1
                
            avg_context_len += len(item.get("context", ""))
            avg_question_len += len(item.get("question", ""))
            
        avg_context_len = avg_context_len / total_samples if total_samples > 0 else 0
        avg_question_len = avg_question_len / total_samples if total_samples > 0 else 0
        
        stats[split_name] = {
            "Total Samples": total_samples,
            "Samples with Valid Spans": valid_spans_count,
            "Average Context Length (chars)": round(avg_context_len, 2),
            "Average Question Length (chars)": round(avg_question_len, 2)
        }
        
    md_content = "# ViSpanExtractQA Dataset Statistics\n\n"
    md_content += "This report was auto-generated to show the characteristics of `ntphuc149/ViSpanExtractQA`.\n\n"
    
    for split, data in stats.items():
        md_content += f"## Split: `{split}`\n"
        md_content += f"- **Total Samples**: {data['Total Samples']}\n"
        md_content += f"- **Samples w/ Valid Extractable Spans**: {data['Samples with Valid Spans']} ({data['Samples with Valid Spans']/data['Total Samples']*100:.2f}%)\n"
        md_content += f"- **Avg Context Length**: {data['Average Context Length (chars)']} chars\n"
        md_content += f"- **Avg Question Length**: {data['Average Question Length (chars)']} chars\n\n"
        
    md_content += "\n> [!NOTE]\n> Valid extractable spans are answers whose `answer_start` index aligns perfectly with the `text` inside the `context`.\n"
    
    os.makedirs(os.path.dirname(TARGET_FILE), exist_ok=True)
    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"Successfully wrote stats to {TARGET_FILE}")

if __name__ == "__main__":
    main()
