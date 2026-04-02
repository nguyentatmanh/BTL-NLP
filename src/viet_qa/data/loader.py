from datasets import load_dataset, DatasetDict
from typing import Dict, Any, List, Union

def load_qa_dataset(split: str = "all", max_samples: int = None) -> Union[Any, DatasetDict]:
    """
    Hàm tĩnh để tải bộ dữ liệu (dataset) ViSpanExtractQA từ HuggingFace Hub.
    
    Tham số:
        split: Tên tập dữ liệu con cần tải ('train', 'validation', 'test' hoặc 'all').
        max_samples: Số lượng mẫu giới hạn để tải (Hữu ích khi test nhanh để không tốn RAM).
        
    Đầu ra:
        Dataset hoặc DatasetDict chứa các mẫu hỏi-đáp.
    """
    if split == "all":
        dataset = load_dataset("ntphuc149/ViSpanExtractQA")
        if max_samples:
            for s in dataset.keys():
                dataset[s] = dataset[s].select(range(min(len(dataset[s]), max_samples)))
        return dataset
    else:
        dataset = load_dataset("ntphuc149/ViSpanExtractQA", split=split)
        if max_samples:
            dataset = dataset.select(range(min(len(dataset), max_samples)))
        return dataset

def format_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hàm chuẩn hóa lại định dạng của dữ liệu thô.
    Lọc và giữ lại đúng các trường thiết yếu: 'id', 'question', 'context', 'answers'.
    """
    answers = example.get("answers", {"text": []})
    text_answers = answers.get("text", [])
    
    return {
        "id": example.get("id", ""),
        "question": example.get("question", ""),
        "context": example.get("context", ""),
        "answers": text_answers
    }
