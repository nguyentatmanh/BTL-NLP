from datasets import load_dataset, DatasetDict
from typing import Dict, Any, List, Union

def load_qa_dataset(split: str = "all", max_samples: int = None) -> Union[Any, DatasetDict]:
    """
    Loads the ntphuc149/ViSpanExtractQA dataset.
    Args:
        split: 'train', 'validation', 'test', or 'all'.
        max_samples: Optional limit on the number of samples to load.
    Returns:
        HuggingFace Dataset or DatasetDict object.
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
    Standardize example format.
    The dataset usually contains 'question', 'context', and 'answers'.
    """
    answers = example.get("answers", {"text": []})
    text_answers = answers.get("text", [])
    
    return {
        "id": example.get("id", ""),
        "question": example.get("question", ""),
        "context": example.get("context", ""),
        "answers": text_answers
    }
