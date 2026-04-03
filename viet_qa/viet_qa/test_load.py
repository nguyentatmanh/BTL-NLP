from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_path = r"D:\BTL NLP\viet_qa\src\viet_qa\checkpoints\extractive"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

print("Load model thành công!")