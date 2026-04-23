import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from viet_qa.config.train_config import TrainConfig
from viet_qa.data.loader import load_qa_dataset
from viet_qa.data.preprocess import preprocess_extractive


def prepare_train_features(examples, tokenizer, config):
    """
    Hàm cực kỳ quan trọng: Băm (Tokenize) dữ liệu để nhét vào mô hình.
    Mô hình không hiểu chuỗi văn bản, nó chỉ hiểu các ID số học (Token ID).
    """
    # Bước 1: Gọi Tokenizer cắt chữ thành số
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second", # Chỉ chặt bỏ phần Context nếu quá dài, giữ lại trọn vẹn Câu hỏi
        max_length=config.MAX_SEQ_LENGTH,
        stride=config.STRIDE,     # Gối đầu các từ ở phần biên bị chặt để không băm nát câu trả lời
        return_overflowing_tokens=True,
        return_offsets_mapping=True, # Lập bản đồ ánh xạ từ Token ID quay ngược lại số thứ tự ký tự gốc (Quan trọng nhất)
        padding="max_length",
    )

    # Bản đồ giúp biết được đoạn văn bị rách nát (overflow) này thuộc về data mẫu nào
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    # Bước 2: Dò tìm lại tọa độ Token chứa câu trả lời
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id) # Vị trí ký hiệu đặc biệt [CLS] (Chỉ sự vô vọng - không có đáp án)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["valid_answers"][sample_index]

        # Nếu không có câu trả lời, trỏ Start và End vào thẻ [CLS]
        if len(answers) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            answer = answers[0]
            start_char = answer["answer_start"]
            end_char = start_char + len(answer["text"])

            # Điển hình: 0 là Question, 1 là Context. Di chuyển trỏ tới đầu Context.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # Trỏ từ đuôi Context đổ ngược về
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Kiểm tra xem câu trả lời có rơi ra ngoài khu vực bị chặt (truncate) không
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Ép biên trái nhảy vào tìm vị trí Token sát nhất với chữ cái bắt đầu
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)

                # Ép biên phải nhảy ngược lại tìm Token cuối
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def main():
    """Luồng Huấn luyện (Train) mô hình Extractive QA"""
    config = TrainConfig()

    print(f"Đang gọi Tokenizer và xương sống Mô hình: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError(
            f"{config.MODEL_NAME} không hỗ trợ Fast Tokenizer. "
            "Thuật toán Extractive bắt buộc phải có offset_mapping từ nền tảng Rust của HuggingFace."
        )

    model = AutoModelForQuestionAnswering.from_pretrained(config.MODEL_NAME)

    print("Đang cào dữ liệu (Dataset)...")
    dataset = load_qa_dataset("all")  

    # 1. Chèn hàm tiền xử lý (Preprocess) để lọc và rà soát câu trả lời lỗi
    train_ds = dataset["train"].map(preprocess_extractive, load_from_cache_file=False)
    # Lọc bỏ data rác (Data không chứa valid_answers)
    train_ds = train_ds.filter(lambda x: len(x["valid_answers"]) > 0, load_from_cache_file=False)
    print(f"  Số lượng mẫu Train sau khi lọc nhiễu: {len(train_ds)}")

    val_ds = dataset["validation"].map(preprocess_extractive, load_from_cache_file=False)
    val_ds = val_ds.filter(lambda x: len(x["valid_answers"]) > 0, load_from_cache_file=False)
    print(f"  Số lượng mẫu Val sau khi lọc: {len(val_ds)}")

    # 2. Map hàm Tokenizer vào toàn bộ Dataset
    print("Đang băm Data Train (Tokenizing)...")
    train_tokenized = train_ds.map(
        lambda x: prepare_train_features(x, tokenizer, config),
        batched=True,
        remove_columns=train_ds.column_names, # Xóa bỏ chữ, chỉ ném Số (Token) vào mạng Neural
        load_from_cache_file=False,
        desc="Tokenizing train",
    )
    print(f"  Số lượng mảnh Token train: {len(train_tokenized)}")

    print("Đang băm Data Val...")
    val_tokenized = val_ds.map(
        lambda x: prepare_train_features(x, tokenizer, config),
        batched=True,
        remove_columns=val_ds.column_names,
        load_from_cache_file=False,
        desc="Tokenizing val",
    )
    print(f"  Số lượng mảnh Token val: {len(val_tokenized)}")

    if len(train_tokenized) == 0:
        raise ValueError(
            "Mảng Dữ liệu vô tình bị rỗng sạch sẽ sau khi Tokenize! "
            "Hãy kiểm tra lại độ dài chuỗi max_length hoặc bộ lọc tiền xử lý."
        )

    # 3. Kê đơn huấn luyện cho Lò luyện đan Trainer của HuggingFace
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        eval_strategy="epoch",  # Thi nháp 1 lần sau mỗi vòng (Epoch)
        save_strategy="epoch",  # Bấm Save Game liên tục sau mỗi vòng (Tránh cúp điện)
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.NUM_EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        fp16=torch.cuda.is_available(), # Bật nén ép số thực FP16 (X2 tốc độ nếu có card đồ họa)
        save_total_limit=2, # Chỉ giữ lại 2 checkpoint nặng nhất cho nhẹ ổ cứng
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
    )

    print("Phóng Hỏa! Bắt đầu huấn luyện...")
    trainer.train()

    print(f"Lưu Cúp (Model) vào thư mục: {config.OUTPUT_DIR}")
    trainer.save_model(config.OUTPUT_DIR)

if __name__ == "__main__":
    main()
