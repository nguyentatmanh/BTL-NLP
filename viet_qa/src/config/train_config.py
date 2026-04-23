import os

class TrainConfig:
    """Cấu hình dùng để Train mô hình Học vẹt (Extractive)."""
    
    # Sử dụng backbone xlm-roberta. Bắt buộc phải dùng FastTokenizer để nó tính được số chỉ mục index của chữ cái (return_offset_mapping)
    MODEL_NAME = "xlm-roberta-base"

    # Cấu hình Tokenizer (Băm chữ)
    MAX_SEQ_LENGTH = 448  # Độ dài tối đa 1 đoạn văn (Dựa trên báo cáo generate_stats là 400+)
    STRIDE = 160          # Nếu đoạn văn bị cắt làm đôi, 160 kí tự sẽ được gối đầu lên nhau để tránh chặt đứt mất câu trả lời đứng ở rìa.

    # Các siêu tham số Train (Hyperparameters)
    LEARNING_RATE = 2e-5  # Tốc độ học (Nhỏ để tránh phá hỏng trọng số RoBERTa)
    NUM_EPOCHS = 4        # Học đi học lại 4 lần
    BATCH_SIZE = 4        # Mỗi lô (Batch) cầm qua RAM là 4 mẫu
    WEIGHT_DECAY = 0.01   # Phạt L2 tránh chống học vẹt cục bộ (Overfitting)

    # Thư mục xuất kho
    OUTPUT_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints",
        "extractive"
    )
