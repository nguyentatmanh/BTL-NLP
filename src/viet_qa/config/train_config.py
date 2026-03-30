import os


class TrainConfig:
    # Dùng model có fast tokenizer để hỗ trợ return_offset_mapping
    MODEL_NAME = "xlm-roberta-base"

    # Text tokenization
    MAX_SEQ_LENGTH = 384
    STRIDE = 128

    # Training hyperparameters
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    WEIGHT_DECAY = 0.01

    # Checkpoint output directory
    OUTPUT_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints",
        "extractive"
    )
