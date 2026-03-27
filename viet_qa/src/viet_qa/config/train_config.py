import os

class TrainConfig:
    # Model configuration
    # Using PhoBERT as the base encoder. 
    # Can also be "nguyenvulebinh/vi-mrc-base" for further fine-tuning.
    MODEL_NAME = "vinai/phobert-base-v2"  
    
    # Text Tokenization configs
    MAX_SEQ_LENGTH = 384
    STRIDE = 128
    
    # Training hyperparameters
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    BATCH_SIZE = 16
    WEIGHT_DECAY = 0.01
    
    # Checkpoint output directory
    OUTPUT_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "checkpoints", 
        "extractive"
    )
