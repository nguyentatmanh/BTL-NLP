import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from viet_qa.config.train_config import TrainConfig
from viet_qa.data.loader import load_qa_dataset
from viet_qa.data.preprocess import preprocess_extractive


def prepare_train_features(examples, tokenizer, config):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=config.MAX_SEQ_LENGTH,
        stride=config.STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["valid_answers"][sample_index]

        if len(answers) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            answer = answers[0]
            start_char = answer["answer_start"]
            end_char = start_char + len(answer["text"])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def main():
    config = TrainConfig()

    print(f"Loading tokenizer and model: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError(
            f"{config.MODEL_NAME} does not provide a fast tokenizer. "
            "This extractive QA pipeline requires offset_mapping."
        )

    model = AutoModelForQuestionAnswering.from_pretrained(config.MODEL_NAME)

    print("Loading datasets...")
    dataset = load_qa_dataset("all")  

    train_ds = dataset["train"].map(preprocess_extractive, load_from_cache_file=False)
    train_ds = train_ds.filter(lambda x: len(x["valid_answers"]) > 0, load_from_cache_file=False)
    print(f"  Train samples after filter: {len(train_ds)}")

    val_ds = dataset["validation"].map(preprocess_extractive, load_from_cache_file=False)
    val_ds = val_ds.filter(lambda x: len(x["valid_answers"]) > 0, load_from_cache_file=False)
    print(f"  Val samples after filter: {len(val_ds)}")

    print("Tokenizing train dataset...")
    train_tokenized = train_ds.map(
        lambda x: prepare_train_features(x, tokenizer, config),
        batched=True,
        remove_columns=train_ds.column_names,
        load_from_cache_file=False,
        desc="Tokenizing train",
    )
    print(f"  Train tokenized samples: {len(train_tokenized)}")

    print("Tokenizing val dataset...")
    val_tokenized = val_ds.map(
        lambda x: prepare_train_features(x, tokenizer, config),
        batched=True,
        remove_columns=val_ds.column_names,
        load_from_cache_file=False,
        desc="Tokenizing val",
    )
    print(f"  Val tokenized samples: {len(val_tokenized)}")

    if len(train_tokenized) == 0:
        raise ValueError(
            "Train dataset is empty after tokenization! "
            "Check prepare_train_features for errors or verify dataset structure."
        )

    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.NUM_EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {config.OUTPUT_DIR}")
    trainer.save_model(config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
