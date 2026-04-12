# src/textSummarizer/components/data_transformation.py

from transformers import AutoTokenizer
from datasets import load_from_disk, DatasetDict
from pathlib import Path
from textSummarizer.entity.config_entity import DataTransformationConfig
from textSummarizer.logging.logger import logger


class DataTransformation:
    """
    Tokenizes the dialogue and summary columns using the
    Pegasus tokenizer and saves the result to disk as
    a HF DatasetDict ready for the Trainer API.
    """

    def __init__(self, config: DataTransformationConfig):
        self.config = config
        logger.info(f"Loading tokenizer: {config.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        logger.info("Tokenizer loaded successfully.")

    def convert_examples_to_features(self, batch: dict) -> dict:
    """
    Tokenizes a batch of dialogue-summary pairs.
    - Dialogues  → model inputs  (truncated to max_input_length)
    - Summaries  → model labels  (truncated to max_target_length)
    """
    # Tokenize inputs and targets together (modern API)
    model_inputs = self.tokenizer(
        batch["dialogue"],
        text_target=batch["summary"],
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    # Replace padding token id with -100 so loss ignores padding
    model_inputs["labels"] = [
        [(token if token != self.tokenizer.pad_token_id else -100)
         for token in label]
        for label in model_inputs["labels"]
    ]

    return model_inputs

    def convert(self):
        """
        Loads raw dataset from disk, tokenizes all splits,
        removes original text columns, and saves to disk.
        """
        save_path = Path(self.config.root_dir) / "samsum_dataset"

        if save_path.exists():
            logger.info(
                f"Tokenized dataset already exists at {save_path}. Skipping."
            )
            return

        logger.info(f"Loading raw dataset from: {self.config.data_path}")
        dataset = load_from_disk(str(self.config.data_path))
        logger.info(f"Dataset loaded. Splits: {list(dataset.keys())}")

        logger.info("Tokenizing dataset (batched)...")
        tokenized_dataset = dataset.map(
            self.convert_examples_to_features,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        logger.info("Tokenization complete.")

        # Log shapes for verification
        for split in tokenized_dataset.keys():
            size = len(tokenized_dataset[split])
            cols = tokenized_dataset[split].column_names
            logger.info(f"  {split}: {size:,} samples | columns: {cols}")

        logger.info(f"Saving tokenized dataset to: {save_path}")
        tokenized_dataset.save_to_disk(str(save_path))
        logger.info("Tokenized dataset saved successfully.")

    def run(self):
        """Full data transformation pipeline."""
        self.convert()
        logger.info("Data transformation complete.")