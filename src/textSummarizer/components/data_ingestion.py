# src/textSummarizer/components/data_ingestion.py

from datasets import load_dataset, load_from_disk
from pathlib import Path
from textSummarizer.entity.config_entity import DataIngestionConfig
from textSummarizer.logging.logger import logger


class DataIngestion:
    """
    Loads the SAMSum dataset from Hugging Face Hub and saves it
    to disk for all downstream pipeline stages.
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_and_save_dataset(self):
        """
        Downloads the dataset from HF Hub if not already cached,
        then saves it to disk in Arrow format.
        """
        save_path = Path(self.config.dataset_save_path)

        if save_path.exists():
            logger.info(
                f"Dataset already exists at {save_path}. Skipping download."
            )
            return

        logger.info(f"Loading dataset: '{self.config.dataset_name}' from HF Hub...")
        dataset = load_dataset(self.config.dataset_name, trust_remote_code=True)

        logger.info(f"Saving dataset to disk at: {save_path}")
        dataset.save_to_disk(str(save_path))
        logger.info("Dataset saved successfully.")

    def validate_dataset(self):
        """
        Loads the saved dataset from disk and validates:
        - All expected splits are present (train / validation / test)
        - Required columns exist (dialogue, summary)
        - No empty rows in key columns
        """
        logger.info("Validating dataset...")
        dataset = load_from_disk(str(self.config.dataset_save_path))

        # --- Split validation ---
        expected_splits = {"train", "validation", "test"}
        actual_splits = set(dataset.keys())
        missing = expected_splits - actual_splits
        if missing:
            raise ValueError(f"Missing dataset splits: {missing}")
        logger.info(f"Splits found: {list(actual_splits)}")

        # --- Column validation ---
        required_columns = {"dialogue", "summary"}
        for split in expected_splits:
            actual_cols = set(dataset[split].column_names)
            missing_cols = required_columns - actual_cols
            if missing_cols:
                raise ValueError(
                    f"Split '{split}' is missing columns: {missing_cols}"
                )

        # --- Size validation ---
        for split in expected_splits:
            size = len(dataset[split])
            logger.info(f"  {split}: {size:,} samples")
            if size == 0:
                raise ValueError(f"Split '{split}' is empty.")

        # --- Null check on key columns ---
        for split in expected_splits:
            for col in required_columns:
                null_count = sum(
                    1 for x in dataset[split][col]
                    if x is None or str(x).strip() == ""
                )
                if null_count > 0:
                    logger.warning(
                        f"  [{split}][{col}] has {null_count} null/empty rows."
                    )

        logger.info("Dataset validation passed.")
        return dataset

    def run(self):
        """Full data ingestion pipeline: download → validate."""
        self.download_and_save_dataset()
        dataset = self.validate_dataset()
        logger.info("Data ingestion complete.")
        return dataset
