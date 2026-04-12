# src/textSummarizer/config/configuration.py

from pathlib import Path
from textSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from textSummarizer.utils.common import read_yaml, create_directories
from textSummarizer.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)
from textSummarizer.logging.logger import logger


class ConfigurationManager:
    """
    Reads config.yaml and params.yaml once.
    Exposes typed config objects for each pipeline component.
    """

    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # Ensure top-level artifacts directory exists
        create_directories([self.config.artifacts_root])
        logger.info("ConfigurationManager initialized.")

    # ------------------------------------------------------------------ #
    # Data Ingestion                                                       #
    # ------------------------------------------------------------------ #
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config.data_ingestion
        create_directories([cfg.root_dir])

        return DataIngestionConfig(
            root_dir=Path(cfg.root_dir),
            dataset_name=cfg.dataset_name,
            dataset_save_path=Path(cfg.dataset_save_path),
        )

    # ------------------------------------------------------------------ #
    # Data Transformation                                                  #
    # ------------------------------------------------------------------ #
    def get_data_transformation_config(self) -> DataTransformationConfig:
        cfg = self.config.data_transformation
        create_directories([cfg.root_dir])

        return DataTransformationConfig(
            root_dir=Path(cfg.root_dir),
            tokenizer_name=cfg.tokenizer_name,
            data_path=Path(cfg.data_path),
        )

    # ------------------------------------------------------------------ #
    # Model Trainer                                                        #
    # ------------------------------------------------------------------ #
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        cfg = self.config.model_trainer
        params = self.params.TrainingArguments
        data_params = self.params.DataArguments
        create_directories([cfg.root_dir])

        return ModelTrainerConfig(
            root_dir=Path(cfg.root_dir),
            data_path=Path(cfg.data_path),
            model_ckpt=cfg.model_ckpt,
            tokenizer_save_path=Path(cfg.tokenizer_save_path),
            model_save_path=Path(cfg.model_save_path),
            # Training hyperparams
            num_train_epochs=params.num_train_epochs,
            warmup_steps=params.warmup_steps,
            per_device_train_batch_size=params.per_device_train_batch_size,
            per_device_eval_batch_size=params.per_device_eval_batch_size,
            weight_decay=params.weight_decay,
            logging_steps=params.logging_steps,
            evaluation_strategy=params.evaluation_strategy,
            eval_steps=params.eval_steps,
            save_steps=params.save_steps,
            gradient_accumulation_steps=params.gradient_accumulation_steps,
            fp16=params.fp16,
            push_to_hub=params.push_to_hub,
            hub_model_id=params.hub_model_id,
            # Data params
            train_subset_size=data_params.train_subset_size,
            val_subset_size=data_params.val_subset_size,
            max_input_length=data_params.max_input_length,
            max_target_length=data_params.max_target_length,
            source_column=data_params.source_column,
            target_column=data_params.target_column,
        )

    # ------------------------------------------------------------------ #
    # Model Evaluation                                                     #
    # ------------------------------------------------------------------ #
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        cfg = self.config.model_evaluation
        data_params = self.params.DataArguments
        create_directories([cfg.root_dir])

        return ModelEvaluationConfig(
            root_dir=Path(cfg.root_dir),
            data_path=Path(cfg.data_path),
            model_path=Path(cfg.model_path),
            tokenizer_path=Path(cfg.tokenizer_path),
            metric_file_name=Path(cfg.metric_file_name),
            max_input_length=data_params.max_input_length,
            max_target_length=data_params.max_target_length,
            source_column=data_params.source_column,
            target_column=data_params.target_column,
        )