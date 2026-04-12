# src/textSummarizer/entity/config_entity.py

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    dataset_save_path: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    tokenizer_name: str
    data_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: str
    tokenizer_save_path: Path
    model_save_path: Path
    # Training hyperparameters (injected from params.yaml)
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_steps: float
    gradient_accumulation_steps: int
    fp16: bool
    push_to_hub: bool
    hub_model_id: str
    train_subset_size: int
    val_subset_size: int
    max_input_length: int
    max_target_length: int
    source_column: str
    target_column: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path
    max_input_length: int
    max_target_length: int
    source_column: str
    target_column: str