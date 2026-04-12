# src/textSummarizer/utils/common.py

import os
import yaml
import json
import joblib
from box import ConfigBox
from pathlib import Path
from typing import Any
from ensure import ensure_annotations
from textSummarizer.logging.logger import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns a ConfigBox object
    for dot-notation access (config.model_trainer.root_dir).
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully: {path_to_yaml}")
            return ConfigBox(content)
    except Exception as e:
        raise ValueError(f"Error reading YAML file at {path_to_yaml}: {e}")


@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """Creates a list of directories if they don't already exist."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created (or already exists): {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves a dictionary as a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads a JSON file and returns as ConfigBox."""
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON loaded from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_object(path: Path, obj: Any):
    """Serializes a Python object using joblib."""
    joblib.dump(obj, path)
    logger.info(f"Object saved at: {path}")


@ensure_annotations
def load_object(path: Path) -> Any:
    """Loads a serialized object from disk."""
    obj = joblib.load(path)
    logger.info(f"Object loaded from: {path}")
    return obj


def get_size_in_kb(path: Path) -> str:
    """Returns file size in KB as a formatted string."""
    size = round(os.path.getsize(path) / 1024)
    return f"~ {size} KB"