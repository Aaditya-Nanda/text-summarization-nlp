# main.py

from textSummarizer.logging.logger import logger
from pipeline.stage_01_data_ingestion import DataIngestionPipeline
from pipeline.stage_02_data_transformation import DataTransformationPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    DataIngestionPipeline().run()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx{'='*50}x")
except Exception as e:
    logger.exception(f"{STAGE_NAME} failed: {e}")
    raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    DataTransformationPipeline().run()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx{'='*50}x")
except Exception as e:
    logger.exception(f"{STAGE_NAME} failed: {e}")
    raise e