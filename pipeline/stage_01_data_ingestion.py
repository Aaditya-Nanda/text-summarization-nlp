# pipeline/stage_01_data_ingestion.py

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_ingestion import DataIngestion
from textSummarizer.logging.logger import logger

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        pipeline = DataIngestionPipeline()
        pipeline.run()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx{'='*50}x")
    except Exception as e:
        logger.exception(f"{STAGE_NAME} failed: {e}")
        raise e