# pipeline/stage_02_data_transformation.py

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_transformation import DataTransformation
from textSummarizer.logging.logger import logger

STAGE_NAME = "Data Transformation Stage"


class DataTransformationPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        pipeline = DataTransformationPipeline()
        pipeline.run()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx{'='*50}x")
    except Exception as e:
        logger.exception(f"{STAGE_NAME} failed: {e}")
        raise e