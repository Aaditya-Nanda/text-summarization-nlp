# pipeline/stage_04_model_evaluation.py

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_evaluation import ModelEvaluation
from textSummarizer.logging.logger import logger

STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        pipeline = ModelEvaluationPipeline()
        pipeline.run()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx{'='*50}x")
    except Exception as e:
        logger.exception(f"{STAGE_NAME} failed: {e}")
        raise e