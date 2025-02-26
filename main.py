from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTraningPipeline
from src.cnnClassifier.pipeline.stage_02prepare_base_model import PrepareBaseModelTrainingPipeline
from src.cnnClassifier.pipeline.stage_03_training import ModelTrainingPipeline
from src.cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline
from src.logger import logging
from src.exception import CustomException
import sys

STAGE_NAME = "DATA INGESTION  STAGE"

if __name__ == "__main__":
    try:
        logging.info(f">>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<")
        data_ingestion = DataIngestionTraningPipeline()
        data_ingestion.main()
        logging.info(f">>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<")

    except Exception as e:
        raise CustomException(e, sys)

STAGE_NAME = "Prepare base Model"
if __name__ == "__main__":
    try:
        logging.info(f">>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logging.info(f">>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<")

    except Exception as e:
        raise CustomException(e, sys)


STAGE_NAME = "Training"
if __name__ == "__main__":
    try:
        logging.info(f">>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logging.info(f">>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<")

    except Exception as e:
        raise CustomException(e, sys)


STAGE_NAME = "Evaluation"

if __name__ == "__main__":
    try:
        logging.info(f">>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logging.info(f">>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<")

    except Exception as e:
        raise CustomException(e, sys)
