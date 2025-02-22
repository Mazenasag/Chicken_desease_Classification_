from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTraningPipeline
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
