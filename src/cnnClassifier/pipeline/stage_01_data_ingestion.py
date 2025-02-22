from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from exception import CustomException
from logger import logging
import sys

STAGE_NAME = "DATA INGESTION PIPELINE STAGE"


class DataIngestionTraningPipeline:
    def __init__(self):

        pass

    def main(self):

        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        logging.info(f">>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<")
        obj = DataIngestionTraningPipeline()
        obj.main()
        logging.info(f">>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<")

    except Exception as e:
        raise CustomException(e, sys)
