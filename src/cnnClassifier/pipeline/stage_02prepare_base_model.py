from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import prepareBasedModel
from exception import CustomException
from logger import logging
import sys


STAGE_NAME = "Prepare base Model"


class PreparBasedModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_based_model_config = config.get_prepare_base_model_config()
        prepare_based_model = prepareBasedModel(
            config=prepare_based_model_config)
        prepare_based_model.get_based_model()
        prepare_based_model.update_based_model()


if __name__ == "__main__":
    try:
        logging.info(f">>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<")
        obj = PreparBasedModelTrainingPipeline()
        obj.main()
        logging.info(f">>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<")

    except Exception as e:
        raise CustomException(e, sys)
