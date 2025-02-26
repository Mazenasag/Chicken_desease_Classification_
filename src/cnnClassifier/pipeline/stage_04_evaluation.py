from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.evaluation import Evaluation
from exception import CustomException
from logger import logging
import sys


STAGE_NAME = "Evaluation"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(val_config)
        evaluation.evaluation()
        evaluation.save_score()


if __name__ == "__main__":
    try:
        logging.info(f">>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logging.info(f">>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<")

    except Exception as e:
        raise CustomException(e, sys)
