from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, EvaluationConfig, PrepareCallbackConfig, TrainingConfig
import os
from pathlib import Path


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )
        return prepare_base_model_config

    def get_prepare_callback_config(self) -> PrepareCallbackConfig:
        config = self.config.prepare_callbacks
        checkpoint_path = Path(config.checkpoint_model_filepath)

        # Create parent directories using pathlib
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tensorboard_dir = Path(config.tensorboard_root_log_dir)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        return PrepareCallbackConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=tensorboard_dir,
            checkpoint_model_filepath=checkpoint_path
        )

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model

        # Create training root directory
        training_root = Path(training.root_dir)
        training_root.mkdir(parents=True, exist_ok=True)

        # Build paths using pathlib
        training_data = Path(
            self.config.data_ingestion.unzip_dir) / "Chicken-fecal-images"

        return TrainingConfig(
            root_dir=training_root,
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(
                prepare_base_model.updated_base_model_path),
            training_data=training_data,
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_is_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE
        )

    def get_validation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model=Path("artifacts/training/model.keras"),
            training_data=Path(
                "artifacts/data_ingestion/Chicken-fecal-images"),
            all_parameters=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )

        return eval_config
