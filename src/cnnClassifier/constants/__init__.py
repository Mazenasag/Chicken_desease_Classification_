from pathlib import Path

# Define the file paths as constants.
# These paths point to configuration files that are read once and should not change.
# Path to the main configuration file
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")         # Path to the parameters file

# The __all__ variable defines the public API of this module.
# It restricts what gets imported when using `from constants import *`
# __all__ = ["CONFIG_FILE_PATH", "PARAMS_FILE_PATH"]
