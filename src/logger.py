import os
import sys
from datetime import datetime
import logging
# Generate the Log File

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"


# Define logs directory (with out the File name)
logs_dir = os.path.join(os.getcwd(), "logs")

# Create logs Directory if it dose not exsit

os.makedirs(logs_dir, exist_ok=True)

# Full log file path
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)
# ✅ This ensures we get "my_module" in logs

logging.basicConfig(
    # filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(filename)s - %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,  # Ensure it overwrites any previous logging settings
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)

    ]
)
# logger.setLevel(logging.INFO)
