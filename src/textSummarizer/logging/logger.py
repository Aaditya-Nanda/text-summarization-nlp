# src/textSummarizer/logging/logger.py

import os
import sys
import logging

LOG_DIR = "logs"
LOG_FILE = "running_logs.log"

os.makedirs(LOG_DIR, exist_ok=True)
log_filepath = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("textSummarizerLogger")