import logging
import os
import sys
import datetime
source = 'log'
log_str = '[%(asctime)s : %(levelname)s : %(module)s : %(message)s]'
log_file = os.path.join(source,f"{datetime.date.today().isoformat()}.log")
os.makedirs(source,exist_ok=True)

logging.basicConfig(
    format=log_str,
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
