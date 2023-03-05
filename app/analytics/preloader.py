import sys
import os
import logging
if os.path.exists(f"{os.environ.get('shared_app_path')}/_vendor"):
    sys.path.append(f"{os.environ.get('shared_app_path')}/_vendor")
    logging.info(f"{os.environ.get('shared_app_path')}/_vendor")