import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import stat

def setup_logging():
    # Define the logs directory relative to this file's location
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    
    # Ensure that the logs directory exists with correct permissions
    if not os.path.exists(logs_dir):
        try:
            # Create directory with full permissions
            os.makedirs(logs_dir, mode=0o777, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Unable to create logs directory at {logs_dir}. Error: {e}") from e
    else:
        # If directory exists, ensure it has correct permissions
        try:
            os.chmod(logs_dir, 0o777)
        except Exception as e:
            raise RuntimeError(f"Unable to set permissions for {logs_dir}. Error: {e}") from e

    # Construct log file path
    log_path = os.path.join(logs_dir, f"camera_manager_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Create log file if it doesn't exist and set permissions
    if not os.path.exists(log_path):
        try:
            # Create empty file
            open(log_path, 'a').close()
            # Set permissions to allow read/write for everyone
            os.chmod(log_path, 0o666)
        except Exception as e:
            raise RuntimeError(f"Unable to create or set permissions for log file {log_path}. Error: {e}") from e
    else:
        # If file exists, ensure it has correct permissions
        try:
            os.chmod(log_path, 0o666)
        except Exception as e:
            raise RuntimeError(f"Unable to set permissions for log file {log_path}. Error: {e}") from e

    # Set up rotating file handler
    try:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10*1024*1024,  # 10MB per file
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        file_handler.setFormatter(formatter)
        
        # Create and configure the logger
        logger = logging.getLogger('camera_manager')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        
        return logger
    except Exception as e:
        raise RuntimeError(f"Failed to setup logging handler: {str(e)}") from e 