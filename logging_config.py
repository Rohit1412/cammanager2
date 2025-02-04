import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import stat

def setup_logging():
    logger = logging.getLogger('camera_manager')
    
    # Clear any existing handlers
    logger.handlers = []
    
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler for debugging
    file_handler = logging.FileHandler('camera_debug.log')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Prevent logging from propagating to the root logger
    logger.propagate = False
    
    return logger 