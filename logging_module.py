import logging
import sys
import os
import datetime
from parameters import log_dir

log_date = datetime.datetime.today().strftime('%d-%b-%Y')
log_file_name = f'{log_dir}\log_file_{log_date}.log'

formatter = logging.Formatter(
    "%(asctime)s: %(levelname)s: %(name)s: " 
    "%(funcName)s: %(lineno)d - %(message)s"
    )

def get_file_stream_logger(logger_name):
    
    file_stream_logger = logging.getLogger(logger_name)    
    file_stream_logger.setLevel(logging.DEBUG)    
    file_stream_logger.addHandler(get_console_handler())
    file_stream_logger.addHandler(get_file_handler())
    file_stream_logger.propagate = False    
    
    return file_stream_logger

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    return console_handler

def get_file_handler():
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setFormatter(formatter)
    return file_handler
