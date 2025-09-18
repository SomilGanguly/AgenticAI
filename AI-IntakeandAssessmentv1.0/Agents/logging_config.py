# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

import logging
import sys
from typing import Optional
import os

def configure_logging(log_file_name: Optional[str] = None, logger_name: str = "azureaiapp") -> logging.Logger:
    """
    Configure and return a logger with both stream (stdout) and optional file handlers.

    :param log_file_name: The path to the log file. If provided, logs will also be written to this file.
    :type log_file_name: Optional[str]
    :param logger_name: The name of the logger to configure.
    :type logger_name: str
    :return: The configured logger instance.
    :rtype: logging.Logger
    """
    logger = logging.getLogger(logger_name)
    
    # Only configure if not already configured (check if handlers exist)
    if logger.handlers:
        return logger  # Already configured, return existing logger
    
    # Determine log level from env: APP_VERBOSE=true -> DEBUG
    verbose = os.getenv("APP_VERBOSE", "false").strip().lower() in {"1", "true", "yes", "on", "debug"}
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    # Stream handler (stdout) with UTF-8 encoding support
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    
    # Custom formatter that handles Unicode
    class SafeFormatter(logging.Formatter):
        def format(self, record):
            try:
                return super().format(record)
            except UnicodeEncodeError:
                # Replace problematic characters with safe alternatives
                record.msg = str(record.msg).encode('ascii', 'replace').decode('ascii')
                return super().format(record)
    
    stream_formatter = SafeFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    # File handler if a log file is specified (with UTF-8 encoding)
    if log_file_name:
        file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger