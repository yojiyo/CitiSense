# app/logging_config.py
import logging
import sys
from pathlib import Path

def setup_logging():
    """Set up logging configuration for production."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'citisense.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logger for the application
    logger = logging.getLogger('citisense')
    return logger 