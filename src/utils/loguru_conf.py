from loguru import logger
import sys
import os

def setup_logger():
    os.makedirs("output", exist_ok=True)  # Ensure directory exists
    
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level}</level> | "
            "<level>{message}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
        )
    )
    
    # File handler with rotation (creates a new file every 100 MB)
    logger.add(
        "output/logs.json",
        serialize=True,
        mode="a",
        rotation="10 MB",  # rotate after 10 megabytes
        retention="7 days",  # keep logs for 7 days
        compression="zip"    # compress old logs
    )
    
    return logger

logger = setup_logger()
