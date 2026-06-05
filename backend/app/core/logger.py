from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("logs/app_{time}.log", rotation="10 MB", level="DEBUG", backtrace=True, diagnose=True)