from __future__ import annotations

from dotenv import load_dotenv
from loguru import logger
from rich.logging import RichHandler

load_dotenv()

logger.remove()
logger.add(
    RichHandler(rich_tracebacks=True, markup=False),
    format="{message}",
    level="INFO",
)

__all__ = ["logger"]
