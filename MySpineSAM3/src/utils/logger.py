"""
TensorBoard Logger
==================
Wrapper for TensorBoard logging.
"""

from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """TensorBoard logging wrapper."""
    
    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter
                    self.writer = SummaryWriter(log_dir)
                except ImportError:
                    logger.warning("TensorBoard not available")
                    self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer: self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, values: Dict[str, float], step: int):
        if self.writer: self.writer.add_scalars(main_tag, values, step)
    
    def log_image(self, tag: str, image, step: int):
        if self.writer: self.writer.add_image(tag, image, step)
    
    def log_histogram(self, tag: str, values, step: int):
        if self.writer: self.writer.add_histogram(tag, values, step)
    
    def close(self):
        if self.writer: self.writer.close()
