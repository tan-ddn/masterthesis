import os
import logging
import time
from pathlib import Path
from torch import cuda

class Log():
    def __init__(self, output_dir : str, prefix : str = 'log_') -> None:
        self.output_dir = output_dir
        unix_time = str(int(time.time()))
        self.log_filename = prefix + unix_time
        
        """Create an output directory if it doesn't exist"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        """Set up logging system"""
        self.log_file = self.output_dir + self.log_filename + ".txt"

        format_str = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        formatter = logging.Formatter(format_str)

        """Set up file log"""
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        """Set up console log"""
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(formatter)

        """Configure logging for model file"""
        self.logger = logging.getLogger("tnguyen_train")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console)
        self.logger.info(f"Training log is set up")

    def pid(self) -> None:
        self.logger.info(f'PID: {os.getpid()}')  # Process ID

    def gpu(self) -> None:
        if cuda.is_available():
            self.logger.info(f"Device count {cuda.device_count()}")
            self.logger.info(f"Current device {cuda.current_device()}")
            self.logger.info(f"Device name {cuda.get_device_name(cuda.current_device())}")
    