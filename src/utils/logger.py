import sys
from pathlib import Path
from datetime import datetime

class ConsoleLogger:
    """
    Logger that writes to both console and file
    """
    def __init__(self, log_file=None):
        self.terminal = sys.stdout
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"calibration_log_{timestamp}.txt"
        self.log_file = Path(log_file)
        self.file = open(self.log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure immediate write
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def start_logging(log_file=None):
    """Start logging console output to file"""
    logger = ConsoleLogger(log_file)
    sys.stdout = logger
    return logger

def stop_logging(logger):
    """Stop logging and restore normal console output"""
    sys.stdout = logger.terminal
    logger.close()
    print(f"\nLog saved to: {logger.log_file}")
