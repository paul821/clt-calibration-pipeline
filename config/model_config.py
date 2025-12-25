# Libraries to import:
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for model paths and parameters"""
    
    # Paths (from lines 414-415)
    texas_input_path: Optional[Path] = None
    calibration_input_path: Optional[Path] = None
    
    timesteps_per_day: int = 4
    
    # Subpopulation names and beta values (from lines 418-419)
    subpop_names: list = None
    beta_values: list = None
    
    # Initial seed location (from line 421)
    seed_region_idx: int = 1
    seed_age_idx: int = 2
    seed_value: float = 1.0
    
    def __post_init__(self):
        if self.subpop_names is None:
            self.subpop_names = ["subpopA", "subpopB", "subpopC"]
        
        if self.beta_values is None:
            self.beta_values = [0.22, 0.28, 0.25]
