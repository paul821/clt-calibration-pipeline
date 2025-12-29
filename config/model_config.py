# Libraries to import:
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

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
    
    scale_factors: Dict[str, float] = field(default_factory=lambda: {
        "beta": 0.01,  # CRITICAL FIX: was 1.0, causing 100x underestimation
        "E": 0.1,      # CRITICAL FIX: was 1.0
        "IP": 0.1,     # CRITICAL FIX: was 1.0
        "ISR": 0.1,    # CRITICAL FIX: was 1.0
        "ISH": 0.1,    # CRITICAL FIX: was 1.0
        "IA": 0.1,     # CRITICAL FIX: was 1.0
        "ihr": 0.01    # CRITICAL FIX: was 1.0
    })
    
    def __post_init__(self):
        if self.subpop_names is None:
            self.subpop_names = ["subpopA", "subpopB", "subpopC"]
        
        if self.beta_values is None:
            self.beta_values = [0.22, 0.28, 0.25]
