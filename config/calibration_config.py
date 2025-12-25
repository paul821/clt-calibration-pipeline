# Libraries to import:
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class CalibrationConfig:
    """Configuration for calibration parameters"""
    
    # From lines 27-29
    T: int = 180
    timesteps_per_day: int = 4
    
    # From lines 32-34
    mode: str = "SEQUENTIAL"  # Options: "BETA_ONLY", "IHR_ONLY", "SEQUENTIAL"
    calibration_method: str = "L-BFGS-B"
    gss_tolerance: float = 1.0
    verbose_lbfgs: bool = False
    
    # From lines 36
    age_labels: List[str] = None
    
    # From lines 38-41
    estimation_config: Dict = None
    
    # From lines 43-48
    reg_config: Dict = None
    
    # From lines 407-408 (random seeds)
    torch_seed: int = 0
    numpy_seed: int = 0
    
    def __post_init__(self):
        if self.age_labels is None:
            self.age_labels = ["0-4", "5-17", "18-49", "50-64", "65+"]
        
        if self.estimation_config is None:
            self.estimation_config = {
                "beta_param": "L",
                "estimate_initial": {"E": True, "IP": False, "ISR": False, "ISH": False, "IA": False},
            }
        
        if self.reg_config is None:
            self.reg_config = {
                "lambda_e0_zero": 10.0,
                "lambda_e0_target": 10.0,
                "target_e0_values": [0.0, 1.0, 0.0],
                "target_age_idx": 2
            }
