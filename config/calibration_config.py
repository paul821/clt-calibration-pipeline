# Libraries to import:
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class CalibrationConfig:
    """Configuration for calibration parameters"""
    
    # Lines 27-29
    T: int = 180
    timesteps_per_day: int = 4
    
    # Lines 31-32 - MISSED THESE!
    torch_seed: int = 0
    numpy_seed: int = 0
    
    # Lines 34 (not 32-34 as I said)
    mode: str = "SEQUENTIAL"  # "BETA_ONLY", "IHR_ONLY", "SEQUENTIAL"
    
    # Line 35
    calibration_method: str = "L-BFGS-B"
    
    # Line 36
    gss_tolerance: float = 1.0
    
    # Line 37 - THIS IS CRITICAL!
    verbose_lbfgs: bool = False  # Surgical Edit: Gating flag for all iteration outputs
    
    # Line 39
    age_labels: List[str] = None
    
    # Lines 41-44
    estimation_config: Dict = None
    
    # Lines 46-51
    reg_config: Dict = None
    
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
