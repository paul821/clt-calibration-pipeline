from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class RegularizationConfig:
    """Configuration for regularization penalties"""
    
    # Beta regularization (magnitude-based)
    beta_type: str = "l2_magnitude"  # "l2_magnitude" or "none"
    beta_lambda: float = 1e-6  #1e-2 was too large for log-space
    
    # Compartment regularization
    compartment_configs: Dict[str, Dict] = field(default_factory=dict)

@dataclass
class CalibrationConfig:
    """Configuration for calibration parameters"""
    
    T: int = 180
    timesteps_per_day: int = 4
    
    torch_seed: int = 0
    numpy_seed: int = 0
    
    # Calibration mode
    mode: str = "SEQUENTIAL"  # "BETA_ONLY", "IHR_ONLY", "SEQUENTIAL", "IHR_MODE"
    
    optimizers: List[str] = field(default_factory=lambda: ["L-BFGS-B"])
    # Options: "L-BFGS-B", "CG", "Adam", "least_squares_fd"
    
    # GSS parameters (only used in IHR_MODE)
    gss_tolerance: float = 1.0
    gss_offset_range: tuple = (-30, 15)  # (min_offset, max_offset) in days
    enable_gss: bool = True  # Set False to disable offset search
    
    # Optimization parameters
    verbose_lbfgs: bool = False
    verbosity: int = 1  # 0=silent, 1=summary, 2=detailed, 3=debug
    early_stop_r2: float = 0.90
    early_stop_patience: int = 10
    
    # Loss aggregation mode
    loss_aggregation: str = "regional"  # "regional", "location_age", "global"
    # "regional": sum of per-location SSE (Stage 1 style)
    # "location_age": sum of per-(location,age) SSE (Stage 2 style)
    # "global": single global SSE (legacy)
    
    # Time stretch estimation
    apply_time_stretch: bool = False
    time_stretch_factor: float = 1.0  
    estimate_time_stretch: bool = False  
    time_stretch_bounds: tuple = (0.5, 2.0)
    
    # Noise injection for robustness testing
    noise_type: str = "none"  # "none", "poisson", "gaussian"
    noise_seed: int = 12345
    
    # Age group labels
    age_labels: List[str] = field(default_factory=lambda: ["0-4", "5-17", "18-49", "50-64", "65+"])
    
    # Estimation configuration
    estimation_config: Dict = field(default_factory=dict)
    
    # Regularization
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    
    # Restart strategy (your approach)
    num_wide_restarts: int = 2
    num_medium_restarts: int = 2
    num_narrow_restarts: int = 2
    restart_widths: Dict[str, float] = field(default_factory=lambda: {
        "Wide Search": 0.75,
        "Medium Search": 0.50,
        "Narrow Search": 0.25
    })
    
    # IHR mode specific
    ihr_restart_widths: Dict[str, float] = field(default_factory=lambda: {
        "Wide Search": 0.25,
        "Medium Search": 0.15,
        "Narrow Search": 0.05
    })
    
    output_prefix: str = ""
    
    def __post_init__(self):
        if not self.estimation_config:
            self.estimation_config = {
                "beta_param": "L",
                "estimate_initial": {
                    "E": False, 
                    "IP": False, 
                    "ISR": False, 
                    "ISH": False, 
                    "IA": False
                },
            }
        
        if not self.output_prefix:
            self.output_prefix = f"{self.mode}_"
        
        # Set IHR-specific settings if in IHR_MODE
        if self.mode == "IHR_MODE":
            self.loss_aggregation = "regional"  # Force regional for Stage 1
            if "ihr_param" not in self.estimation_config:
                self.estimation_config["ihr_param"] = "LA"
