# Libraries to import:
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class RegularizationConfig:
    """Configuration for regularization penalties"""
    
    # Beta regularization (magnitude-based)
    beta_type: str = "l2_magnitude"  # "l2_magnitude" or "none"
    beta_lambda: float = 1e-6  # CRITICAL FIX: was 1e-2, too large for log-space
    
    # Compartment regularization (can be structural or magnitude)
    compartment_configs: Dict[str, Dict] = field(default_factory=dict)
    # Example:
    # {
    #     "E": {
    #         "type": "structural",
    #         "location_targets": [0.0, 1.0, 0.0],  # which locations seeded
    #         "age_targets": [0, 0, 1, 0, 0],       # which ages seeded (one-hot)
    #         "lambda_on_target": 100000.0,  # CRITICAL FIX: was 10.0, had no effect
    #         "lambda_off_target": 100000.0  # CRITICAL FIX: was 10.0, had no effect
    #     },
    #     "IP": {
    #         "type": "l2_magnitude",
    #         "lambda": 1e-3
    #     }
    # }

@dataclass
class CalibrationConfig:
    """Configuration for calibration parameters"""
    
    # From lines 27-29 (professor's code)
    T: int = 180
    timesteps_per_day: int = 4
    
    # Random seeds
    torch_seed: int = 0
    numpy_seed: int = 0
    
    # Calibration mode
    mode: str = "SEQUENTIAL"  # "BETA_ONLY", "IHR_ONLY", "SEQUENTIAL", "IHR_MODE"
    
    # IHR_MODE: Uses professor's GSS + regional loss approach
    # SEQUENTIAL: Your multi-optimizer suite approach
    
    # Optimizer selection
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
    
    # Time stretching (professor's approach)
    time_stretch_factor: float = 1.0  # Set to 5.0 for professor's time-stretching
    apply_time_stretch: bool = False
    
    # Noise injection for robustness testing
    noise_type: str = "none"  # "none", "poisson", "gaussian"
    noise_seed: int = 12345
    
    # Age group labels
    age_labels: List[str] = field(default_factory=lambda: ["0-4", "5-17", "18-49", "50-64", "65+"])
    
    # Estimation configuration
    estimation_config: Dict = field(default_factory=dict)
    # {
    #     "beta_param": "L" or "LA",
    #     "estimate_initial": {"E": True, "IP": False, ...},
    #     "ihr_param": "L" or "LAR"  (only for IHR mode)
    # }
    
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