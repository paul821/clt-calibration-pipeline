#!/usr/bin/env python3
"""
Example: Basic calibration with custom settings
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.calibration_config import CalibrationConfig
from config.model_config import ModelConfig
from scripts.run_calibration import run_calibration

def main():
    # Create custom configuration
    calib_config = CalibrationConfig(
        T=180,
        timesteps_per_day=4,
        mode="SEQUENTIAL",
        verbose_lbfgs=True,  # Enable verbose output
        gss_tolerance=0.5    # Tighter tolerance for GSS
    )
    
    model_config = ModelConfig(
        beta_values=[0.22, 0.28, 0.25],
        seed_region_idx=1,
        seed_age_idx=2
    )
    
    # Run calibration
    final_state, final_params, predictions = run_calibration(calib_config, model_config)
    
    print("\nExample calibration complete!")

if __name__ == "__main__":
    main()
