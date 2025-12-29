#!/usr/bin/env python3
"""
Example: Basic calibration with default settings
"""

import sys
from pathlib import Path
from datetime import datetime
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.calibration_config import CalibrationConfig, RegularizationConfig
from config.model_config import ModelConfig
from scripts.run_calibration import run_calibration
from src.utils.logger import start_logging, stop_logging

def main():
    """
    Basic calibration example: Beta estimation only, single optimizer
    """
    # Create configuration
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = start_logging(f"basic_calibration_log_{timestamp}.txt")
    
    try:
        calib_config = CalibrationConfig(
            T=180,
            timesteps_per_day=4,
            mode="BETA_ONLY",
            optimizers=["L-BFGS-B"],
            verbose_lbfgs=False,
            verbosity=1,
            loss_aggregation="regional",
            noise_type="poisson"
        )
        
        # Set estimation config
        calib_config.estimation_config = {
            "beta_param": "L",
            "estimate_initial": {
                "E": False,
                "IP": False,
                "ISR": False,
                "ISH": False,
                "IA": False
            }
        }
        
        # Regularization (L2 on beta only)
        calib_config.regularization = RegularizationConfig(
            beta_type="l2_magnitude",
            beta_lambda=1e-2,
            compartment_configs={}
        )
        
        model_config = ModelConfig()
        
        print("="*100)
        print("BASIC CALIBRATION EXAMPLE")
        print("="*100)
        print(f"Mode: {calib_config.mode}")
        print(f"Optimizers: {calib_config.optimizers}")
        print(f"Estimating: Beta only")
        print(f"Noise: {calib_config.noise_type}")
        print("="*100)
        
        # Run calibration
        stage1_results, stage2_results = run_calibration(calib_config, model_config)
        
        print("\nExample complete!")
    
    finally:
        stop_logging(logger)

if __name__ == "__main__":
    main()