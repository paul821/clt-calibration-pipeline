#!/usr/bin/env python3
"""
Example: IHR MODE calibration (Professor's GSS approach with multi-optimizer)
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
    IHR MODE example:
    - Stage 1: Beta + E0 with GSS offset discovery
    - Stage 2: IHR estimation
    - Multiple optimizers: L-BFGS-B and CG
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = start_logging(f"ihr_mode_log_{timestamp}.txt")
    
    try:
        # Create configuration
        calib_config = CalibrationConfig(
            T=180,
            timesteps_per_day=4,
            mode="IHR_MODE",
            optimizers=["L-BFGS-B", "CG"],  # Compare both optimizers
            verbose_lbfgs=False,
            verbosity=2,  # Detailed output
            loss_aggregation="regional",  # Stage 1 uses regional
            noise_type="poisson",
            enable_gss=True,
            gss_tolerance=1.0,
            gss_offset_range=(-30, 15)
        )
        
        # Estimation config: Beta + E0
        calib_config.estimation_config = {
            "beta_param": "L",
            "estimate_initial": {
                "E": True,   # Estimate initial exposed
                "IP": False,
                "ISR": False,
                "ISH": False,
                "IA": False
            },
            "ihr_param": "LA"  # Per-location IHR in Stage 2
        }
        
        # Structured regularization for E0 (prevent shadow solutions)
        calib_config.regularization = RegularizationConfig(
            beta_type="l2_magnitude",
            beta_lambda=1e-6,
            compartment_configs={
                "E": {
                    "type": "structural",
                    "location_targets": [0.0, 1.0, 0.0],  # Seed only in location 1
                    "age_targets": [0, 0, 1, 0, 0],       # Only age group 2 (18-49)
                    "lambda_on_target": 100000.0,  # CRITICAL FIX: was 10.0
                    "lambda_off_target": 100000.0  # CRITICAL FIX: was 10.0
                }
            }
        )
        
        model_config = ModelConfig(
            seed_region_idx=1,
            seed_age_idx=2,
            seed_value=1.0
        )
        
        print("="*100)
        print("IHR MODE CALIBRATION EXAMPLE")
        print("="*100)
        print(f"Mode: {calib_config.mode}")
        print(f"Optimizers: {calib_config.optimizers}")
        print(f"Stage 1: Beta + E0 with GSS offset discovery")
        print(f"Stage 2: IHR estimation (per-location)")
        print(f"Noise: {calib_config.noise_type}")
        print(f"GSS enabled: {calib_config.enable_gss}")
        print(f"Offset range: {calib_config.gss_offset_range}")
        print("="*100)
        
        # Run calibration
        stage1_results, stage2_results = run_calibration(calib_config, model_config)
        
        print("\nIHR MODE example complete!")
        print("\nGenerated files:")
        print("  - stage1_optimizer_comparison.png")
        print("  - stage1_loss_breakdown.png")
        print("  - stage1_parameter_recovery.png")
        print("  - stage2_optimizer_comparison.png")
        print("  - calibration_15_panel.png")
        print("  - final_calibration_regional_aggregate.png")
        print("  - SSE_Stage1_Convergence_*.png (per optimizer)")
    
    finally:
        stop_logging(logger)

if __name__ == "__main__":
    main()